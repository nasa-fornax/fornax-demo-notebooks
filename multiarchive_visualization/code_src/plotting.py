"""
Interactive plotting functions for multi-wavelength images.

This module provides classes for creating interactive RGB composites
and multi-panel stretches using HoloViz (Panel, HoloViews, Datashader).
"""
import os
import gc
import asyncio
import logging
import numpy as np
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import regrid
from holoviews.streams import Pipe
from astropy.visualization import (
    AsymmetricPercentileInterval,
    LinearStretch, LogStretch, SqrtStretch,
    ImageNormalize
)
from image_processing import create_rgb_composite

# Configure logging for Fornax diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FornaxPlotting')

# Silences the OpenMP warning
os.environ['KMP_WARNINGS'] = '0'

# Default visualization constants
RGB_PANEL_WIDTH = 700
RGB_PANEL_HEIGHT = 700
MULTI_PANEL_WIDTH = 337
MULTI_PANEL_HEIGHT = 337
CONTROL_PANEL_WIDTH = 250

DEFAULT_RED_PERC = 95
DEFAULT_GREEN_PERC = 95
DEFAULT_BLUE_PERC = 99.5
DEFAULT_Q = 8
DEFAULT_STRETCH = 0.1


class InteractiveRGBPanel:
    """
    An interactive Panel dashboard for creating RGB composite images from 3 bands.

    Parameters
    ----------
    red_data : numpy.ndarray
        Array containing the red channel data.
    green_data : numpy.ndarray
        Array containing the green channel data.
    blue_data : numpy.ndarray
        Array containing the blue channel data.
    """

    def __init__(self, red_data, green_data, blue_data):
        self.r = np.nan_to_num(red_data, nan=0)
        self.g = np.nan_to_num(green_data, nan=0)
        self.b = np.nan_to_num(blue_data, nan=0)

        self.width = self.r.shape[1]
        self.height = self.r.shape[0]

        # Initialize sliders
        self.q_slider = pn.widgets.FloatSlider(
            name='Lupton Q', start=0.1, end=20, value=DEFAULT_Q, step=0.5
        )
        self.stretch_slider = pn.widgets.FloatSlider(
            name='Stretch', start=0.01, end=2, value=DEFAULT_STRETCH, step=0.05
        )
        self.r_perc = pn.widgets.FloatSlider(
            name='Red %', start=50, end=100, value=DEFAULT_RED_PERC, step=0.5
        )
        self.g_perc = pn.widgets.FloatSlider(
            name='Green %', start=50, end=100, value=DEFAULT_GREEN_PERC, step=0.5
        )
        self.b_perc = pn.widgets.FloatSlider(
            name='Blue %', start=50, end=100, value=DEFAULT_BLUE_PERC, step=0.5
        )

        # Diagnostic pane for providing explicit feedback to the user
        self.status_pane = pn.pane.Markdown("System: Ready", visible=False)

        # Placeholder for the main image pane (to handle spinner status)
        self.image_pane = None

        # 1. Compute initial data and setup the Pipe
        initial_rgb = create_rgb_composite(
            self.r, self.g, self.b,
            red_percentile=DEFAULT_RED_PERC,
            green_percentile=DEFAULT_GREEN_PERC,
            blue_percentile=DEFAULT_BLUE_PERC,
            Q=DEFAULT_Q, stretch=DEFAULT_STRETCH
        )
        self.pipe = Pipe(data=initial_rgb)

        # 2. DynamicMap only reads from the Pipe
        def render_rgb(data):
            try:
                return hv.RGB(data, bounds=(0, 0, self.width, self.height))
            except Exception as e:
                logger.exception("Error rendering RGB frame")
                self.status_pane.object = f"### System Status\nError rendering frame: {type(e).__name__}"
                self.status_pane.visible = True
                raise e

        self.dmap = hv.DynamicMap(render_rgb, streams=[self.pipe])

        self.interactive_plot = regrid(self.dmap).opts(
            width=RGB_PANEL_WIDTH, height=RGB_PANEL_HEIGHT,
            tools=['pan', 'box_zoom', 'wheel_zoom', 'reset', 'save'],
            default_tools=[],
            active_tools=['wheel_zoom', 'box_zoom'],
            xlim=(0, self.width), ylim=(0, self.height),
            xaxis=None, yaxis=None
        )

    async def process_and_push(self, *_):
        """
        Asynchronously computes the new RGB composite and pushes it to the Pipe.
        Manages the loading spinner on the image pane.
        """
        try:
            if self.image_pane:
                self.image_pane.loading = True

            await asyncio.sleep(0.05)  # Allow UI to render spinner

            new_rgb = create_rgb_composite(
                self.r, self.g, self.b,
                red_percentile=self.r_perc.value,
                green_percentile=self.g_perc.value,
                blue_percentile=self.b_perc.value,
                Q=self.q_slider.value,
                stretch=self.stretch_slider.value
            )

            self.pipe.send(new_rgb)

            # Explicitly clean up large array to help garbage collection
            del new_rgb
            gc.collect()

        except Exception as e:
            logger.exception("Internal error in process_and_push")
            self.status_pane.object = (
                "### System Status\n"
                f"An internal error occurred during processing: {type(e).__name__}. "
                "Please check the kernel logs for more information."
            )
            self.status_pane.visible = True
        finally:
            if self.image_pane:
                self.image_pane.loading = False

    def view(self):
        """
        Returns the Panel layout for the dashboard.

        Returns
        -------
        panel.Row
            A row containing the controls and the main image plot.
        """
        controls = pn.Column(
            self.q_slider, self.stretch_slider,
            self.r_perc, self.g_perc, self.b_perc,
            self.status_pane,
            width=CONTROL_PANEL_WIDTH
        )

        self.image_pane = pn.pane.HoloViews(self.interactive_plot)

        # Trigger the async process on slider release
        sliders = [self.q_slider, self.stretch_slider, self.r_perc, self.g_perc, self.b_perc]
        for slider in sliders:
            slider.param.watch(self.process_and_push, 'value_throttled')

        return pn.Row(controls, self.image_pane)


class InteractiveMultiPanel:
    """
    An interactive Panel dashboard for multi-panel multi-wavelength visualization.
    Uses independent DynamicMaps in a linked Layout for stability.

    Parameters
    ----------
    reprojected_data : dict
        A dictionary mapping mission names to 2D numpy arrays.
    cmaps : dict
        A dictionary mapping mission names to colormap strings.
    """

    def __init__(self, reprojected_data, cmaps):
        # NaNs replaced with 0.0 to restore the original visual style
        self.data_dict = {k: np.nan_to_num(v, nan=0.0) for k, v in reprojected_data.items()}
        self.cmaps = cmaps

        first_img = list(self.data_dict.values())[0]
        self.height, self.width = first_img.shape

        self.perc_sliders = {}
        self.stretch_widgets = {}

        # Diagnostic pane for providing explicit feedback
        self.status_pane = pn.pane.Markdown("System: Ready", visible=False)

        # Initialize widgets for each mission
        for miss in self.data_dict.keys():
            self.perc_sliders[miss] = pn.widgets.FloatSlider(
                name=f'{miss} Max %', start=90.0, end=100.0, value=99.0, step=0.1
            )
            self.stretch_widgets[miss] = pn.widgets.RadioButtonGroup(
                name=f'{miss} Stretch', options=['linear', 'sqrt', 'log'], value='linear'
            )

        # Create independent panels for each mission
        plots = []
        for miss, data in self.data_dict.items():
            plots.append(self._create_panel(miss, data))

        # Create the grid and merge tools for a single reset/save toolbar
        self.grid = hv.Layout(plots).cols(2).opts(
            shared_axes=True,
            merge_tools=True
        )

    def apply_bounds(self, plot, _):
        """Enforces hard panning bounds at the Bokeh level."""
        plot.state.x_range.bounds = (0, self.width)
        plot.state.y_range.bounds = (0, self.height)

    @staticmethod
    def _process_image(data, stretch_name, upper_pct):
        """Applies stretch and normalization to an image."""
        interval = AsymmetricPercentileInterval(0.0, upper_pct)
        vmin, vmax = interval.get_limits(data)

        if stretch_name == 'sqrt':
            stretch = SqrtStretch()
        elif stretch_name == 'log':
            stretch = LogStretch()
        else:
            stretch = LinearStretch()

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
        return norm(data).filled(0.0)

    def _create_panel(self, mission, data):
        """Creates a single interactive regridded panel."""
        stretch_widget = self.stretch_widgets[mission]
        perc_slider = self.perc_sliders[mission]
        cmap = self.cmaps.get(mission, 'viridis')

        def generate_img(stretch_val, pct_val):
            try:
                processed = self._process_image(data, stretch_val, pct_val)
                return hv.Image(processed, bounds=(0, 0, self.width, self.height), label=mission)
            except Exception as e:
                # Silence specific AttributeError that occurs during Jupyter teardown/re-run
                # where widgets attempt to update plots that have lost their document connection.
                if isinstance(e, AttributeError) and "'NoneType' object has no attribute 'document'" in str(e):
                    return hv.Image(np.array([[0]]), bounds=(0, 0, 1, 1))

                logger.exception(f"Internal error in panel: {mission}")
                self.status_pane.object = (
                    "### System Status\n"
                    f"Error in {mission}: {type(e).__name__}. Check kernel logs."
                )
                self.status_pane.visible = True
                raise e

        bound_fn = pn.bind(
            generate_img,
            stretch_val=stretch_widget,
            pct_val=perc_slider.param.value_throttled
        )

        # Applying regrid here ensures each panel resamples independently
        return regrid(hv.DynamicMap(bound_fn)).opts(
            cmap=cmap,
            width=MULTI_PANEL_WIDTH, height=MULTI_PANEL_HEIGHT,
            title=mission,
            tools=[],
            default_tools=['box_zoom', 'pan', 'wheel_zoom', 'reset', 'save'],
            active_tools=['box_zoom', 'wheel_zoom'],
            xaxis=None, yaxis=None,
            hooks=[self.apply_bounds]
        )

    def view(self):
        """Returns the dashboard layout."""
        controls_list = []
        for miss in self.data_dict.keys():
            controls_list.extend([
                pn.pane.Markdown(f"**{miss}**"),
                self.stretch_widgets[miss],
                self.perc_sliders[miss],
                pn.layout.Divider()
            ])

        controls_list.append(self.status_pane)
        controls = pn.Column(*controls_list, width=CONTROL_PANEL_WIDTH)

        return pn.Row(controls, self.grid)