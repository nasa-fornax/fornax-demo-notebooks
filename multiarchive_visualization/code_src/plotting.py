"""
Interactive plotting functions for multi-wavelength images.

This module provides classes for creating interactive RGB composites
and multi-panel stretches using HoloViz (Panel, HoloViews, Datashader).
"""
import os
os.environ['KMP_WARNINGS'] = '0' # Silences the OpenMP warning

import numpy as np
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import regrid
from astropy.visualization import (
    AsymmetricPercentileInterval,
    LinearStretch, LogStretch, SqrtStretch,
    ImageNormalize
)
from image_processing import create_rgb_composite

# Initialize Panel and HoloViews with the loading spinner
pn.extension(loading_spinner='dots', loading_color='#00aa41')
hv.extension('bokeh')


import asyncio
import panel as pn
import holoviews as hv
import numpy as np
from holoviews.operation.datashader import regrid
from holoviews.streams import Pipe
from image_processing import create_rgb_composite

class InteractiveRGBPanel:
    def __init__(self, red_data, green_data, blue_data):
        self.r = np.nan_to_num(red_data, nan=0)
        self.g = np.nan_to_num(green_data, nan=0)
        self.b = np.nan_to_num(blue_data, nan=0)

        self.width = self.r.shape[1]
        self.height = self.r.shape[0]

        self.q_slider = pn.widgets.FloatSlider(name='Lupton Q', start=0.1, end=20, value=8, step=0.5)
        self.stretch_slider = pn.widgets.FloatSlider(name='Stretch', start=0.01, end=2, value=0.1, step=0.05)
        self.r_perc = pn.widgets.FloatSlider(name='Red %', start=50, end=100, value=95, step=0.5)
        self.g_perc = pn.widgets.FloatSlider(name='Green %', start=50, end=100, value=95, step=0.5)
        self.b_perc = pn.widgets.FloatSlider(name='Blue %', start=50, end=100, value=99.5, step=0.5)

        # 1. Compute initial data and setup the Pipe
        initial_rgb = create_rgb_composite(
            self.r, self.g, self.b,
            red_percentile=95, green_percentile=95, blue_percentile=99.5,
            Q=8, stretch=0.1
        )
        self.pipe = Pipe(data=initial_rgb)

        # 2. DynamicMap only reads from the Pipe
        def render_rgb(data):
            return hv.RGB(data, bounds=(0, 0, self.width, self.height))

        self.dmap = hv.DynamicMap(render_rgb, streams=[self.pipe])

        self.interactive_plot = regrid(self.dmap).opts(
            width=700, height=700,
            tools=[],
            active_tools=['wheel_zoom', 'box_zoom'],
            xaxis=None, yaxis=None,
            hooks=[self.apply_bokeh_tweaks]
        )

    def apply_bokeh_tweaks(self, plot, element):
        plot.state.x_range.bounds = (0, self.width)
        plot.state.y_range.bounds = (0, self.height)
        if hasattr(plot.state, 'toolbar') and plot.state.toolbar:
            plot.state.toolbar.logo = None

    # 3. Panel watcher handles the spinner and pushes data to the Pipe
    async def process_and_push(self, *events):
        if hasattr(self, 'image_pane'):
            self.image_pane.loading = True

        await asyncio.sleep(0.05) # Allow UI to render spinner

        new_rgb = create_rgb_composite(
            self.r, self.g, self.b,
            red_percentile=self.r_perc.value,
            green_percentile=self.g_perc.value,
            blue_percentile=self.b_perc.value,
            Q=self.q_slider.value,
            stretch=self.stretch_slider.value
        )

        self.pipe.send(new_rgb)

        if hasattr(self, 'image_pane'):
            self.image_pane.loading = False

    def view(self):
        controls = pn.Column(
            self.q_slider, self.stretch_slider,
            self.r_perc, self.g_perc, self.b_perc,
            width=250
        )

        self.image_pane = pn.pane.HoloViews(self.interactive_plot)

        # Trigger the async process on slider release
        sliders = [self.q_slider, self.stretch_slider, self.r_perc, self.g_perc, self.b_perc]
        for slider in sliders:
            slider.param.watch(self.process_and_push, 'value_throttled')

        return pn.Row(controls, self.image_pane)


class InteractiveMultiPanel:
    def __init__(self, reprojected_data, cmaps):
        # NaNs replaced with 0.0 to restore the original visual style
        self.data_dict = {k: np.nan_to_num(v, nan=0.0) for k, v in reprojected_data.items()}
        self.cmaps = cmaps

        first_img = list(self.data_dict.values())[0]
        self.height, self.width = first_img.shape

        self.perc_sliders = {}
        self.stretch_widgets = {}

        for miss in self.data_dict.keys():
            self.perc_sliders[miss] = pn.widgets.FloatSlider(
                name=f'{miss} Max %', start=90.0, end=100.0, value=99.0, step=0.1
            )
            self.stretch_widgets[miss] = pn.widgets.RadioButtonGroup(
                name=f'{miss} Stretch', options=['linear', 'sqrt', 'log'], value='linear'
            )

        plots = []
        for miss, data in self.data_dict.items():
            plots.append(self._create_panel(miss, data))

        self.grid = hv.Layout(plots).cols(2)

    def apply_bounds(self, plot, element):
        """Enforces hard panning bounds (leaves the logo alone for the multi-panel)"""
        plot.state.x_range.bounds = (0, self.width)
        plot.state.y_range.bounds = (0, self.height)

    def _process_image(self, data, stretch_name, upper_pct):
        interval = AsymmetricPercentileInterval(0.0, upper_pct)
        vmin, vmax = interval.get_limits(data)

        if stretch_name == 'linear':
            stretch = LinearStretch()
        elif stretch_name == 'sqrt':
            stretch = SqrtStretch()
        elif stretch_name == 'log':
            stretch = LogStretch()

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
        return norm(data).filled(0.0)

    def _create_panel(self, mission, data):
        stretch_widget = self.stretch_widgets[mission]
        perc_slider = self.perc_sliders[mission]
        cmap = self.cmaps.get(mission, 'viridis')

        def generate_img(stretch_val, pct_val):
            processed = self._process_image(data, stretch_val, pct_val)
            return hv.Image(processed, bounds=(0, 0, self.width, self.height))

        bound_fn = pn.bind(
            generate_img,
            stretch_val=stretch_widget,
            pct_val=perc_slider.param.value_throttled
        )

        return regrid(hv.DynamicMap(bound_fn)).opts(
            cmap=cmap,
            width=337, height=337,
            title=mission,
            tools=[],
            default_tools=['box_zoom', 'pan', 'wheel_zoom', 'reset', 'save'],
            active_tools=['box_zoom', 'wheel_zoom'],
            xaxis=None, yaxis=None,
            hooks=[self.apply_bounds]
        )

    def view(self):
        controls_list = []
        for miss in self.data_dict.keys():
            controls_list.extend([
                pn.pane.Markdown(f"**{miss}**"),
                self.stretch_widgets[miss],
                self.perc_sliders[miss],
                pn.layout.Divider()
            ])

        controls = pn.Column(*controls_list, width=250)

        # Display layout
        return pn.Row(controls, self.grid)