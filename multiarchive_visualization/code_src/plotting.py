"""
Interactive plotting functions for multi-wavelength images.

This module provides classes for creating interactive RGB composites
and multi-panel stretches using Matplotlib and ipywidgets.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import ipywidgets as widgets
from IPython.display import clear_output
from astropy.visualization import (
    AsymmetricPercentileInterval,
    LinearStretch, LogStretch, SqrtStretch,
    ImageNormalize
)

from image_processing import create_rgb_composite

# Default visualization constants
RGB_FIG_SIZE = (6.5, 6.5)
MULTI_FIG_SIZE = (6.5, 6.5)
CONTROL_PANEL_WIDTH = '480px'

# DEFAULT_RED_PERC = 95
# DEFAULT_GREEN_PERC = 95
# DEFAULT_BLUE_PERC = 99.5
# DEFAULT_Q = 8
# DEFAULT_STRETCH = 0.1

# Max dimension for interactive display to prevent memory issues
# MAX_DISPLAY_DIM = 1024


def get_display_data(data, max_pix: int):
    """
    Downsample image data for interactive display if it exceeds the input `max_pix`.
    """
    if data is None:
        return None

    longest_im_pix_size = data.shape[:2].max()

    if longest_im_pix_size > max_pix:
        step = int(np.ceil(longest_im_pix_size / max_pix))
        return data[::step, ::step]
    return data


def apply_colormap(data, cmap_name, upper_pct=99.0, stretch_name='linear'):
    """
    Normalize data and apply a colormap, returning an RGB array.
    """
    interval = AsymmetricPercentileInterval(0, upper_pct)
    vmin, vmax = interval.get_limits(data)

    if stretch_name == 'sqrt':
        stretch = SqrtStretch()
    elif stretch_name == 'log':
        stretch = LogStretch()
    else:
        stretch = LinearStretch()

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
    normalized = norm(data).filled(0.0)

    cmap = colormaps.get_cmap(cmap_name)
    return cmap(normalized)[:, :, :3]


class InteractiveRGBPanel:
    """
    An interactive dashboard for creating RGB composite images.
    """

    def __init__(self, red_data: np.ndarray,
                 green_data: np.ndarray,
                 blue_data: np.ndarray,
                 start_red_int_perc: float = 95,
                 start_green_int_perc: float = 95,
                 start_blue_int_perc: float = 99.5,
                 start_q: float = 8,
                 start_stretch: float = 0.1,
                 max_pix_per_side: int = 1024):

        # Downsample for display stability
        self.r = get_display_data(np.nan_to_num(red_data, nan=0, copy=False), max_pix_per_side)
        self.g = get_display_data(np.nan_to_num(green_data, nan=0, copy=False), max_pix_per_side)
        self.b = get_display_data(np.nan_to_num(blue_data, nan=0, copy=False), max_pix_per_side)

        # Initialize widgets
        self.q_slider = widgets.FloatSlider(
            description='Lupton Q', min=0.1, max=20, value=start_q, step=0.5, continuous_update=False
        )
        self.stretch_slider = widgets.FloatSlider(
            description='Stretch', min=0.01, max=2, value=start_stretch, step=0.05, continuous_update=False
        )
        self.r_perc = widgets.FloatSlider(
            description='Red %', min=50, max=100, value=start_red_int_perc, step=0.5, continuous_update=False
        )
        self.g_perc = widgets.FloatSlider(
            description='Green %', min=50, max=100, value=start_green_int_perc, step=0.5, continuous_update=False
        )
        self.b_perc = widgets.FloatSlider(
            description='Blue %', min=50, max=100, value=start_blue_int_perc, step=0.5, continuous_update=False
        )

        self.status = widgets.HTML(
            "<div style='display: flex; flex-direction: column; align-items: center; margin-top: 30px;'>"
            "<span style='color: #666; font-size: 1.2em;'><i>✓ Ready</i></span>"
            "</div>"
        )

        self.out = widgets.Output(layout=widgets.Layout(width='100%', height='100%'))
        
        # Attach observers
        for w in [self.q_slider, self.stretch_slider, self.r_perc, self.g_perc, self.b_perc]:
            w.observe(self._update_plot, names='value')

        # Initial render
        with self.out:
            self._render_initial()

    def _render_initial(self):
        plt.close('all')  # Clean up
        self.fig, self.ax = plt.subplots(figsize=RGB_FIG_SIZE)
        rgb = create_rgb_composite(
            self.r, self.g, self.b,
            red_percentile=self.r_perc.value,
            green_percentile=self.g_perc.value,
            blue_percentile=self.b_perc.value,
            Q=self.q_slider.value, stretch=self.stretch_slider.value
        )
        self.ax.imshow(rgb, origin='lower')
        self.ax.axis('off')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.show()

    def _update_plot(self, _):
        self.status.value = (
            "<div style='display: flex; flex-direction: column; align-items: center; color: #00aa41; margin-top: 30px;'>"
            "<div class='loader' style='border: 8px solid #f3f3f3; border-top: 8px solid #00aa41; "
            "border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; "
            "margin-bottom: 15px;'></div>"
            "<span style='font-size: 1.3em;'><b>Processing blend...</b></span>"
            "<style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>"
            "</div>"
        )
        
        with self.out:
            clear_output(wait=True)
            self._render_initial()

        self.status.value = (
            "<div style='display: flex; flex-direction: column; align-items: center; margin-top: 30px;'>"
            "<span style='color: #666; font-size: 1.2em;'><i>✓ Ready</i></span>"
            "</div>"
        )

    def view(self):
        controls = widgets.VBox([
            widgets.HTML("<b>RGB Composite Controls</b>"),
            self.q_slider, self.stretch_slider,
            self.r_perc, self.g_perc, self.b_perc,
            self.status
        ], layout=widgets.Layout(width=CONTROL_PANEL_WIDTH))
        return widgets.HBox([controls, self.out])


class InteractiveMultiPanel:
    """
    An interactive dashboard for multi-panel visualization.
    """

    def __init__(self, reprojected_data, cmaps, max_pix: int = 1024):
        self.data_dict = {
            k: get_display_data(np.nan_to_num(v, nan=0.0, copy=False), max_pix)
            for k, v in reprojected_data.items()
        }
        self.cmaps = cmaps
        self.missions = list(self.data_dict.keys())

        # Create widgets
        self.perc_sliders = {}
        self.stretch_widgets = {}

        for miss in self.missions:
            self.perc_sliders[miss] = widgets.FloatSlider(
                description='Interval %', min=50.0, max=100.0, value=99.0, step=0.1, continuous_update=False
            )
            self.stretch_widgets[miss] = widgets.Dropdown(
                description='Stretch', options=['linear', 'sqrt', 'log'], value='linear'
            )
            self.perc_sliders[miss].observe(lambda _, m=miss: self._update_plot(m), names='value')
            self.stretch_widgets[miss].observe(lambda _, m=miss: self._update_plot(m), names='value')

        self.status = widgets.HTML(
            "<div style='display: flex; flex-direction: column; align-items: center; margin-top: 30px;'>"
            "<span style='color: #666; font-size: 1.2em;'><i>✓ Ready</i></span>"
            "</div>"
        )

        self.out = widgets.Output(layout=widgets.Layout(width='100%', height='100%'))

        # Initialize RGB cache for faster updates
        self.rgb_cache = {}
        for miss in self.missions:
            self.rgb_cache[miss] = apply_colormap(
                self.data_dict[miss], self.cmaps.get(miss, 'viridis'),
                upper_pct=self.perc_sliders[miss].value,
                stretch_name=self.stretch_widgets[miss].value
            )

        with self.out:
            self._render_from_cache()

    def _render_from_cache(self):
        """
        Renders the multi-panel plot using the cached RGB arrays.
        """
        n_cols = 2
        n_rows = (len(self.missions) + 1) // 2
        # Use a consistent figure creation pattern
        plt.close('all')  # Clean up to prevent memory creep
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=MULTI_FIG_SIZE)
        self.axes = np.atleast_2d(self.axes).flatten()

        for i, miss in enumerate(self.missions):
            rgb = self.rgb_cache[miss]
            self.axes[i].imshow(rgb, origin='lower')
            self.axes[i].set_title(miss, fontsize=10, pad=5)
            self.axes[i].axis('off')

        # Hide any unused axes
        for j in range(i + 1, len(self.axes)):
            self.axes[j].axis('off')

        plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.02, hspace=0.1)
        plt.show()

    def _update_plot(self, mission):
        self.status.value = (
            "<div style='display: flex; flex-direction: column; align-items: center; color: #00aa41; margin-top: 30px;'>"
            "<div class='loader' style='border: 8px solid #f3f3f3; border-top: 8px solid #00aa41; "
            "border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; "
            "margin-bottom: 15px;'></div>"
            f"<span style='font-size: 1.3em;'><b>Processing {mission}...</b></span>"
            "<style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>"
            "</div>"
        )

        # Update only the changed mission in the cache
        self.rgb_cache[mission] = apply_colormap(
            self.data_dict[mission], self.cmaps.get(mission, 'viridis'),
            upper_pct=self.perc_sliders[mission].value,
            stretch_name=self.stretch_widgets[mission].value
        )

        with self.out:
            clear_output(wait=True)
            self._render_from_cache()

        self.status.value = (
            "<div style='display: flex; flex-direction: column; align-items: center; margin-top: 30px;'>"
            "<span style='color: #666; font-size: 1.2em;'><i>✓ Ready</i></span>"
            "</div>"
        )

    def view(self):
        all_controls = []
        for miss in self.missions:
            all_controls.append(widgets.HTML(f"<b>{miss}</b>"))
            all_controls.append(self.stretch_widgets[miss])
            all_controls.append(self.perc_sliders[miss])
            all_controls.append(widgets.HTML("<hr>"))

        all_controls.append(self.status)
        # Increase max_height to prevent cutting off the status spinner
        controls_layout = widgets.Layout(width=CONTROL_PANEL_WIDTH, overflow='auto', max_height='800px')
        controls = widgets.VBox(all_controls, layout=controls_layout)

        return widgets.HBox([controls, self.out])
