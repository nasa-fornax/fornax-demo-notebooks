"""
Functions for image processing: reprojection, normalization, and RGB compositing.

This module provides utilities to:
- Reproject images to a common WCS grid
- Normalize image data for visualization
- Create RGB composite images from multiple wavelengths
"""

import warnings

import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.visualization import (
    ImageNormalize, PercentileInterval, LinearStretch, LogStretch,
    SqrtStretch, AsinhStretch, make_lupton_rgb
)
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_interp

def get_pixel_scale(hdu_or_header):
    """
    Calculate pixel scale from WCS.

    Parameters
    ----------
    hdu_or_header : astropy.io.fits.HDU or astropy.io.fits.Header
        FITS HDU object or header

    Returns
    -------
    astropy.units.Quantity
        Pixel scale in degrees per pixel
    """
    # Extract header if HDU object
    if isinstance(hdu_or_header, (fits.ImageHDU, fits.PrimaryHDU)):
        header = hdu_or_header.header
    else:
        header = hdu_or_header

    # Warning catch still necessary because proj_plane_pixel_scales will
    #  produce these warnings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FITSFixedWarning)
        # Calculate pixel scale from WCS
        wcs = WCS(header, fix=False)
        pixel_scales = wcs.proj_plane_pixel_scales()
    return pixel_scales[0] / Quantity(1, 'pix')


def reproject_to_common_grid(image, targ_hdr):
    # Warning catch still necessary because proj_plane_pixel_scales will
    #  produce these warnings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FITSFixedWarning)
        try:
            # Handle both HDU objects and (data, header) tuples
            if isinstance(image, tuple):
                data, header = image
                reproj_data, reproj_foot = reproject_interp(
                    (data, header), targ_hdr
                )
            else:
                reproj_data, reproj_foot = reproject_interp(image, targ_hdr)

        except (ValueError, TypeError):
            reproj_data = None
            reproj_foot = None

    return reproj_data, reproj_foot


def normalize_image(data, percentile=95, stretch='linear', stretch_params=None):
    """
    Normalize image data for visualization.

    Parameters
    ----------
    data : numpy.ndarray
        Image data
    percentile : float, optional
        Percentile for interval normalization (default: 95)
    stretch : str, optional
        Stretch type: 'linear', 'log', 'sqrt', 'asinh' (default: 'linear')
    stretch_params : dict, optional
        Parameters for the stretch function (e.g., {'intercept': 0.2} for LinearStretch)

    Returns
    -------
    numpy.ndarray
        Normalized image data (values in range [0, 1])
    """
    # Replace NaN with zeros
    data_clean = data.copy()
    data_clean[np.isnan(data_clean)] = 0

    # Set up stretch
    if stretch_params is None:
        stretch_params = {}

    if stretch == 'linear':
        stretch_obj = LinearStretch(**stretch_params)
    elif stretch == 'log':
        stretch_obj = LogStretch(**stretch_params)
    elif stretch == 'sqrt':
        stretch_obj = SqrtStretch(**stretch_params)
    elif stretch == 'asinh':
        stretch_obj = AsinhStretch(**stretch_params)
    else:
        raise ValueError(f"Unknown stretch type: {stretch}")

    # Normalize
    norm = ImageNormalize(
        data_clean,
        interval=PercentileInterval(percentile),
        stretch=stretch_obj
    )

    # Apply normalization and fill masked values
    return np.ma.filled(norm(data_clean), 0.0)


def create_rgb_composite(red_data, green_data, blue_data,
                        red_percentile=95, green_percentile=95, blue_percentile=99.5,
                        Q=8, stretch=0.1):
    """
    Create an RGB composite image using the Lupton algorithm.

    Parameters
    ----------
    red_data : numpy.ndarray
        Data for red channel
    green_data : numpy.ndarray
        Data for green channel
    blue_data : numpy.ndarray
        Data for blue channel
    red_percentile : float, optional
        Percentile for red channel normalization (default: 95)
    green_percentile : float, optional
        Percentile for green channel normalization (default: 95)
    blue_percentile : float, optional
        Percentile for blue channel normalization (default: 99.5)
    Q : float, optional
        Lupton Q parameter (softening parameter, default: 8)
    stretch : float, optional
        Lupton stretch parameter (default: 0.1)

    Returns
    -------
    numpy.ndarray
        RGB image array with shape (height, width, 3) and values in [0, 1]
    """
    # Normalize each channel
    r_norm = normalize_image(red_data, percentile=red_percentile)
    g_norm = normalize_image(green_data, percentile=green_percentile)
    b_norm = normalize_image(blue_data, percentile=blue_percentile)

    # Create RGB composite
    rgb_array = make_lupton_rgb(r_norm, g_norm, b_norm, Q=Q, stretch=stretch)

    return rgb_array


def load_fits_from_s3(s3_path):
    """
    Load a FITS file directly from S3 using fsspec.

    Parameters
    ----------
    s3_path : str
        S3 URI (s3://bucket/key)

    Returns
    -------
    astropy.io.fits.HDUList
        FITS file object
    """
    return fits.open(s3_path, use_fsspec=True, fsspec_kwargs={'anon': True})