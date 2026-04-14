"""
Functions for querying NASA archives for multi-wavelength imaging data.

This module provides functions to search for and download imaging data from:
- HEASARC (X-ray: Chandra)
- IRSA (Infrared: Spitzer)
- MAST (Optical: HST, UV: Swift)
"""

import os
import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from astroquery.ipac.irsa import Irsa
from astroquery.mast import MastMissions, Observations


def query_chandra(coord, detector="ACIS-S", grating="NONE"):
    """
    Query HEASARC for Chandra observations at a given position.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Target coordinates
    detector : str, optional
        Chandra detector to search for (default: "ACIS-S")
    grating : str, optional
        Grating configuration (default: "NONE")

    Returns
    -------
    astropy.table.Table or None
        Table of matching observations, sorted by exposure time (longest first).
        Returns None if no data found.
    """
    try:
        results = Heasarc.query_region(
            coord,
            'chanmaster',
            column_filters={"detector": detector, "grating": grating},
            columns='*'
        )

        if results is None or len(results) == 0:
            return None

        results.sort('exposure', reverse=True)
        return results

    except (ValueError, KeyError, OSError) as e:
        print(f"Chandra query failed: {e}")
        return None


def download_chandra(obs_table, data_dir, obsid=None):
    """
    Download Chandra data products.

    Parameters
    ----------
    obs_table : astropy.table.Table
        Results from query_chandra
    data_dir : str
        Directory to save downloaded files
    obsid : int, optional
        Specific observation ID to download. If None, downloads longest exposure.

    Returns
    -------
    str or None
        Path to downloaded image file, or None if download failed
    """
    if obs_table is None or len(obs_table) == 0:
        return None

    # Select observation
    if obsid is not None:
        sel_obs = obs_table[obs_table['obsid'] == obsid]
    else:
        sel_obs = obs_table[0]  # Longest exposure (already sorted)

    try:
        download_list = Heasarc.locate_data(sel_obs)
        Heasarc.download_data(download_list, host='aws', location=data_dir)

        # Return path to image file (Chandra archive structure)
        obsid_str = str(sel_obs['obsid'][0])
        img_path = os.path.join(data_dir, obsid_str, "primary",
                               f"acisf{obsid_str.zfill(5)}N005_full_img2.fits.gz")

        if os.path.exists(img_path):
            return img_path
        else:
            print(f"Warning: Expected image file not found at {img_path}")
            return None

    except (ValueError, KeyError, OSError) as e:
        print(f"Chandra download failed: {e}")
        return None


def query_spitzer(coord, radius_arcmin=3.0):
    """
    Query IRSA for Spitzer IRAC imaging at a given position.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Target coordinates
    radius_arcmin : float, optional
        Search radius in arcminutes (default: 3.0)

    Returns
    -------
    astropy.table.Row or None
        Single best matching observation (mean mosaic, closest to target).
        Returns None if no data found.
    """
    try:
        results = Irsa.query_sia(
            pos=(coord, Quantity(radius_arcmin, 'arcmin')),
            facility="Spitzer Space Telescope",
            data_type="image",
            spatial_resolution=1.66,  # IRAC pixel scale
            res_format='image/fits',
            calib_level=3
        )

        if results is None or len(results) == 0:
            return None

        # Filter to science products only
        results = results[results['dataproduct_subtype'] == 'science']

        if len(results) == 0:
            return None

        # Filter to mean mosaics (exclude short HDR exposures and median mosaics)
        not_short_median = (
            (~np.char.find(results['access_url'].data.astype(str), 'short') > -1) &
            (~np.char.find(results['access_url'].data.astype(str), 'median') > -1)
        )

        results = results[not_short_median]

        if len(results) == 0:
            return None

        # Select closest to target
        results.sort('dist_to_point')
        return results[0]

    except (ValueError, KeyError, OSError) as e:
        print(f"Spitzer query failed: {e}")
        return None


def get_spitzer_s3_path(obs_row):
    """
    Extract S3 path from Spitzer observation metadata.

    Parameters
    ----------
    obs_row : astropy.table.Row
        Single row from query_spitzer results

    Returns
    -------
    str or None
        S3 URI (s3://bucket/key) for direct cloud access
    """
    if obs_row is None:
        return None

    try:
        aws_info = eval(obs_row['cloud_access'])['aws']
        s3_path = f"s3://{aws_info['bucket_name']}/{aws_info['key']}"
        return s3_path
    except (KeyError, TypeError, SyntaxError) as e:
        print(f"Failed to extract S3 path: {e}")
        return None


def query_hst(coord, instrument="ACS", aperture="WFC", filter_spec="F550M;CLEAR2L",
              min_exposure=1000):
    """
    Query MAST for HST observations at a given position.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Target coordinates
    instrument : str, optional
        HST instrument (default: "ACS")
    aperture : str, optional
        Instrument aperture (default: "WFC")
    filter_spec : str, optional
        Filter specification with both wheel elements separated by semicolon
        (default: "F550M;CLEAR2L")
    min_exposure : float, optional
        Minimum exposure time in seconds (default: 1000)

    Returns
    -------
    astropy.table.Table or None
        Table of matching observations.
        Returns None if no data found.
    """
    try:
        mast_hst = MastMissions(mission="hst")

        results = mast_hst.query_criteria(
            coordinates=coord,
            sci_instrume=instrument,
            sci_aper_1234=aperture,
            sci_actual_duration=f">{min_exposure}",
            sci_spec_1234=filter_spec,
            sci_status='PUBLIC'
        )

        if results is None or len(results) == 0:
            return None

        return results

    except (ValueError, KeyError, OSError) as e:
        print(f"HST query failed: {e}")
        return None


def download_hst(obs_table, data_dir, dataset_name=None):
    """
    Download HST data products.

    Parameters
    ----------
    obs_table : astropy.table.Table
        Results from query_hst
    data_dir : str
        Directory to save downloaded files
    dataset_name : str, optional
        Specific dataset name to download. If None, downloads first result.

    Returns
    -------
    str or None
        Path to downloaded drizzled image file, or None if download failed
    """
    if obs_table is None or len(obs_table) == 0:
        return None

    try:
        mast_hst = MastMissions(mission="hst")

        # Select dataset
        if dataset_name is not None:
            target_dataset = dataset_name
        else:
            target_dataset = obs_table[0]['sci_data_set_name']

        # Download drizzled product
        mast_hst.download_products(
            target_dataset,
            download_dir=data_dir,
            extension='fits',
            type='science',
            file_suffix=['drc'],
            flat=True
        )

        # Return path to downloaded file
        img_path = os.path.join(data_dir, f"{target_dataset.lower()}_drc.fits")

        if os.path.exists(img_path):
            return img_path
        else:
            print(f"Warning: Expected image file not found at {img_path}")
            return None

    except (ValueError, KeyError, OSError) as e:
        print(f"HST download failed: {e}")
        return None


def query_swift(coord, filter_name='UVW2'):
    """
    Query MAST for Swift UVOT observations at a given position.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Target coordinates
    filter_name : str, optional
        UVOT filter name (default: 'UVW2')

    Returns
    -------
    astropy.table.Table or None
        Table of matching observations, sorted by exposure time (longest first).
        Returns None if no data found.
    """
    try:
        results = Observations.query_criteria(
            coordinates=coord,
            obs_collection='SWIFT',
            filters=filter_name
        )

        if results is None or len(results) == 0:
            return None

        results.sort('t_exptime', reverse=True)
        return results

    except (ValueError, KeyError, OSError) as e:
        print(f"Swift query failed: {e}")
        return None


def download_swift(obs_table, data_dir, obs_id=None):
    """
    Download Swift UVOT data products.

    Parameters
    ----------
    obs_table : astropy.table.Table
        Results from query_swift
    data_dir : str
        Directory to save downloaded files
    obs_id : str, optional
        Specific observation ID to download. If None, downloads longest exposure.

    Returns
    -------
    str or None
        Path to downloaded sky image file, or None if download failed
    """
    if obs_table is None or len(obs_table) == 0:
        return None

    try:
        # Select observation
        if obs_id is not None:
            sel_obs = obs_table[obs_table['obs_id'] == obs_id]
        else:
            sel_obs = obs_table[0]  # Longest exposure (already sorted)

        # Get products and download
        products = Observations.get_product_list(sel_obs)
        Observations.download_products(
            products,
            mrp_only=True,  # Minimum recommended products
            download_dir=data_dir,
            flat=True
        )

        # Find the sky image file
        obs_id_str = sel_obs['obs_id'][0]

        # Swift filenames follow pattern: sw{obsid}{filter}_sk.img
        pattern = os.path.join(data_dir, f"sw{obs_id_str}*_sk.img")
        matches = glob.glob(pattern)

        if len(matches) > 0:
            return matches[0]
        else:
            print(f"Warning: Expected Swift image file not found")
            return None

    except (ValueError, KeyError, OSError) as e:
        print(f"Swift download failed: {e}")
        return None
