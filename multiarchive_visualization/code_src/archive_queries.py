"""
Functions for querying NASA archives for multi-wavelength imaging data.

This module provides functions to search for and download imaging data from:
- HEASARC (X-ray: Chandra)
- IRSA (Infrared: Spitzer)
- MAST (Optical: HST, UV: Swift)
"""

from warnings import warn
import ast

import os
import glob
import numpy as np
from s3fs import S3FileSystem
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.table.column import Column
from astropy.io import fits
from astroquery.heasarc import Heasarc
from astroquery.ipac.irsa import Irsa
from astroquery.mast import MastMissions, Observations

S3_CONN = S3FileSystem(anon=True)

VETTED_OBS = {"crab": {'chandra': "1994", "hubble": "JC6801010", "swift": "00030371012", "spitzer": "50059401.50059401-10.IRAC"}, }


def vetted_source_check(check_src_name, check_miss_name):
    check_src_name = check_src_name.lower()
    check_miss_name = check_miss_name.lower()

    if check_src_name not in VETTED_OBS:
        return None
    elif check_miss_name not in VETTED_OBS[check_src_name]:
        return None
    else:
        return VETTED_OBS[check_src_name][check_miss_name]


def load_chandra_image(chandra_datalink, preproc_cent_hi_res: bool = True):
    if preproc_cent_hi_res:
        im_patt = "*cntr_img2*.fits*"
    else:
        im_patt = "*full_img2*.fits*"

    # Validity checks on chandra_datalink
    if isinstance(chandra_datalink, Column):
        if len(chandra_datalink) == 1:
            chandra_datalink = chandra_datalink[0]
        elif len(chandra_datalink) > 1:
            raise ValueError("The 'chandra_datalink' argument must have only "
                             "one element.")
        else:
            raise ValueError("Passed 'chandra_datalink' argument is empty.")
    elif not isinstance(chandra_datalink, str):
        raise TypeError("Pass either a single-entry Astropy column, or a string, to "
                        "the 'chandra_datalink' argument.")

    patt_uri = os.path.join(chandra_datalink, "primary", im_patt)

    try:
        patt_res = S3_CONN.expand_path(patt_uri)

        if len(patt_res) != 1:
            warn(f"Multiple Chandra images found, selecting the first; {patt_res}")

        # Either way selecting the first entry
        im_s3_uri = patt_res[0]

    except FileNotFoundError:
        warn('No Chandra image can be identified.')
        return None

    im_s3_uri = os.path.join(chandra_datalink, "primary", os.path.basename(im_s3_uri))

    return fits.open(im_s3_uri, use_fsspec=True, fsspec_kwargs={'anon': True})


def load_spitzer_image(spitzer_cloud_info):

    # TODO HANDLE NULL INPUTS, RETURN NONE

    if isinstance(spitzer_cloud_info, str):
        spitzer_cloud_info = ast.literal_eval(spitzer_cloud_info)
    elif not isinstance(spitzer_cloud_info, dict):
        raise TypeError("'spitzer_cloud_info' must be either a dictionary or a "
                        "string representation of a dictionary.")

    if 'aws' not in spitzer_cloud_info:
        raise KeyError("The 'spitzer_cloud_info' dictionary must have an 'aws' entry.")
    elif 'bucket_name' not in spitzer_cloud_info['aws'] or 'key' not in spitzer_cloud_info['aws']:
        raise KeyError("The 'spitzer_cloud_info['aws']' dictionary must contain "
                       "'bucket_name' and 'key' keys.")

    im_s3_uri = f"s3://{spitzer_cloud_info['aws']['bucket_name']}/{spitzer_cloud_info['aws']['key']}"

    return fits.open(im_s3_uri, use_fsspec=True, fsspec_kwargs={'anon': True})


def load_swift_image(swift_url):

    if isinstance(swift_url, Column):
        if len(swift_url) == 1:
            swift_url = swift_url[0]
        elif len(swift_url) > 1:
            raise ValueError("The 'swift_url' argument must have only "
                             "one element.")
        else:
            return None

    return fits.open(swift_url)


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
            file_suffix=['DRC'],
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

