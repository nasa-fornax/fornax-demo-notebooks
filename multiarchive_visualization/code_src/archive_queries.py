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
from s3fs import S3FileSystem
from astropy.table.column import Column
from astropy.table.row import Row
from astropy.table import Table
from astropy.io import fits
from astroquery.mast import Observations

Observations.enable_cloud_dataset(provider='AWS')

S3_CONN = S3FileSystem(anon=True)

# Crab - hst_skycell-p1784x19y11_acs_wfc_f550m_all is a fantastic multi-visit-mosaic,
#  but actually a bit large for our purposes.

VETTED_OBS = {"crab": {'chandra': "1994", "hubble": "jc6801010", "swift": "00030371012", "spitzer": "50059401.50059401-10.IRAC"}, }

# Setting up the directory structure
ROOT_DATA_DIR = "data/"

# Separate directories for each mission
CHAN_DATA_DIR = os.path.join(ROOT_DATA_DIR, "HEASARC", "Chandra")
SPITZER_DATA_DIR = os.path.join(ROOT_DATA_DIR, "IRSA", "Spitzer")
HST_DATA_DIR = os.path.join(ROOT_DATA_DIR, "MAST", "Hubble")
SWIFT_DATA_DIR = os.path.join(ROOT_DATA_DIR, "MAST", "Swift")

# Now where any outputs will be stored
OUTPUT_DIR = "output/"


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


def load_spitzer_image(chosen_spitzer_im):

    if isinstance(chosen_spitzer_im, (Table, Row)) and 'cloud_access' not in chosen_spitzer_im.columns and 'access_url' not in chosen_spitzer_im.columns:
        raise KeyError("The 'chosen_spitzer_im' argument must have a 'cloud_access' or 'access_url' column.")
    elif not isinstance(chosen_spitzer_im, (Table, Row)):
        raise TypeError("The 'chosen_spitzer_im' argument must be either an Astropy Table or Row instance.")
    elif isinstance(chosen_spitzer_im, Table):
        if len(chosen_spitzer_im) != 1:
            raise ValueError("The 'chosen_spitzer_im' argument must represent a "
                             "single Spitzer image.")
        else:
            chosen_spitzer_im = chosen_spitzer_im[0]

    spitzer_cloud_info =  ast.literal_eval(chosen_spitzer_im['cloud_access'])
    spitzer_access_url = chosen_spitzer_im['access_url']

    if len(spitzer_cloud_info) == 0 and spitzer_access_url is None:
        raise ValueError("The 'chosen_spitzer_im' value has null entries for "
                         "'cloud_access' and 'access_url' columns.")
    elif len(spitzer_cloud_info) > 0:
        if 'aws' not in spitzer_cloud_info:
            raise KeyError("The 'spitzer_cloud_info' dictionary must have an 'aws' entry.")
        elif 'bucket_name' not in spitzer_cloud_info['aws'] or 'key' not in spitzer_cloud_info['aws']:
            raise KeyError("The 'spitzer_cloud_info['aws']' dictionary must contain "
                           "'bucket_name' and 'key' keys.")

        im_s3_uri = f"s3://{spitzer_cloud_info['aws']['bucket_name']}/{spitzer_cloud_info['aws']['key']}"

        return fits.open(im_s3_uri, use_fsspec=True, fsspec_kwargs={'anon': True})
    else:
        return fits.open(spitzer_access_url)


def load_swift_image(chosen_swift_im):

    # If the input is a string, we'll assume it's the URL
    if isinstance(chosen_swift_im, str):
        # Certain Astroquery-MAST methods don't get on well with numpy strings, which
        #  are typically the dtype returned from accessing an astropy table string
        #  column. As such we make sure to turn them into base Python strings
        rel_swift_url = str(chosen_swift_im)
    elif isinstance(chosen_swift_im, (Table, Row, Column)):

        if isinstance(chosen_swift_im, (Table, Row)):
            chosen_swift_im = chosen_swift_im['dataURL']

        # It should (fingers crossed) definitely be a Column instance by now
        #  We'll quickly check that the name of the Column is correct
        if chosen_swift_im.name != 'dataURL':
            raise ValueError("If an Astropy Column instance is passed for "
                             "'chosen_swift_im', it must be named 'dataURL'.")

        if len(chosen_swift_im) == 1:
            rel_swift_url = str(chosen_swift_im[0])

        elif len(chosen_swift_im) > 1:
            raise ValueError("The 'chosen_swift_im' should represent a single "
                             "image, rather than multiple products.")

    else:
        raise TypeError("The 'chosen_swift_im' argument must be either a string "
                        "representing a URL, or an Astropy Table, Row, or Column.")

    return fits.open(rel_swift_url)


def load_hubble_image(chosen_hubble_im):

    # If the input is a string, we'll assume it's a URI.
    if isinstance(chosen_hubble_im, str):
        # Certain Astroquery-MAST methods don't get on well with numpy strings, which
        #  are typically the dtype returned from accessing an astropy table string
        #  column. As such we make sure to turn them into base Python strings
        rel_mast_uri = str(chosen_hubble_im)

    elif isinstance(chosen_hubble_im, (Table, Row, Column)):

        if isinstance(chosen_hubble_im, (Table, Row)):
            chosen_hubble_im = chosen_hubble_im['dataURI']

        # It should (fingers crossed) definitely be a Column instance by now
        #  We'll quickly check that the name of the Column is correct
        if chosen_hubble_im.name != 'dataURI':
            raise ValueError("If an Astropy Column instance is passed for "
                             "'chosen_hubble_im', it must be named 'dataURI'.")

        if len(chosen_hubble_im) == 1:
            rel_mast_uri = str(chosen_hubble_im[0])

        elif len(chosen_hubble_im) > 1:
            raise ValueError("The 'chosen_hubble_im' should represent a single "
                             "image, rather than multiple products.")

    else:
        raise TypeError("The 'chosen_hubble_im' argument must be either a string "
                        "representing a URI, or an Astropy Table, Row, or Column.")

    hubble_im_s3_uri = Observations.get_cloud_uri(rel_mast_uri)

    # Possible that the product in question won't be available in a MAST S3
    #  bucket, and unfortunately, neither 'get_cloud_uri' nor 'get_cloud_uris'
    #  actually throw an error when that happens.
    # As such, if the return is None, we have to try to get the file another way
    if hubble_im_s3_uri is None:
        # Ideally, we would fail over to fetching the MAST on prem URL, but there
        #  isn't a method that returns that URL in the MAST submodule of Astroquery,
        #  and it would be a little fragile to re-implement the URL construction
        #  here (see the Observations.download_file(...) method for the steps
        #  required to get the URL.
        # As such, we just have to download the image.
        Observations.download_file(rel_mast_uri,
                                   local_path=HST_DATA_DIR,
                                   verbose=False)

        # We force on prem to stop it
        #  trying to download from S3, as if it were available on S3 we'd never have
        #  gotten to this fallback method
        # THIS ARGUMENT IS NOT YET AVAILABLE IN THE RELEASED VERSION OF
        #  ASTROQUERY - SHOULD BE IN v0.4.12
        # force_on_prem=True

        hst_im = fits.open(os.path.join(HST_DATA_DIR, os.path.basename(rel_mast_uri)))

    # In this case we have a valid S3 URI, so we can stream the file as we hoped
    else:
        hst_im = fits.open(hubble_im_s3_uri, use_fsspec=True,
                           fsspec_kwargs={'anon': True})

    return hst_im
