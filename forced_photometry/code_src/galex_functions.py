import os
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
import pyvo
from astroquery.mast import Observations

def galex_get_images(coords, search_radius_arcsec=60, min_exptime=40000, 
                     output_dir="data/Galex", max_products_per_source=None, verbose=True):
    """
    Download GALEX science and sky background images for a given coordinate using astroquery.

    Parameters
    ----------
    coords : astropy.coordinates.SkyCoord
        Sky coordinates of the target region.
    search_radius_arcsec : float, optional
        Search radius in arcseconds. Default is 60.
    min_exptime : float, optional
        Minimum exposure time in seconds. Default is 40000.
    output_dir : str, optional
        Directory to store downloaded files. Default is "data/Galex".
    max_products_per_source : int or None, optional
        Maximum number of products to download. If None, download all. Default is None.
    verbose : bool, optional
        Whether to print status messages. Default is True.

    Returns
    -------
    downloaded_files : list of str
        List of paths to the downloaded files.
    """

    if verbose:
        print(f"Querying GALEX around RA={coords.ra.deg:.5f}, Dec={coords.dec.deg:.5f}...")

    Observations.enable_cloud_dataset()
    obs_table = Observations.query_criteria(
        coordinates=coords,
        radius=search_radius_arcsec * u.arcsec,
        obs_collection="GALEX"
    )

    if len(obs_table) == 0:
        print("No GALEX observations found for the target.")
        return []

    if verbose:
        print(f"Found {len(obs_table)} observations. Getting product lists...")

    product_tables = [Observations.get_product_list(obs) for obs in obs_table]
    all_products = vstack(product_tables)


    # Filter for SCIENCE + skybg files
    is_sci = (all_products['productType'] == 'SCIENCE') & (all_products['dataproduct_type'] == 'image')
    desc_col = all_products['description']
    desc_filled = np.array([str(x) if x is not np.ma.masked else "" for x in desc_col])
    is_bkg = np.char.find(desc_filled, 'skybg') >= 0
    product_subset = all_products[is_sci | is_bkg]

    # Count and list how many match
    if verbose:
        print(f"Found {np.sum(is_bkg)} files with 'skybg' in the filename")

    
    # Filter by exposure time
    obs_ids_good = set(obs_table[obs_table['t_exptime'] > min_exptime]['obs_id'])
    is_good_obs = np.array([obs_id in obs_ids_good for obs_id in product_subset['obs_id']])

    final_products = product_subset[is_good_obs]

    # Filter out unwanted file types like -cat.fits.gz or -xd-mcat.fits.gz
    unwanted_substrings = ["-cat.fits", "-xd-mcat.fits"]
    final_products = final_products[
        [not any(substr in name for substr in unwanted_substrings)
         for name in final_products['productFilename']]
    ]

   
    # Download files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    manifest = Observations.download_products(final_products, 
                                              mrp_only=False, 
                                              download_dir=output_dir)
    print(manifest.colnames)
    downloaded_files = []
    for observation in manifest:

        #get the path to the downloaded file
        local_path = observation['Local Path']
        fname = os.path.basename(local_path)

        #put these files in the output directory
        #not the 'mastDownloads' directory structure
        final_path = os.path.join(output_dir,fname)
        shutil.move(local_path, final_path)
        downloaded_files.append(final_path)

    if verbose:
        print("Download complete.")

    return downloaded_files

def galex_get_skybg(coords: SkyCoord,
                    output_dir = "data/Galex",
                    verbose = False):
    """
    Download GALEX sky background images using the SIA (Simple Image Access) VO service.

    Parameters
    ----------
    coords : SkyCoord
        Sky position to search.
    output_dir : str, optional
        Destination folder for sky background FITS files. Default is "data/Galex".
    verbose : bool, optional
        Whether to print status messages.

    Returns
    -------
    downloaded_files : list of str
        List of paths to the downloaded sky background files.
    """
    galex_sia_url = 'https://mast.stsci.edu/portal_vo/Mashup/VoQuery.asmx/SiaV1?MISSION=GALEX&'

    if verbose:
        print(f"Querying GALEX SIA service for background images at {coords.to_string('hmsdms')}")

    # Query the service (cone search with size=0 returns all overlaps at that exact point)
    query_result = pyvo.dal.sia.search(galex_sia_url, pos=coords, size=0.0)
    result_table: Table = query_result.to_table()

    if verbose:
        print(f"Found {len(result_table)} total products")
        print(query_result.to_table().colnames)

    # Look for filenames matching the COSMOS_*skybg.fits.gz pattern
    skybkg_pattern = re.compile(r"COSMOS_0[1-4]-[fn]d-skybg.*\.fits\.gz")

    # The 'name' field contains the full URL path or file basename
    mask = [bool(skybkg_pattern.search(name)) for name in result_table['name']]
    skybg_products = result_table[mask]

    if verbose:
        print(f"Found {len(skybg_products)} sky background files")

    # Download each file via HTTP
    downloaded_files = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for row in skybg_products:
        file_url = row["accessURL"]
        fname = os.path.basename(file_url)
        dest_path = os.path.join(output_dir, fname)

        if os.path.exists(dest_path):
            if verbose:
                print(f"Skipping existing file: {fname}")
            downloaded_files.append(dest_path)
            continue

        try:
            if verbose:
                print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(file_url, dest_path)
            downloaded_files.append(dest_path)
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")

    return downloaded_files
