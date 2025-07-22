import os
import shutil
import numpy as np
from astropy.table import vstack
import astropy.units as u
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

    #product_tables = [Observations.get_product_list(obs) for obs in obs_table]
    all_products = Observations.get_product_list(obs_table)

    # Filter for SCIENCE images
    sci_products = Observations.filter_products(
        all_products,
        dataproduct_type="image",
        productType="SCIENCE"
    )

    # Filter for sky background images based on description
    skybg_products = Observations.filter_products(
        all_products, 
        dataproduct_type='image', 
        description='Sky background image (J2000)')

    if verbose:
        print(f"Found {len(skybg_products)} sky background image(s)")

    #get science and sky bg products:
    combined_products = vstack([sci_products, skybg_products])
    
     # Filter by exposure time
    obs_ids_good = set(obs_table[obs_table['t_exptime'] > min_exptime]['obs_id'])
    is_good_obs = np.array([obs_id in obs_ids_good for obs_id in combined_products['obs_id']])

    final_products = combined_products[is_good_obs]

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
        print("Downloaded ", len(downloaded_files), " total files")

    return downloaded_files


