---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: py-multiband_photometry
  language: python
  name: py-multiband_photometry
---

# Automated Multiband Forced Photometry on Large Datasets

## Learning Goals

By the end of this tutorial, you will be able to:
- get catalogs and images from NASA archives in the cloud where possible
- measure fluxes at any location by running forced photometry using "The Tractor"
- employ parallel processing to make this as fast as possible
- cross match large catalogs
- plot results

## Introduction

This code performs photometry in an automated fashion at all locations in an input catalog on 4 bands of IRAC data from IRSA and 2 bands of Galex data from MAST.  The resulting catalog is then cross-matched with a Chandra catalog from HEASARC to generate a multiband catalog to facilitate galaxy evolution studies.

If you run this code as is, it will only look at a small region of the COSMOS survey, and as a result, the plots near the end will not be so very informative.  We do this for faster runtimes and to show proof of concept as to how to do this work.  Please change the radius in section 1. to be something larger if you want to work with more data, but expect longer runtimes associated with larger areas.

The code makes full use of multiple processors to optimize run time on large datasets.

### Input

- RA and DEC within COSMOS catalog
- desired catalog radius in arcminutes
- mosaics of that region for IRAC and Galex

### Output

- merged, multiband, science ready pandas dataframe
- IRAC color color plot for identifying interesting populations

### Runtime

As of 2025 September, this notebook takes about 13 minutes to run to completion on Fornax using a server with 8GB RAM/2 CPU' and Environment: 'Default Astrophysics' (image).

## Imports

Non-standard Dependencies:
- `tractor` code which does the forced photometry from Lang et al., 2016
- `astroquery` to interface with archives APIs
- `astropy` to work with coordinates/units and data structures
- `skimage` to work with the images

This cell will install the Python ones if needed:

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_multiband_photometry.txt
```

```{code-cell} ipython3
# standard lib imports
import time
import sys
import os
import shutil

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.heasarc import Heasarc
import pyvo

# Local code imports
sys.path.append('code_src/')

import cutout
from exceptions import TractorError
import photometry
from nway_write_header import nway_write_header
from photometry import Band
from photometry import lookup_img_pair
from galex_functions import galex_get_images

# This code is to parse cloud access information; currently in `code_src`, eventually will be part of pyvo
import fornax

# temporarily let the notebook start without tractor as a dependency
try:
    from find_nconfsources import find_nconfsources
except ImportError:
    print("tractor is missing")
    pass


%matplotlib inline
```

## 1. Retrieve Initial Catalog from IRSA
In this section we query the COSMOS2015 catalog (Laigle et al. 2016) from the IRSA archive using the Table Access Protocol (TAP). We specify the sky position and search radius, select a curated subset of columns needed for forced photometry, validation, and later multiwavelength analysis, and download only those rows that fall within the requested region of the COSMOS field. The result is a compact catalog containing positions, photometric redshifts, optical and infrared fluxes, UV measurements, and existing X-ray associations, which forms the starting point for the remainder of the workflow.

We access the COSMOS2015 catalog using the Table Access Protocol (TAP), an IVOA standard that allows SQL-like queries across astronomical databases. For more about TAP, see the IVOA [documentation](https://www.ivoa.net/documents/TAP/) or [suggestions](https://irsa.ipac.caltech.edu/docs/program_interface/astropy_TAP.html) on its usage from IRSA

```{code-cell} ipython3
# Pull a COSMOS catalog from IRSA using pyVO

# Central RA/Dec of the COSMOS field (from SIMBAD).
# Used as the center of the TAP cone search.
coords = SkyCoord('150.01d 2.2d', frame='icrs')  

# Search radius (in arcminutes).
# Full COSMOS radius is ~48', but querying that area is extremely slow
# (∼24 hours on 128 cores). We start with a small radius for speed.
radius = 0.5 * u.arcmin


# The COSMOS2015 catalog has several thousand columns. Requesting all of
# them is unnecessary and makes the TAP query more memory intensive
# and slower than required.
# We select only the columns needed for:
#   • forced-photometry inputs (Ks and SPLASH fluxes)
#   • validation of Tractor photometry (SPLASH fluxes → Section 3.6)
#   • UV/X-ray diagnostics (GALEX + Chandra fluxes)
#   • source classification (PHOTOZ, type)
#   • optical context (r-band mags)

# These columns allow us to *extend* COSMOS2015 with new IRAC+GALEX
# forced photometry and improved X-ray associations (via nway).

cols = [
    # Astrometry + unique ID
    'ra', 'dec', 'id',

    # Ks-band photometry (used by Tractor for contamination modeling)
    'Ks_FLUX_APER2', 'Ks_FLUXERR_APER2',

    # Photometric redshift
    'PHOTOZ',

    # SPLASH (IRAC) magnitudes and fluxes — used to validate Tractor outputs
    'SPLASH_1_MAG', 'SPLASH_1_MAGERR',
    'SPLASH_1_FLUX', 'SPLASH_1_FLUX_ERR',
    'SPLASH_2_FLUX', 'SPLASH_2_FLUX_ERR',
    'SPLASH_3_FLUX', 'SPLASH_3_FLUX_ERR',
    'SPLASH_4_FLUX', 'SPLASH_4_FLUX_ERR',

    # GALEX UV fluxes
    'FLUX_GALEX_NUV', 'FLUX_GALEX_FUV',

    # Chandra X-ray fluxes + identifier
    'FLUX_CHANDRA_05_2', 'FLUX_CHANDRA_2_10', 'FLUX_CHANDRA_05_10', 'ID_CHANDRA09 ',

    # Morphology / source type (0=galaxy, 1=star, etc.)
    'type',

    # Optical r-band photometry
    'r_MAG_AUTO', 'r_MAGERR_AUTO',

    # 24 μm photometry (used for additional diagnostics)
    'FLUX_24', 'FLUXERR_24',

    # GALEX UV magnitudes (for color–color diagrams in Section 5)
    'MAG_GALEX_NUV', 'MAGERR_GALEX_NUV',
    'MAG_GALEX_FUV', 'MAGERR_GALEX_FUV'
]

# Create TAP service object and submit the cone-search query.
# The WHERE clause selects only sources within the requested radius.
tap = pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')
result = tap.run_sync("""
           SELECT {}
           FROM cosmos2015
           WHERE CONTAINS(POINT('ICRS',ra, dec), CIRCLE('ICRS',{}, {}, {}))=1
    """.format(','.join(cols), coords.ra.value, coords.dec.value, radius.to(u.deg).value))

# Convert the TAP result into an astropy Table for further processing.
cosmos_table = result.to_table()


print("Number of objects: ", len(cosmos_table))
```

### 1.1 Filter Catalog

If desired, you can filter the initial catalog to include only sources that meet specific criteria.

```{code-cell} ipython3
# Here is an example of how to filter the catalog to
# select those rows with either chandra fluxes or Galex NUV fluxes

# example_table = cosmos_table[(cosmos_table['flux_chandra_05_10']> 0) | (cosmos_table['flux_galex_fuv'] > 0)]
```

## 2. Retrieve Image Datasets from the Cloud

+++

### 2.1 Use the fornax cloud access API to obtain the IRAC data from the IRSA S3 bucket

Details in this section may change over time.

```{code-cell} ipython3
# Retrieve the COSMOS service entry from the VO registry using the standard PyVO workflow.
# This avoids hard-coding the service URL and ensures the query follows VO best practices.
image_services = pyvo.regsearch(servicetype='sia')
irsa_cosmos = [s for s in image_services if 'irsa' in s.ivoid and 'cosmos' in s.ivoid][0]

# The search returns 11191 entries, but unfortunately we cannot really filter efficiently in the query
# itself (https://irsa.ipac.caltech.edu/applications/Atlas/AtlasProgramInterface.html#inputparam)
# to get only the Spitzer IRAC results from COSMOS as a mission. We will do the filtering in a next step before download.
cosmos_results = irsa_cosmos.search(coords).to_table()

spitzer = cosmos_results[cosmos_results['dataset'] == 'IRAC']
```

```{code-cell} ipython3
# Temporarily add the cloud_access metadata to the Atlas response.
# This dataset has limited access, thus 'region' should be used instead of 'open'.
# S3 access should be available from the Fornax Science Console.

fname = spitzer['fname']
spitzer['cloud_access'] = [(f'{{"aws": {{ "bucket_name": "irsa-fornax-testdata",'
                            f'              "region": "us-east-1",'
                            f'              "access": "restricted",'
                            f'              "key": "COSMOS/{fn}" }} }}') for fn in fname]
```

```{code-cell} ipython3
# Requires https://github.com/nasa-fornax/fornax-cloud-access-API/pull/4

def fornax_download(data_table, data_subdirectory, access_url_column='access_url',
                    fname_filter=None, verbose=False):
    """
    Downloads data files if they do not already exist in the specified directory.

    Parameters
    ----------
    data_table : iterable
        An iterable containing metadata for files to be downloaded. Each element
        should be a dictionary-like object with at least a 'fname' key.
    data_subdirectory : str
        Name of the subdirectory where the downloaded files will be stored.
    access_url_column : str, optional
        Column name containing the access URLs for downloading the files.
        Default is 'access_url'.
    fname_filter : str, optional
        If provided, only files whose names contain this substring will be downloaded.
    verbose : bool, optional
        If True, print status messages. Default is False.

    Raises
    ------
    ValueError
        If neither 'fname' nor 'name' columns are found in `data_table`.

    Notes
    -----
    - The function checks if a file already exists before downloading it.
    - It creates the target directory if it does not exist.
    - Uses `fornax.get_data_product()` to retrieve the data handler.

    Examples
    --------
    >>> data_table = [{'fname': 'file1.fits', 'access_url': 'https://example.com/file1.fits'},
    ...               {'fname': 'file2.fits', 'access_url': 'https://example.com/file2.fits'}]
    >>> fornax_download(data_table, 'my_data', verbose=True)
    Skipping file1.fits: already exists.
    Downloaded and saved file2.fits to data/my_data
    Download process complete.
    """
    # Define the absolute path of the target directory
    data_directory = os.path.join("data", data_subdirectory)
    os.makedirs(data_directory, exist_ok=True)  # Ensure the directory exists

    # Check which filename column exists
    filename_column = None
    for col in ['fname', 'name']:
        if col in data_table.colnames:
            filename_column = col
            break  # Use the first found column

    if not filename_column:
        raise ValueError("Error: Neither 'fname' nor 'name' columns found in the data table.")

    for row in data_table:
        filename = os.path.basename(row[filename_column])  # Extract filename
        file_path = os.path.join(data_directory, filename)  # Full file path

        # Skip download if file already exists
        if os.path.exists(file_path):
            if verbose:
                print(f"Skipping {filename}: already exists.")
            continue

        # Apply filename filter, if provided
        if fname_filter is not None and fname_filter not in filename:
            continue

        # Download the file
        handler = fornax.get_data_product(row, 'aws', access_url_column=access_url_column, verbose=verbose)
        temp_file = handler.download()

        # Move the downloaded file if a local file path is returned
        if temp_file:
            shutil.move(temp_file, file_path)
            if verbose:
                print(f"Downloaded and saved {filename} to {data_directory}")

    if verbose:
        print("Download process complete.")
```

```{code-cell} ipython3
fornax_download(spitzer, access_url_column='sia_url', fname_filter='go2_sci',
                data_subdirectory='IRAC', verbose=False)
```

### 2.2 Obtain Galex from the MAST archive

```{code-cell} ipython3
# The GALEX COSMOS mosaic is split into four separate tiles, each with its own
# central pointing. To determine which tile each catalog source belongs to,
# we compute the angular distance from every source to the center of each tile.

# Coordinates for the four GALEX tile centers (in degrees)
ra_center=[150.369,150.369,149.869,149.869]
dec_center=[2.45583,1.95583,2.45583,1.95583]

# Build SkyCoord objects for the tile centers and for all catalog sources
galex = SkyCoord(ra = ra_center*u.degree, dec = dec_center*u.degree)
catalog = SkyCoord(ra = cosmos_table['ra'], dec = cosmos_table['dec'])

# For each of the four tiles, compute the angular separation (in degrees)
# between the tile center and every source in the catalog. These columns will
# allow us to identify which tile each source falls into.
cosmos_table['COSMOS_01'] = galex[0].separation(catalog)
cosmos_table['COSMOS_02'] = galex[1].separation(catalog)
cosmos_table['COSMOS_03'] = galex[2].separation(catalog)
cosmos_table['COSMOS_04'] = galex[3].separation(catalog)

# Convert to pandas for easier column operations and downstream merging
df = cosmos_table.to_pandas()

# For each source, identify the GALEX tile with the *minimum* separation.
# The result is a label ('COSMOS_01' ... 'COSMOS_04') indicating which of the
# four GALEX mosaics should be used when extracting UV cutouts.
df['galex_image'] = df[['COSMOS_01','COSMOS_02','COSMOS_03','COSMOS_04']].idxmin(axis = 1)
```

```{code-cell} ipython3
# We expect 76k rows with 15arcmin diameter IRAC images
df.describe()
```

```{code-cell} ipython3
# Download both the images and the skybg files using astroquery.mast
galex_images = galex_get_images(coords, verbose=True)
```

```{code-cell} ipython3
# Check the catalog for missing or invalid values to ensure the data are usable.
df.isna().sum()

# It is acceptable for some flux columns to contain missing values; 
# the remaining columns are complete.
```

```{code-cell} ipython3
# As an exploratory check, examine how many sources of each 
# classification type appear in the catalog.
#Type: 0 = galaxy, 1 = star, 2 = X-ray source, -9 is failure to fit
df.type.value_counts()
```

## 3. Run Forced Photometry

This section performs forced photometry at the positions of all COSMOS2015 sources using The Tractor. We prepare the inputs required for Tractor by first collecting the IRAC and GALEX images and then supplying the corresponding point response functions (PRFs), which describe how a point source is spread across the detector. The PRFs are included with the notebook rather than downloaded at runtime because they are stable, calibrated instrument files that do not change across observations; if they were not provided, users could obtain them from the Spitzer IRAC Instrument Handbook or the GALEX calibration database. Next, we identify nearby contaminating sources for each target, estimate the local sky background, and define the parameters needed to extract a cutout around each object. The following two code cells create Band objects — small data structures that bundle together the PRF, pixel scale, cutout size, flux-conversion factor, and band index — so that each IRAC and GALEX image can be handled consistently during the photometry. Once these inputs are assembled, Tractor is run to measure instrumental fluxes and uncertainties in each band. These measurements form the foundation of the multiband catalog constructed later in the workflow.

+++

### 3.1 Setup

In this step we initialize the DataFrame columns that will hold the Tractor-derived fluxes and uncertainties. We then assemble the configuration for each IRAC and GALEX band, including the PRFs, pixel scales, and cutout sizes, and load the corresponding science and background images needed for the photometry. This setup ensures that all band-specific parameters are ready for the main photometry loop.

```{code-cell} ipython3
# initialize columns in data frame for photometry results
cols = ["ch1flux", "ch1flux_unc", "ch2flux", "ch2flux_unc", "ch3flux", "ch3flux_unc",
        "ch4flux", "ch4flux_unc", "ch5flux", "ch5flux_unc", "ch6flux", "ch6flux_unc"]
df[cols] = 0.0

# list to collect all the bands
all_bands = []
```

```{code-cell} ipython3
# IRAC channels
irac_band_indexes = [
    0,  # ch1
    1,  # ch2
    2,  # ch3
    3,  # ch4
]

irac_fluxconversion = (1E12) / (4.254517E10) * (0.6) *(0.6)

irac_mosaic_pix_scale = 0.6

irac_cutout_width = 10 # in arcseconds, taken from Nyland et al. 2017

irac_prfs = [
    fits.open('data/IRAC/PRF_IRAC_ch1.fits')[0].data,
    fits.open('data/IRAC/PRF_IRAC_ch2.fits')[0].data,
    fits.open('data/IRAC/PRF_IRAC_ch3.fits')[0].data,
    fits.open('data/IRAC/PRF_IRAC_ch4.fits')[0].data,
]

# zip parameters for each band into a container and append to the master list
irac_bands = [
    Band(
        idx, prf, irac_cutout_width, irac_fluxconversion, irac_mosaic_pix_scale
    )
    for idx, prf in zip(irac_band_indexes, irac_prfs)
]
all_bands += irac_bands
```

```{code-cell} ipython3
# GALEX bands
galex_band_indexes = [
    4,  # nuv
    5,  # fuv
]

galex_cutout_width = 40

galex_fluxconversions = [
    3.373E1,  # uJy. fudging this to make the numbers bigger for plotting later
    1.076E2,  # uJy. fudging this to make the numbers bigger for plotting later
]

galex_mosaic_pix_scale = 1.5

prf_nuv = fits.open("data/Galex/PSFnuv_faint.fits")[0].data
prf_fuv = fits.open("data/Galex/PSFfuv.fits")[0].data
prf_nuv = prf_nuv[0:119, 0:119]
prf_fuv = prf_fuv[0:119, 0:119]

# These are much larger than the cutouts we are using, so only keep the central 
# region which is the size of our cutouts
ngalex_pix = galex_cutout_width / galex_mosaic_pix_scale
prf_cen = int(60)
prf_nuv = prf_nuv[(prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2)),
                  (prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2))]
prf_fuv = prf_fuv[(prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2)),
                  (prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2))]
galex_prfs = [prf_nuv, prf_fuv]

# Zip parameters for each band into a container and append to the master list
galex_bands = [
    Band(
        idx, prf, galex_cutout_width, flux_conv, galex_mosaic_pix_scale
    )
    for idx, prf, flux_conv in zip(galex_band_indexes, galex_prfs, galex_fluxconversions)
]
all_bands += galex_bands
```

```{code-cell} ipython3
#Collect input images
# collect the files in pairs: (science image, sky-background image)
# if the same file should be used for both, just send it once
sci_bkg_pairs = [
    # IRAC. use the science image to calculate the background
    ('data/IRAC/irac_ch1_go2_sci_10.fits', ),
    ('data/IRAC/irac_ch2_go2_sci_10.fits', ),
    ('data/IRAC/irac_ch3_go2_sci_10.fits', ),
    ('data/IRAC/irac_ch4_go2_sci_10.fits', ),
    # GALEX. calculate the background from a dedicated file
    ('data/Galex/COSMOS_01-nd-int.fits.gz', 'data/Galex/COSMOS_01-nd-skybg.fits.gz'),
    ('data/Galex/COSMOS_01-fd-int.fits.gz', 'data/Galex/COSMOS_01-fd-skybg.fits.gz'),
    ('data/Galex/COSMOS_02-nd-int.fits.gz', 'data/Galex/COSMOS_02-nd-skybg.fits.gz'),
    ('data/Galex/COSMOS_02-fd-int.fits.gz', 'data/Galex/COSMOS_02-fd-skybg.fits.gz'),
    ('data/Galex/COSMOS_03-nd-int.fits.gz', 'data/Galex/COSMOS_03-nd-skybg.fits.gz'),
    ('data/Galex/COSMOS_03-fd-int.fits.gz', 'data/Galex/COSMOS_03-fd-skybg.fits.gz'),
    ('data/Galex/COSMOS_04-nd-int.fits.gz', 'data/Galex/COSMOS_04-nd-skybg.fits.gz'),
    ('data/Galex/COSMOS_04-fd-int.fits.gz', 'data/Galex/COSMOS_04-fd-skybg.fits.gz'),
]
```

### 3.2 Main Function to do the Forced Photometry

```{code-cell} ipython3
def calc_instrflux(band, ra, dec, stype, ks_flux_aper2, img_pair, df):
    """
    Calculate single-band instrumental fluxes and uncertainties at the given ra, dec
    using tractor.

    Parameters:
    -----------
    band : `Band`
        Collection of parameters for a single band.
        A `Band` is a named tuple with the following attributes:
            idx : int
                Identifier for the band/channel.
                (integer in [0, 1, 2, 3, 4, 5] for the four IRAC bands and two Galex bands)
            prf : np.ndarray
                Point spread function for the band/channel.
            cutout_width : int
                width of desired cutout in arcseconds
            flux_conv : float
                factor used to convert tractor result to microjanskies
            mosaic_pix_scale : float
                Pixel scale of the image
    ra, dec : float
        celestial coordinates for measuring photometry
    stype : int
        0, 1, 2, -9 for star, galaxy, x-ray source
    ks_flux_aper_2 : float
        flux in aperture 2
    img_pair : tuple
        Pair of images for science and background respectively.
        If the tuple only contains one element it will serve double duty.
        A tuple element can be a `fits.ImageHDU` or the path to a FITS file as a `str`.
    df : pd.DataFrame
        Source catalog.
        Previous arguments (ra, dec, stype, ks_flux_aper_2) come from a single row of this df.
        However, we must also pass the entire dataframe in order to find nearby sources which are possible contaminates.

    Returns:
    --------
    outband : int
        reflects the input band index for identification purposes
    flux : float
        Measured flux in microJansky.
        NaN if the forced photometery failed.
    flux_unc : float
        Flux uncertainty in microJansky, calculated from the tractor results.
        NaN if the forced photometery failed or if tractor didn't report a flux variance.
    """

    # Extract a cutout centered on the target object.
    # This returns:
    #   - subimage:    the science cutout
    #   - bkgsubimage: small background region used for sky estimation
    #   - x1, y1:      pixel coordinates of the target within the cutout
    #   - subimage_wcs: WCS for converting RA/Dec of neighbors into pixel coords
    subimage, bkgsubimage, x1, y1, subimage_wcs = cutout.extract_pair(
        ra, dec, img_pair=img_pair, cutout_width=band.cutout_width, mosaic_pix_scale=band.mosaic_pix_scale
    )

    # Identify all nearby sources that fall inside the cutout.
    # These “confusing sources” are modeled alongside the target so that Tractor
    # can correctly account for blending and crowding. `objsrc` is a list of
    # Tractor-ready source objects (target first, then neighbors).
    objsrc, nconfsrcs = find_nconfsources(
        ra, dec, stype, ks_flux_aper2, x1, y1, band.cutout_width, subimage_wcs, df
    )

    # Estimate the local sky background and noise from the background cutout.
    # Tractor needs this information for likelihood evaluation.
    skymean, skynoise = photometry.calc_background(bkgsubimage=bkgsubimage)

    # Run Tractor to fit the model PRF + sky + neighbors to the science cutout.
    # If Tractor cannot converge (common when sources are faint or blended),
    # return NaNs to signal failure for this band.
    try:
        flux_var = photometry.run_tractor(
            subimage=subimage, prf=band.prf, objsrc=objsrc, skymean=skymean, skynoise=skynoise
        )
    except TractorError:
        return (band.idx, np.nan, np.nan)

    # Convert Tractor’s instrumental flux and variance into physical units
    # (microJanskys) using the band-specific flux conversion factor.
    # The uncertainty is derived from the reported flux variance.
    microJy_flux, microJy_unc = photometry.interpret_tractor_results(
        flux_var=flux_var, flux_conv=band.flux_conv, objsrc=objsrc, nconfsrcs=nconfsrcs
    )

    return (band.idx, microJy_flux, microJy_unc)
```

### 3.3 Calculate Forced Photometry with Straightforward but Slow Method

Here we demonstrate a baseline implementation of forced photometry that processes each source sequentially and computes fluxes for all bands. Although this approach is simple and instructive, it is computationally inefficient for large catalogs, and we therefore do not use it.

```{raw-cell}
%%time
# Do the calculation without multiprocessing for benchmarking

# Make a copy for parallel computation
pl_df = df.copy(deep=True)

t0 = time.time()
# For each object
for row in df.itertuples():
    # For each band
    for band in range(6):
        # Measure the flux with tractor
        outband, flux, unc = calc_instrflux(band, row.ra, row.dec, row.type, row.ks_flux_aper2)

        # Put the results back into the dataframe
        df.loc[row.Index, 'ch{:d}flux'.format(outband+1)] = flux
        df.loc[row.Index, 'ch{:d}flux_unc'.format(outband+1)] = unc

t1 = time.time()

#10,000 sources took 1.5 hours with this code on a medium sized machine
```

### 3.4 Calculate Forced Photometry - Parallelization

To improve performance, we parallelize the forced photometry computation across multiple CPU cores. We construct a parameter list containing the source coordinates and band configurations, distribute the workload to worker processes, and combine the results into the main DataFrame. This parallel implementation drastically reduces runtime and enables practical processing of large areas of the COSMOS field.

```{code-cell} ipython3
#Setup:

# list of parameter sets to pass to the parallel photometry function
paramlist = []

for row in df.itertuples():
    for band in all_bands:
         # Select the appropriate science/background image pair for this band and GALEX tile
        img_pair = lookup_img_pair(sci_bkg_pairs, band.idx, row.galex_image)  
        
        # Append the full set of parameters needed by calc_instrflux()
        paramlist.append(
            [row.Index, band, row.ra, row.dec, row.type, row.ks_flux_aper2, img_pair, df]
        )
print('paramlist: ', len(paramlist)) # total number of photometry jobs to run
```

```{code-cell} ipython3
#prove we can do this for one object
calc_instrflux(*paramlist[0][1:])

#same thing, different syntax
# calc_instrflux(paramlist[0][1], paramlist[0][2], paramlist[0][3], paramlist[0][4], paramlist[0][5], paramlist[0][6])
```

```{code-cell} ipython3
#wrapper to measure the photometry on a single object, single band
def calculate_flux(args):
    """
    Wrapper function for parallel photometry that computes fluxes for a
    single (source, band) combination.

    This function unpacks the argument list expected by ``calc_instrflux()``,
    calls that function to perform the forced photometry, and returns both
    the job index and the photometry result. It also performs lightweight
    logging by writing progress updates to a log file every 100 iterations.

    Parameters
    ----------
    args : list or tuple
        A structured argument list of the form
        ``[index, band, ra, dec, stype, ks_flux_aper2, img_pair, df]``
        where:
            index : int
                Row index of the source in the input catalog.
            band : Band
                Band configuration object used by Tractor.
            ra, dec : float
                Celestial coordinates of the source (degrees).
            stype : int
                Source type code for Tractor (0 = star, 1 = galaxy, etc.).
            ks_flux_aper2 : float
                Ks-band flux used as an initial flux estimate.
            img_pair : tuple
                Science/background image pair for the band.
            df : pandas.DataFrame
                Full catalog used to identify neighboring contaminants.

    Returns
    -------
    index : int
        The original job index corresponding to this source–band combination.
    val : tuple
        Output from ``calc_instrflux()``, containing
        ``(band_index, flux_microJy, flux_unc_microJy)``.

    Notes
    -----
    This function is designed to be executed inside a multiprocessing pool,
    which requires functions to accept a single argument. It serves as a
    thin wrapper around ``calc_instrflux()`` and adds optional logging
    of long-running jobs for monitoring purposes.
    """

    val = calc_instrflux(*args[1:])
    # add simple logging
    if (args[0] % 100) == 0 and val[0] == 0:
        with open('output/output.log', 'a') as fp: fp.write(f'{args[0]}\n')
    return(args[0], val)
```

```{code-cell} ipython3
# if results were previously saved to this location, load them
# else start a pool of workers to calculate results in parallel, and save them here


# File where parallel photometry results are stored (one file per search radius)
fname = f'output/results_{radius.value}.npz'

# If cached results exist, load them to avoid re-running the computation
if os.path.exists(fname):
    results = np.load(fname, allow_pickle=True)['results']

# Otherwise, compute all fluxes in parallel and save the output
else:
    from  multiprocessing import Pool
    t0 = time.time()
    # Reset the lightweight progress log
    with open('output/output.log', 'w') as fp: fp.write('')

    # Launch a multiprocessing pool to compute fluxes for all parameter sets
    with Pool() as pool:
        results = pool.map(calculate_flux, paramlist)

    # Save execution time and final results
    dtime = time.time() - t0
    np.savez(fname, results=np.array(results, dtype=object))
    print(f'Parallel calculation took {dtime} seconds')
```

```{code-cell} ipython3
# Put the results into the main daraframe
for res in results:
    idx,(ich, val, err) = res
    df.loc[idx, f'ch{ich+1}flux'] = val
    df.loc[idx, f'ch{ich+1}flux_unc'] = err
```

```{code-cell} ipython3
# Count the number of non-zero ch1 fluxes
print('Parallel calculation: number of ch1 fluxes filled in =',
      np.sum(df.ch1flux > 0))
```

### 3.5 Cleanup

```{code-cell} ipython3
# Had to call the galex flux columns ch5 and ch6
# Fix that by renaming them now
cols = {'ch5flux':'nuvflux', 'ch5flux_unc':'nuvflux_unc','ch6flux':'fuvflux', 'ch6flux_unc':'fuvflux_unc'}
df.rename(columns=cols, inplace = True)
```

```{code-cell} ipython3
# When doing a large run of a large area, save the dataframe with the forced photometry
# so we don't have to do the forced photometry every time

df.to_pickle(f'output/COSMOS_{radius.value}arcmin.pkl')
```

```{code-cell} ipython3
#If you are not runnig the forced photometry, then read in the catalog from a previous run

#df = pd.read_pickle('output/COSMOS_48.0arcmin.pkl')
```

### 3.6 Plot to Confirm our Photometry Results

In this step we evaluate the quality of the Tractor-derived fluxes by comparing them to the corresponding SPLASH IRAC fluxes from the COSMOS2015 catalog. Scatter plots with reference lines allow us to check for consistency and ensure that the forced photometry behaves as expected across all four IRAC channels. These diagnostic figures serve as an important validation of the method.

```{code-cell} ipython3
# Plot tractor fluxes vs. catalog splash fluxes
# Should see a straightline with a slope of 1

# Setup to plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fluxmax = 200
ymax = 80
xmax = 80

# Ch1
# First shrink the dataframe to only those rows where I have tractor photometry
df_tractor = df[(df.splash_1_flux> 0) & (df.splash_1_flux < fluxmax)] #200
#sns.regplot(data = df_tractor, x = "splash_1_flux", y = "ch1flux", ax = ax1, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_1_flux", y = "ch1flux", ax = ax1)

# Add a diagonal line with y = x
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]

# Now plot both limits against eachother
ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax1.set(xlabel = r'COSMOS 2015 flux ($\mu$Jy)', ylabel = r'tractor flux ($\mu$Jy)', title = 'IRAC 3.6')
ax1.set_ylim([0, ymax])
ax1.set_xlim([0, xmax])

# Ch2
# First shrink the dataframe to only those rows where I have tractor photometry
df_tractor = df[(df.splash_2_flux> 0) & (df.splash_2_flux < fluxmax)]
sns.scatterplot(data = df_tractor, x = "splash_2_flux", y = "ch2flux", ax = ax2)

# Add a diagonal line with y = x
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]

# Now plot both limits against eachother
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax2.set(xlabel = r'COSMOS 2015 flux ($\mu$Jy)', ylabel = r'tractor flux ($\mu$Jy)', title = 'IRAC 4.5')
ax2.set_ylim([0, ymax])
ax2.set_xlim([0, xmax])


# Ch3
# First shrink the dataframe to only those rows where I have tractor photometry
df_tractor = df[(df.splash_3_flux> 0) & (df.splash_3_flux < fluxmax)]

sns.scatterplot(data = df_tractor, x = "splash_3_flux", y = "ch3flux", ax = ax3)

# Add a diagonal line with y = x
lims = [
    np.min([ax3.get_xlim(), ax3.get_ylim()]),  # min of both axes
    np.max([ax3.get_xlim(), ax3.get_ylim()]),  # max of both axes
]

# Now plot both limits against eachother
ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax3.set(xlabel = r'COSMOS 2015 flux ($\mu$Jy)', ylabel = r'tractor flux ($\mu$Jy)', title = 'IRAC 5.8')
ax3.set_ylim([0, ymax])
ax3.set_xlim([0, xmax])


# Ch4
# First shrink the dataframe to only those rows where I have tractor photometry
df_tractor = df[(df.splash_4_flux> 0) & (df.splash_4_flux < fluxmax)]

sns.scatterplot(data = df_tractor, x = "splash_4_flux", y = "ch4flux", ax = ax4)

# Add a diagonal line with y = x
lims = [
    np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
    np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
]

# Now plot both limits against each other
ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax4.set(xlabel = r'COSMOS 2015 flux ($\mu$Jy)', ylabel = r'tractor flux ($\mu$Jy)', title = 'IRAC 8.0')
ax4.set_ylim([0, ymax])
ax4.set_xlim([0, xmax])


plt.tight_layout()

fig.subplots_adjust( hspace=0.5)
fig.set_size_inches(8, 12)

#plt.savefig('output/flux_comparison.png')
```

Tractor performs well for the IRAC bands. The panels above compare the Tractor-derived fluxes with the corresponding COSMOS2015 SPLASH fluxes for all four IRAC channels. Each blue point represents an individual object from the selected subset of the catalog. The black line shows the one-to-one relation (y = x), providing a visual reference for agreement between the two measurements.

+++

## 4. Cross Match our New Photometry Catalog with an X-ray archival Catalog

Cross-matching with nway
To identify X-ray counterparts to our IRAC sources, we use the nway algorithm (Salvato et al. 2017), a Bayesian cross-matching tool designed for catalogs with different positional uncertainties, different source densities, and multiple possible counterparts. Unlike a simple cone-based positional match, nway evaluates the full probability that each IRAC source corresponds to each Chandra detection by combining:
the positional uncertainties from both catalogs,
the local surface density of background sources, and
optional priors such as flux distributions or colors.
This probabilistic approach is essential for Chandra data in particular, where positional uncertainties vary with off-axis angle and where multiple IRAC sources may lie within the Chandra error ellipse. Because of this, a TAP crossmatch alone would return many candidate matches, but nway rejects those that are statistically unlikely and keeps only the high-probability associations.


The files
- data/Chandra/COSMOS_chandra.fits
- data/multiband_phot.fits

are pre-generated catalogs included with this tutorial to keep the workflow focused on the cross-matching step rather than on catalog construction.
COSMOS_chandra.fits
Contains Chandra source positions and fluxes extracted from the COSMOS Legacy Survey. This file is a cleaned, pre-formatted version of the public catalog (Civano et al.), reduced to the columns needed for nway (RA, Dec, fluxes, positional uncertainties). It is provided so the notebook does not need to repeat the data-reduction steps required to reproduce this catalog.
multiband_phot.fits
Contains the IRAC catalog with Tractor-measured fluxes and basic metadata. This file is generated earlier in the tutorial workflow and serves as the input photometric catalog for the crossmatch. It is pre-saved to avoid recomputing the photometry every time the notebook is run.
Providing these files keeps the tutorial lightweight and enables readers to run the cross-match step without requiring dedicated compute time to reproduce the upstream photometry or Chandra catalog reduction.

+++

### 4.1 Retrieve the HEASARC Catalog

```{code-cell} ipython3
# Instantiate Heasarc
heasarc = Heasarc()

# List all available catalogs
catalog_list = heasarc.list_catalogs()

# Print names of catalogs that include "ccosmoscat"
# we already know it is there, but just in case we want to be sure, or if you
# want to search for a different catalog and confirm its presence
heasarc.list_catalogs(keywords='ccosmoscat')

# Query the ccosmoscat catalog around our position
ccosmoscat = heasarc.query_region(
    position=coords,
    catalog='ccosmoscat',
    radius=1.0 * u.deg,
    maxrec=5000,
    columns='*'  # Use '*' for all columns instead of "ALL"
)
```

### 4.2 Run `nway` to do the Cross-Match

```{code-cell} ipython3
# Setup:

# Astropy doesn't recognize capitalized units
# so there might be some warnings here on writing out the file, but we can safely ignore those

# Need to make the chandra catalog into a fits table
# and needs to include area of the survey.
ccosmoscat_rad = 1 #radius of chandra cosmos catalog
nway_write_header('data/Chandra/COSMOS_chandra.fits', 'CHANDRA', float(ccosmoscat_rad**2) )


# Also need to transform the main pandas dataframe into fits table for nway
# Make an index column for tracking later
df['ID'] = range(1, len(df) + 1)

# Need this to be a fits table and needs to include area of the survey.
rad_in_arcmin = radius.value  #units attached to this are confusing nway down the line
nway_write_header('data/multiband_phot.fits', 'OPT', float((2*rad_in_arcmin/60)**2) )
```

```{code-cell} ipython3
%%bash

# Call nway
nway.py 'data/Chandra/COSMOS_chandra.fits' :ERROR_RADIUS 'data/multiband_phot.fits' 0.1 --out=data/Chandra/chandra_multiband.fits --radius 15 --prior-completeness 0.9
```

```{code-cell} ipython3
# Clean up the cross match results and merge them back into main pandas dataframe

# Read in the nway matched catalog
xmatch = Table.read('data/Chandra/chandra_multiband.fits', hdu = 1)
df_xmatch = xmatch.to_pandas()

# The manual suggests that p_i should be greater than 0.1 for a pure catalog.
# The matched catalog has multiple optical associations for some of the XMM detections.
# The simplest thing to do is only keep match_flag = 1
matched = df_xmatch.loc[(df_xmatch['p_i']>=0.1) & df_xmatch['match_flag']==1]

# Merge this info back into the df_optical dataframe.
merged = pd.merge(df, matched, 'outer',left_on='ID', right_on = 'OPT_ID')

# Remove all the rows which start with "OPT" because they are duplications of the original catalog
merged = merged.loc[:, ~merged.columns.str.startswith('OPT')]

# Somehow the matching is giving negative fluxes in the band where there is no detection
# If there is a detection in the other band
# clean that up to make those negative fluxes = 0

merged.loc[merged['flux_chandra_2_10'] < 0, 'flux_chandra_2_10'] = 0
merged.loc[merged['flux_chandra_05_2'] < 0, 'flux_chandra_05_2'] = 0
```

```{code-cell} ipython3
# How many Chandra sources are there?
# How many Galex sources are there?

# Make a new column which is a bool of existing chandra measurements
merged['cosmos_chandra_detect'] = 0
merged.loc[merged.flux_chandra_2_10 > 0,'cosmos_chandra_detect']=1

# Make one for Galex too
merged['galex_detect'] = 0
merged.loc[merged.flux_galex_nuv > 0,'galex_detect']=1

# Make chandra hardness ratio column:
# Hard = 'flux_chandra_2_10', soft = flux_chandra_05_2, HR = (H-S)/(H+S)
merged['chandra_HR'] = (merged['flux_chandra_2_10'] - merged['flux_chandra_05_2']) / (merged['flux_chandra_2_10'] + merged['flux_chandra_05_2'])


print('number of Chandra detections =',np.sum(merged.cosmos_chandra_detect > 0))
print('number of Galex detections =',np.sum(merged.galex_detect > 0))
```

## 5. Plot Final Results

In this section we visualize the final multiwavelength photometry by examining the color–color distributions of the sources. These plots help reveal how different populations separate in color space and allow us to assess whether the Tractor-derived fluxes produce the expected trends across the IRAC and GALEX bands.

```{code-cell} ipython3
# IRAC color color plots akin to Lacy et al. 2004
# Overplot galex sources
# Overplot xray sources

# First select on 24 micron
merged_24 = merged[(merged.flux_24 >= 0)].copy()

# Negative Galex fluxes are causing problems, so set those to zero
merged_24.loc[merged_24.fuvflux < 0, 'fuvflux'] = 0
merged_24.loc[merged_24.nuvflux < 0, 'nuvflux'] = 0

# Make color columns
merged_24['F5.8divF3.6'] = merged_24.ch3flux / merged_24.ch1flux
merged_24['F8.0divF4.5'] = merged_24.ch4flux / merged_24.ch2flux

# Detected in all IRAC bands
merged_allirac = merged_24[(merged_24['F8.0divF4.5'] > 0) & (merged_24['F5.8divF3.6'] > 0)]

# Plot all the points
fig, ax = plt.subplots()
sns.scatterplot(data = merged_allirac, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, alpha = 0.5, label = 'all')

# Plot only those points with Galex detections
galex_detect = merged_allirac[merged_allirac.galex_detect > 0]
sns.scatterplot(data = galex_detect, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, alpha = 0.5, label = 'Galex detect')

# Plot only those points with chandra detections
chandra_detect = merged_allirac[merged_allirac.cosmos_chandra_detect > 0]
sns.scatterplot(data = chandra_detect, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, label = 'Chandra detect')



ax.set(xscale="log", yscale="log")
ax.set_ylim([0.1, 10])
ax.set_xlim([0.1, 10])

ax.set(xlabel = 'log F5.8/F3.6', ylabel = 'log F8.0/F4.5')
ax.legend(loc='lower right')
plt.title('IRAC Color Color Plot')
```

This figure shows an IRAC color color plot akin to the seminal work by Lacy et al. 2004.  Points are color coded for those with Galex UV detections and those with Chandra x-ray detections. Note that the different populations are seperating out in this color color space.



+++

:::{admonition}**Note:**  
If you are running this notebook with the default very small search radius, you may not see many (or any) points in this color–color diagram. Increasing the search radius will populate this plot with a more statistically meaningful sample, but doing so will significantly increase the runtime of the forced-photometry step.

```{code-cell} ipython3
# UV IR color color plot akin to Bouquin et al. 2015
fig, ax = plt.subplots()
merged['FUV-NUV'] = merged.mag_galex_fuv - merged.mag_galex_nuv
merged['NUV-3.6'] = merged.mag_galex_nuv - merged.splash_1_mag

# Plot only those points with Galex detections
galex_detect = merged[merged.galex_detect > 0]
sns.kdeplot(data = galex_detect, x = 'NUV-3.6', y = 'FUV-NUV',
            ax = ax, fill = True, levels = 15)#scatterplot , alpha = 0.5

# Plot only those points with chandra detections
# We color code Chandra sources by hardness ratio a la Moutard et al. 2020
chandra_detect = merged[merged.cosmos_chandra_detect > 0]
sns.scatterplot(data = chandra_detect, x = 'NUV-3.6', y = 'FUV-NUV',
                ax = ax, hue= 'chandra_HR',palette="flare")

# Make the legend for hue into a colorbar outside the plot 
norm = plt.Normalize(merged['chandra_HR'].min(), merged['chandra_HR'].max())
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
legend = ax.get_legend()
if legend is not None:
    legend.remove()

ax.figure.colorbar(sm, ax=ax, label='Chandra Hardness Ratio')

#ax.set(xscale="log", yscale="log")
ax.set_ylim([-0.5, 3.5])
ax.set_xlim([-1, 7])

ax.set(xlabel = 'NUV - [3.6]', ylabel = 'FUV - NUV')
#plt.legend([],[], frameon=False)

#fig.savefig("output/color_color.png")
#mpld3.display(fig)
```

:::{note}**Note:**  
This UV–IR color–color diagram may also appear sparsely populated when using the notebook’s default small-area selection. Only a fraction of sources in this subset have reliable GALEX detections, and even fewer have accompanying Chandra matches. Expanding the search radius will produce a richer distribution, but the larger area will require longer processing times in the earlier photometry steps.

```{code-cell} ipython3
print(chandra_detect['chandra_HR'].describe())
```

We extend the works of Bouquin et al. 2015 and Moutard et al. 2020 by showing a GALEX - Spitzer color color diagram over plotted with Chandra detections.  Blue galaxies in these colors are generated by O and B stars and so must currently be forming stars. We find a tight blue cloud in this color space identifying those star forming galaxies.  Galaxies off of the blue cloud have had their star formation quenched, quite possibly by the existence of an AGN through removal of the gas reservoir required for star formation.  Chandra detected galaxies host AGN, and while those are more limited in number, can be shown here to be a hosted by all kinds of galaxies, including quiescent galaxies which would be in the upper right of this plot.  This likely implies that AGN are indeed involved in quenching star formation.  Additionally, we show the Chandra hardness ratio (HR) color coded according to the vertical color bar on the right side of the plot.  Those AGN with higher hardness ratios have their soft x-ray bands heavily obscured and appear to reside preferentially toward the quiescent galaxies.

+++

## About this notebook

- **Authors:** Jessica Krick, David Shupe, Marziye JafariYazani, Brigitta Sipőcz, Vandana Desai, Steve Groom, Troy Raen, and the Fornax team
- **Contact:** For help with this notebook, please open a topic in the [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov/) "Support" category.

### Acknowledgements

- Kristina Nyland for the workflow of the tractor wrapper.
- MAST, HEASARC, & IRSA Fornax teams
- Some content in this notebook was created with the assistance of ChatGPT by OpenAI.  All content has been reviewed and validated by the authors to ensure accuracy.

### References

This work made use of:

- Astroquery; Ginsburg et al., 2019, 2019AJ....157...98G
- Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013, 2022ApJ...935..167A, 2018AJ....156..123A, 2013A&A...558A..33A
- The Tractor; Lang et al. 2016, 2016AJ....151...36L
- Nyland et al. 2017 , 2017ApJS..230....9N
- Salvato et al. 2018, 2018MNRAS.473.4937S
- Laigle et al. 2016, 2016ApJS..224...24L
