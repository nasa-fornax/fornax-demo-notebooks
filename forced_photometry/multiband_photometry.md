---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Automated Multiband Forced Photometry on Large Datasets
***

## Learning Goals:
By the end of this tutorial, you will be able to:
- get catalogs and images from NASA archives in the cloud where possible
- measure fluxes at any location by running forced photometry using "The Tractor" 
- employ parallel processing to make this as fast as possible
- cross match large catalogs
- plot results

## Introduction:
This code performs photometry in an automated fashion at all locations in an input catalog on 4 bands of IRAC data from IRSA and 2 bands of Galex data from MAST.  The resulting catalog is then cross-matched with a Chandra catalog from HEASARC to generate a multiband catalog to facilitate galaxy evolution studies.

The code will run on 2 different science platforms and makes full use of multiple processors to optimize run time on large datasets.

## Input:
- RA and DEC within COSMOS catalog
- desired catalog radius in arcminutes
- mosaics of that region for IRAC and Galex

## Output:
- merged, multiband, science ready pandas dataframe
- IRAC color color plot for identifying interesting populations

## Non-standard Imports
- `tractor` code which does the forced photometry from Lang et al., 2016
- `astroquery` to interface with archives APIs
- `astropy` to work with coordinates/units and data structures
- `skimage` to work with the images


## Authors:
Jessica Krick, David Shupe, Marziye JafariYazani, Brigitta Sipőcz, Vandana Desai, Steve Groom, Troy Raen


## Acknowledgements:
Kristina Nyland for the workflow of the tractor wrapper.\
MAST, HEASARC, & IRSA Fornax teams

```{code-cell} ipython3
#ensure all dependencies are installed
!pip install -r requirements.txt
```

```{code-cell} ipython3
# standard lib imports

import math
import time
import warnings
import concurrent.futures
import sys
import os
import re
from typing import NamedTuple

# Third party imports

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import pandas as pd
import seaborn as sns
import statsmodels
import mpld3
from tqdm.auto import tqdm

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.ipac.irsa import Irsa
from astroquery.heasarc import Heasarc
from astroquery.mast import Observations
import pyvo

# Local code imports
sys.path.append('code_src/')

from display_images import display_images
import cutout
from exceptions import TractorError
import photometry
from plot_SED import plot_SED
from nway_write_header import nway_write_header
from photometry import Band
from photometry import lookup_img_pair
#from prepare_prf import prepare_prf

# temporarily let the notebook start without tractor as dependency
try:
    from find_nconfsources import find_nconfsources
except ImportError:
    print("tractor is missing")
    pass


%matplotlib inline
```

## 1. Retrieve Initial Catalog from IRSA
- Automatically set up a catalog with ra, dec, photometric redshifts, fiducial band fluxes, & probability that it is a star  
- Catalog we are using is COSMOS2015 (Laigle et al. 2016)  
- Data exploration

```{code-cell} ipython3
#pull a COSMOS catalog from IRSA using pyVO

#what is the central RA and DEC of the desired catalog
coords = SkyCoord('150.01d 2.2d', frame='icrs')  #COSMOS center acording to Simbad

#how large is the search radius, in arcmin
# full area of COSMOS is radius = 48'
# running this code on the full COSMOS area takes ~24 hours on a 128core server, therefore
# start with a manageable radius and increase as needed
radius = 0.5 * u.arcmin  


#specify only select columns to limit the size of the catalog
cols = [
    'ra', 'dec', 'id', 'Ks_FLUX_APER2', 'Ks_FLUXERR_APER2', 'PHOTOZ', 'SPLASH_1_MAG',
    'SPLASH_1_MAGERR', 'SPLASH_1_FLUX', 'SPLASH_1_FLUX_ERR', 'SPLASH_2_FLUX',
    'SPLASH_2_FLUX_ERR', 'SPLASH_3_FLUX', 'SPLASH_3_FLUX_ERR', 'SPLASH_4_FLUX',
    'SPLASH_4_FLUX_ERR', 'FLUX_GALEX_NUV', 'FLUX_GALEX_FUV', 'FLUX_CHANDRA_05_2',
    'FLUX_CHANDRA_2_10', 'FLUX_CHANDRA_05_10', 'ID_CHANDRA09 ', 'type', 'r_MAG_AUTO',
    'r_MAGERR_AUTO', 'FLUX_24', 'FLUXERR_24', 'MAG_GALEX_NUV', 'MAGERR_GALEX_NUV',
    'MAG_GALEX_FUV', 'MAGERR_GALEX_FUV'
]

tap = pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')
result = tap.run_sync("""
           SELECT {}
           FROM cosmos2015
           WHERE CONTAINS(POINT('ICRS',ra, dec), CIRCLE('ICRS',{}, {}, {}))=1
    """.format(','.join(cols), coords.ra.value, coords.dec.value, radius.to(u.deg).value))
cosmos_table = result.to_table()


print("Number of objects: ", len(cosmos_table))
```

### 1a. Filter Catalog
- if desired could filter the initial catalog to only include desired sources

```{code-cell} ipython3
#an example of how to filter the catalog to 
#select those rows with either chandra fluxes or Galex NUV fluxes

#example_table = cosmos_table[(cosmos_table['flux_chandra_05_10']> 0) | (cosmos_table['flux_galex_fuv'] > 0)]
```

## 2. Retrieve Image Datasets from the Cloud

+++

#### Use the fornax cloud access API to obtain the IRAC data from the IRSA S3 bucket. 

Details here may change as the prototype code is being added to the appropriate libraries, as well as the data holding to the appropriate NGAP storage as opposed to IRSA resources.

```{code-cell} ipython3
# Temporary solution, remove when the fornax API is added to the image
# This relies on the assumption that https://github.com/nasa-fornax/fornax-cloud-access-API is being cloned to this environment. 
# If it's not, then run a ``git clone https://github.com/nasa-fornax/fornax-cloud-access-API --depth=1`` from a terminal at the highest directory root.
# You may need to update the fork if you forked it in the past

import os
if not os.path.exists('../../fornax-cloud-access-API'):
    ! git clone https://github.com/nasa-fornax/fornax-cloud-access-API --depth=1 ../../fornax-cloud-access-API
```

```{code-cell} ipython3
sys.path.append('../../fornax-cloud-access-API')

import pyvo
import fornax
```

```{code-cell} ipython3
# Getting the COSMOS address from the registry to follow PyVO user case approach. We could hardwire it.
image_services = pyvo.regsearch(servicetype='image')
irsa_cosmos = [s for s in image_services if 'irsa' in s.ivoid and 'cosmos' in s.ivoid][0]

# The search returns 11191 entries, but unfortunately we cannot really filter efficiently in the query
# itself (https://irsa.ipac.caltech.edu/applications/Atlas/AtlasProgramInterface.html#inputparam)
# to get only the Spitzer IRAC results from COSMOS as a mission. We will do the filtering in a next step before download.
cosmos_results = irsa_cosmos.search(coords).to_table()

spitzer = cosmos_results[cosmos_results['dataset'] == 'IRAC']
```

```{code-cell} ipython3
# Temporarily add the cloud_access metadata to the Atlas response. 
# This dataset has limited acces, thus 'region' should be used instead of 'open'.
# S3 access should be available from the daskhub and those who has their IRSA token set up.

fname = spitzer['fname']
spitzer['cloud_access'] = [(f'{{"aws": {{ "bucket_name": "irsa-mast-tike-spitzer-data",'
                            f'              "region": "us-east-1",'
                            f'              "access": "restricted",'
                            f'              "key": "data/COSMOS/{fn}" }} }}') for fn in fname]
```

```{code-cell} ipython3
# Adding function to download multiple files using the fornax API. 
# Requires https://github.com/nasa-fornax/fornax-cloud-access-API/pull/4
def fornax_download(data_table, data_subdirectory, access_url_column='access_url',
                    fname_filter=None, verbose=False):
    working_dir = os.getcwd()
    
    data_directory = f"data/{data_subdirectory}"
    os.chdir(data_directory)
    for row in data_table:
        if fname_filter is not None and fname_filter not in row['fname']:
            continue
        handler = fornax.get_data_product(row, 'aws', access_url_column=access_url_column, verbose=verbose)
        temp_file = handler.download()
        # on-prem download returns temp file path, s3 download downloads file
        if temp_file:
            os.rename(temp_file, os.path.basename(row['fname']))
        
    os.chdir(working_dir)
```

```{code-cell} ipython3
fornax_download(spitzer, access_url_column='sia_url', fname_filter='go2_sci', 
                data_subdirectory='IRAC', verbose=False)
```

#### Use IVOA image search and Fornax download to obtain Galex from the MAST archive

```{code-cell} ipython3
#the Galex mosaic of COSMOS is broken into 4 seperate images
#need to know which Galex image the targets are nearest to.
#make a new column in dataframe which figures this out

#four centers for 1, 2, 3, 4 are
ra_center=[150.369,150.369,149.869,149.869]
dec_center=[2.45583,1.95583,2.45583,1.95583]

galex = SkyCoord(ra = ra_center*u.degree, dec = dec_center*u.degree)
catalog = SkyCoord(ra = cosmos_table['ra'], dec = cosmos_table['dec'])
#idx, d2d, d3d = match_coordinates_sky(galex, catalog)  #only finds the nearest one
#idx, d2d, d3d = galex.match_to_catalog_sky(catalog)  #only finds the nearest one

cosmos_table['COSMOS_01'] = galex[0].separation(catalog)
cosmos_table['COSMOS_02'] = galex[1].separation(catalog)
cosmos_table['COSMOS_03'] = galex[2].separation(catalog)
cosmos_table['COSMOS_04'] = galex[3].separation(catalog)

#convert to pandas
df = cosmos_table.to_pandas()

#which row has the minimum value of distance to the galex images
df['galex_image'] = df[['COSMOS_01','COSMOS_02','COSMOS_03','COSMOS_04']].idxmin(axis = 1)
```

```{code-cell} ipython3
# 76k with 15arcmin diameter IRAC images
df.describe()
```

```{code-cell} ipython3
# Assign the endpoint for the MAST Galex Simple Image Access service.
galex_sia_url = 'https://mast.stsci.edu/portal_vo/Mashup/VoQuery.asmx/SiaV1?MISSION=GALEX&'

# Query the Galex SIA service using the COSMOS coordinates defined above.
query_result = pyvo.dal.sia.search(galex_sia_url, pos=coords, size=0.0)
galex_image_products = query_result.to_table()

# Filter the products so we only download SCIENCE products with exposure time > 40000
filtered_products = galex_image_products[(galex_image_products['timExposure'] > 40000.0)]
filtered_products = filtered_products[(filtered_products['productType'] == "SCIENCE")]

# The column containing the on-prem access to fall back on if cloud is unavailable
access_url_column = query_result.fieldname_with_ucd('VOX:Image_AccessReference')

# Download filtered products from AWS
download_subdir = 'Galex'
fornax_download(filtered_products, access_url_column=access_url_column, data_subdirectory=download_subdir)
```

```{code-cell} ipython3
# Get the GALEX sky background fits files in addition to the mosaics
skybg_products = []
skybkg_pattern = re.compile(r"COSMOS_0[1-4]-[fn]d-skybg")

for row in galex_image_products:
    if skybkg_pattern.search(row['name']):
        skybg_products.append(row)

# Download skybg products from AWS
download_subdir = 'Galex'
fornax_download(skybg_products, access_url_column=access_url_column, data_subdirectory=download_subdir)
```

```{code-cell} ipython3
#make sure there aren't any troublesome rows in the catalog
#are there missing values in any rows?
df.isna().sum()

#don't mind that there are missing values for some of the fluxes
#The rest of the rows are complete
```

```{code-cell} ipython3
#out of curiosity how many of each type of source are in this catalog
#Type: 0 = galaxy, 1 = star, 2 = X-ray source, -9 is failure to fit
df.type.value_counts()
```

## 3.  Run Forced Photometry
- Calculate a flux at a given position in 2 IRAC and 2 Galex bands

+++

### 3a. Setup
- initialize data frame columns to hold the results
- collect the parameters for each band/channel
- collect the input images

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

#these are much larger than the cutouts we are using, so only keep the central region which is the size of our cutouts
ngalex_pix = galex_cutout_width / galex_mosaic_pix_scale
prf_cen = int(60)
prf_nuv = prf_nuv[(prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2)),
                  (prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2))]
prf_fuv = prf_fuv[(prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2)),
                  (prf_cen - int(ngalex_pix / 2) - 1) : (prf_cen + int(ngalex_pix / 2))]
galex_prfs = [prf_nuv, prf_fuv]

# zip parameters for each band into a container and append to the master list
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

### 3b. Main Function to do the Forced Photometry

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
    # cutout a small region around the object of interest
    subimage, bkgsubimage, x1, y1, subimage_wcs = cutout.extract_pair(
        ra, dec, img_pair=img_pair, cutout_width=band.cutout_width, mosaic_pix_scale=band.mosaic_pix_scale
    )
    
    # find nearby sources that are possible contaminants
    objsrc, nconfsrcs = find_nconfsources(
        ra, dec, stype, ks_flux_aper2, x1, y1, band.cutout_width, subimage_wcs, df
    )

    # estimate the background
    skymean, skynoise = photometry.calc_background(bkgsubimage=bkgsubimage)

    # do the forced photometry
    # if tractor fails to converge, just return NaNs
    try:
        flux_var = photometry.run_tractor(
            subimage=subimage, prf=band.prf, objsrc=objsrc, skymean=skymean, skynoise=skynoise
        )
    except TractorError:
        return (band.idx, np.nan, np.nan)

    # convert the results
    microJy_flux, microJy_unc = photometry.interpret_tractor_results(
        flux_var=flux_var, flux_conv=band.flux_conv, objsrc=objsrc, nconfsrcs=nconfsrcs
    )

    return (band.idx, microJy_flux, microJy_unc)
```

### 3c. Calculate Forced Photometry with Straightforward but Slow Method
- no longer in use but keeping to demonstrate this capability\
as well as the increase in speed when going to parallelization

```{raw-cell}
%%time
#do the calculation without multiprocessing for benchmarking

#make a copy for parallel computation
pl_df = df.copy(deep=True)

t0 = time.time()
#for each object
for row in df.itertuples():
    #for each band
    for band in range(6):
        #measure the flux with tractor
        outband, flux, unc = calc_instrflux(band, row.ra, row.dec, row.type, row.ks_flux_aper2)
        #put the results back into the dataframe
        df.loc[row.Index, 'ch{:d}flux'.format(outband+1)] = flux
        df.loc[row.Index, 'ch{:d}flux_unc'.format(outband+1)] = unc
        #print(row.ra, row.dec, row.type, row.ks_flux_aper2, band+1,
        #      outband, flux, unc)
t1 = time.time()


#10,000 sources took 1.5 hours with this code on the IPAC SP
```

### 3d. Calculate Forced Photometry - Parallelization
- Parallelization: we can either interate over the rows of the dataframe and run the four bands in parallel; or we could zip together the row index, band, ra, dec,

```{code-cell} ipython3
paramlist = []
for row in df.itertuples():
    for band in all_bands:
        img_pair = lookup_img_pair(sci_bkg_pairs, band.idx, row.galex_image)  # file paths only
        paramlist.append(
            [row.Index, band, row.ra, row.dec, row.type, row.ks_flux_aper2, img_pair, df]
        )
print('paramlist: ', len(paramlist))
```

```{code-cell} ipython3
#proove we can do this for one object
calc_instrflux(*paramlist[0][1:])

#same thing, different syntax
# calc_instrflux(paramlist[0][1], paramlist[0][2], paramlist[0][3], paramlist[0][4], paramlist[0][5], paramlist[0][6])
```

```{code-cell} ipython3
#wrapper to measure the photometry on a single object, single band
def calculate_flux(args):
    """Calculate flux."""
    val = calc_instrflux(*args[1:])
    # add simple logging
    if (args[0] % 100) == 0 and val[0] == 0:
        with open('output/output.log', 'a') as fp: fp.write(f'{args[0]}\n')
    return(args[0], val)
```

```{code-cell} ipython3
# if results were previously saved to this location, load them
# else start a pool of workers to calculate results in parallel, and save them here
fname = f'output/results_{radius.value}.npz'

if os.path.exists(fname):
    results = np.load(fname, allow_pickle=True)['results']

else:
    from  multiprocessing import Pool
    t0 = time.time()
    with open('output/output.log', 'w') as fp: fp.write('')
    with Pool() as pool:
        results = pool.map(calculate_flux, paramlist)
    dtime = time.time() - t0
    np.savez(fname, results=results)
    print(f'Parallel calculation took {dtime} seconds')
```

```{code-cell} ipython3
## put the results into the main daraframe
for res in results:
    idx,(ich, val, err) = res
    df.loc[idx, f'ch{ich+1}flux'] = val
    df.loc[idx, f'ch{ich+1}flux_unc'] = err
```

```{code-cell} ipython3
#Count the number of non-zero ch1 fluxes
print('Parallel calculation: number of ch1 fluxes filled in =',
      np.sum(df.ch1flux > 0))
```

### 3e. Cleanup

```{code-cell} ipython3
#had to call the galex flux columns ch5 and ch6
#fix that by renaming them now
cols = {'ch5flux':'nuvflux', 'ch5flux_unc':'nuvflux_unc','ch6flux':'fuvflux', 'ch6flux_unc':'fuvflux_unc'}
df.rename(columns=cols, inplace = True)
```

```{code-cell} ipython3
#When doing a large run of a large area, save the dataframe with the forced photometry 
# so we don't have to do the forced photometry every time

df.to_pickle(f'output/COSMOS_{radius.value}arcmin.pkl')
```

```{code-cell} ipython3
#If you are not runnig the forced photometry, then read in the catalog from a previous run

#df = pd.read_pickle('output/COSMOS_48.0arcmin.pkl')
```

### 3f. Plot to Confirm our Photometry Results 
- compare to published COSMOS 2015 catalog

```{code-cell} ipython3
%%time
#plot tractor fluxes vs. catalog splash fluxes
#should see a straightline with a slope of 1

#setup to plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fluxmax = 200
ymax = 100
xmax = 100
#ch1 
#first shrink the dataframe to only those rows where I have tractor photometry 
df_tractor = df[(df.splash_1_flux> 0) & (df.splash_1_flux < fluxmax)] #200
#sns.regplot(data = df_tractor, x = "splash_1_flux", y = "ch1flux", ax = ax1, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_1_flux", y = "ch1flux", ax = ax1)

#add a diagonal line with y = x
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax1.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 3.6')
ax1.set_ylim([0, ymax])
ax1.set_xlim([0, xmax])


#ch2 
#first shrink the dataframe to only those rows where I have tractor photometry 
df_tractor = df[(df.splash_2_flux> 0) & (df.splash_2_flux < fluxmax)]
#sns.regplot(data = df_tractor, x = "splash_2_flux", y = "ch2flux", ax = ax2, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_2_flux", y = "ch2flux", ax = ax2)

#add a diagonal line with y = x
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax2.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 4.5')
ax2.set_ylim([0, ymax])
ax2.set_xlim([0, xmax])


#ch3 
#first shrink the dataframe to only those rows where I have tractor photometry
df_tractor = df[(df.splash_3_flux> 0) & (df.splash_3_flux < fluxmax)]

#sns.regplot(data = df_tractor, x = "splash_3_flux", y = "ch3flux", ax = ax3, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_3_flux", y = "ch3flux", ax = ax3)

#add a diagonal line with y = x
lims = [
    np.min([ax3.get_xlim(), ax3.get_ylim()]),  # min of both axes
    np.max([ax3.get_xlim(), ax3.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax3.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 5.8')
ax3.set_ylim([0, ymax])
ax3.set_xlim([0, xmax])


#ch4 
#first shrink the dataframe to only those rows where I have tractor photometry 
df_tractor = df[(df.splash_4_flux> 0) & (df.splash_4_flux < fluxmax)]

#sns.regplot(data = df_tractor, x = "splash_4_flux", y = "ch4flux", ax = ax4, robust = True)
sns.scatterplot(data = df_tractor, x = "splash_4_flux", y = "ch4flux", ax = ax4)

#add a diagonal line with y = x
lims = [
    np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
    np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax4.set(xlabel = 'COSMOS 2015 flux ($\mu$Jy)', ylabel = 'tractor flux ($\mu$Jy)', title = 'IRAC 8.0')
ax4.set_ylim([0, ymax])
ax4.set_xlim([0, xmax])


plt.tight_layout()

fig.subplots_adjust( hspace=0.5)
fig.set_size_inches(8, 12)

#plt.savefig('output/flux_comparison.png')
```

Tractor is working for IRAC; Comparison of tractor derived fluxes with COSMOS 2015 fluxes for all four Spitzer IRAC channels.  Blue points represent each object from the subset of the COSMOS 2015 catalog.  The blue line is a linear regression robust fit to the data with uncertainties shown as the light blue wedge.  The black line is a y = x line plotted to guide the eye.

+++

## 4. Cross Match our New Photometry Catalog with an X-ray archival Catalog
We are using `nway` as the tool to do the cross match (Salvato et al. 2017).
`nway` expects input as two fits table files and outputs a third table file with all the possible matches and their probabilities of being the correct match.  We then sort that catalog and take only the best matches to be the true matches.

+++

### 4a. Retrieve the HEASARC Catalog

```{code-cell} ipython3
#first get an X-ray catalog from Heasarc
heasarc = Heasarc()
table = heasarc.query_mission_list()
mask = (table['Mission'] =="CHANDRA")
chandratable = table[mask]  

#find out which tables exist on Heasarc
#chandratable.pprint(max_lines = 200, max_width = 130)

#want ccosmoscat
mission = 'ccosmoscat'
#coords already defined above where I pull the original COSMOS catalog
ccosmoscat_rad = 1 #radius of chandra cosmos catalog
ccosmoscat = heasarc.query_region(coords, mission=mission, radius='1 degree', resultmax = 5000, fields = "ALL")
```

### 4b. Run `nway` to do the Cross-Match

```{code-cell} ipython3
#setup:

#astropy doesn't recognize capitalized units
#so there might be some warnings here on writing out the file, but we can safely ignore those

#need to make the chandra catalog into a fits table 
#and needs to include area of the survey.
nway_write_header('data/Chandra/COSMOS_chandra.fits', 'CHANDRA', float(ccosmoscat_rad**2) )


#also need to transform the main pandas dataframe into fits table for nway
#make an index column for tracking later
df['ID'] = range(1, len(df) + 1)

#need this to be a fits table and needs to include area of the survey.
rad_in_arcmin = radius.value  #units attached to this are confusing nway down the line
nway_write_header('data/multiband_phot.fits', 'OPT', float((2*rad_in_arcmin/60)**2) )
```

```{code-cell} ipython3
# call nway
!/home/jovyan/.local/bin/nway.py 'data/Chandra/COSMOS_chandra.fits' :ERROR_RADIUS 'data/multiband_phot.fits' 0.1 --out=data/Chandra/chandra_multiband.fits --radius 15 --prior-completeness 0.9
    
#!/opt/conda/bin/nway.py 'data/Chandra/COSMOS_chandra.fits' :ERROR_RADIUS 'data/multiband_phot.fits' 0.1 --out=data/Chandra/chandra_multiband.fits --radius 15 --prior-completeness 0.9
```

```{code-cell} ipython3
#Clean up the cross match results and merge them back into main pandas dataframe

#read in the nway matched catalog
xmatch = Table.read('data/Chandra/chandra_multiband.fits', hdu = 1)
df_xmatch = xmatch.to_pandas()

#manual suggests that p_i should be greater than 0.1 for a pure catalog.
#The matched catalog has multiple optical associations for some of the XMM detections.
#simplest thing to do is only keep match_flag = 1
matched = df_xmatch.loc[(df_xmatch['p_i']>=0.1) & df_xmatch['match_flag']==1]

#merge this info back into the df_optical dataframe.
merged = pd.merge(df, matched, 'outer',left_on='ID', right_on = 'OPT_ID')

#remove all the rows which start with "OPT" because they are duplications of the original catalog
merged = merged.loc[:, ~merged.columns.str.startswith('OPT')]

#somehow the matching is giving negative fluxes in the band where there is no detection 
#if there is a detection in the other band
#clean that up to make those negative fluxes = 0

merged.loc[merged['flux_chandra_2_10'] < 0, 'flux_chandra_2_10'] = 0
merged.loc[merged['flux_chandra_05_2'] < 0, 'flux_chandra_05_2'] = 0
```

```{code-cell} ipython3
#How many Chandra sources are there?
#How many Galex sources are there?

#make a new column which is a bool of existing chandra measurements
merged['cosmos_chandra_detect'] = 0
merged.loc[merged.flux_chandra_2_10 > 0,'cosmos_chandra_detect']=1

#make one for Galex too
merged['galex_detect'] = 0
merged.loc[merged.flux_galex_nuv > 0,'galex_detect']=1

#make chandra hardness ratio column:
#hard = 'flux_chandra_2_10', soft = flux_chandra_05_2, HR = (H-S)/(H+S)
merged['chandra_HR'] = (merged['flux_chandra_2_10'] - merged['flux_chandra_05_2']) / (merged['flux_chandra_2_10'] + merged['flux_chandra_05_2'])


print('number of Chandra detections =',np.sum(merged.cosmos_chandra_detect > 0))
print('number of Galex detections =',np.sum(merged.galex_detect > 0))
```

## 5. Plot Final Results
- We want to understand something about populations based on their colors

```{code-cell} ipython3
#IRAC color color plots akin to Lacy et al. 2004
#overplot galex sources
#overplot xray sources

#first select on 24 micron 
merged_24 = merged[(merged.flux_24 >= 0) ] 

#negative Galex fluxes are causing problems, so set those to zero
merged_24.loc[merged_24.fuvflux < 0, 'fuvflux'] = 0
merged_24.loc[merged_24.nuvflux < 0, 'nuvflux'] = 0

#make color columns
merged_24['F5.8divF3.6'] = merged_24.ch3flux / merged_24.ch1flux
merged_24['F8.0divF4.5'] = merged_24.ch4flux / merged_24.ch2flux

#detected in all IRAC bands
merged_allirac = merged_24[(merged_24['F8.0divF4.5'] > 0) & (merged_24['F5.8divF3.6'] > 0)]

#plot all the points
fig, ax = plt.subplots()
sns.scatterplot(data = merged_allirac, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, alpha = 0.5, label = 'all')

#plot only those points with Galex detections
galex_detect = merged_allirac[merged_allirac.galex_detect > 0]
sns.scatterplot(data = galex_detect, x = 'F5.8divF3.6', y = 'F8.0divF4.5',
                 ax = ax, alpha = 0.5, label = 'Galex detect')

#plot only those points with chandra detections
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

```{code-cell} ipython3
#UV IR color color plot akin to Bouquin et al. 2015
fig, ax = plt.subplots()
merged['FUV-NUV'] = merged.mag_galex_fuv - merged.mag_galex_nuv
merged['NUV-3.6'] = merged.mag_galex_nuv - merged.splash_1_mag


#plot all the points
#sns.scatterplot(data = merged, x = 'NUV-3.6', y = 'FUV-NUV',
#                 ax = ax, alpha = 0.5)

#plot only those points with Galex detections
galex_detect = merged[merged.galex_detect > 0]
sns.kdeplot(data = galex_detect, x = 'NUV-3.6', y = 'FUV-NUV',
            ax = ax, fill = True, levels = 15)#scatterplot , alpha = 0.5

#plot only those points with chandra detections
#now with color coding Chandra sources by hardness ratio a la Moutard et al. 2020
chandra_detect = merged[merged.cosmos_chandra_detect > 0]
sns.scatterplot(data = chandra_detect, x = 'NUV-3.6', y = 'FUV-NUV',
                ax = ax, hue= 'chandra_HR',palette="flare")

#whew that legend for the hue is terrible
#try making it into a colorbar outside the plot instead
norm = plt.Normalize(merged['chandra_HR'].min(), merged['chandra_HR'].max())
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)

#ax.set(xscale="log", yscale="log")
ax.set_ylim([-0.5, 3.5])
ax.set_xlim([-1, 7])

ax.set(xlabel = 'NUV - [3.6]', ylabel = 'FUV - NUV')
#plt.legend([],[], frameon=False)

#fig.savefig("output/color_color.png")
#mpld3.display(fig)  
```

We extend the works of Bouquin et al. 2015 and Moutard et al. 2020 by showing a GALEX - Spitzer color color diagram over plotted with Chandra detections.  Blue galaxies in these colors are generated by O and B stars and so must currently be forming stars. We find a tight blue cloud in this color space identifying those star forming galaxies.  Galaxies off of the blue cloud have had their star formation quenched, quite possibly by the existence of an AGN through removal of the gas reservoir required for star formation.  Chandra detected galaxies host AGN, and while those are more limited in number, can be shown here to be a hosted by all kinds of galaxies, including quiescent galaxies which would be in the upper right of this plot.  This likely implies that AGN are indeed involved in quenching star formation.  Additionally, we show the Chandra hardness ratio (HR) color coded according to the vertical color bar on the right side of the plot.  Those AGN with higher hardness ratios have their soft x-ray bands heavily obscured and appear to reside preferentially toward the quiescent galaxies.

+++

## References

This work made use of:

- Astroquery; Ginsburg et al., 2019, 2019AJ....157...98G

- Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013, 2022ApJ...935..167A, 2018AJ....156..123A, 2013A&A...558A..33A

- The Tractor; Lang et al. 2016, 2016AJ....151...36L

- Nyland et al. 2017 , 2017ApJS..230....9N

- Salvato et al. 2018, 2018MNRAS.473.4937S

- Laigle et al. 2016, 2016ApJS..224...24L

```{code-cell} ipython3

```
