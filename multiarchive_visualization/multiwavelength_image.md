---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: py-multiwavelength_image
  display_name: py-multiwavelength_image
  language: python
authors:
  - name: David J Turner
---

# Create interactive multi-wavelength images of astronomical sources

## Learning Goals

***NEED MORE ENTRIES***

By the end of this tutorial, you will be able to:
- Reproject images from different missions to a common coordinate grid.
- Create interactive visualizations of individual reprojected images.
- Interactively explore how RGB images can be made from multi-wavelength observations.

## Introduction

***SOMETHING SOMETHING ALL THREE ARCHIVES, SOMETHING SOMETHING DIFFERENT WAVELENGTHS HIGHLIGHT DIFFERENT PROCESSES***

### Input

- The name of the target
- ***CHOICES OF INSTRUMENT????***


### Output

- 

### Runtime

As of 15th April 2026, this notebook takes ***HOW LONG***-minutes to run to completion on Fornax using the small server with 8GB RAM/ 2 CPU.

This demonstration acquires data from remote services, and as such the runtime can vary depending on the state of those services, and the speed of your internet connection (if running locally).

## Imports

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_multiwavelength_images.txt
```

```{code-cell} ipython3
import os
os.environ['KMP_WARNINGS'] = '0' # Silences the OpenMP warning
import sys

import panel as pn
import holoviews as hv
pn.extension(loading_spinner='dots', loading_color='#00aa41', comms='ipywidgets', inline=True)
hv.extension('bokeh')

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from astroquery.ipac.irsa import Irsa

import numpy as np

# Add local code directory to path
sys.path.append('code_src/')

# Import our custom functions
from archive_queries import (
    load_chandra_image, load_spitzer_image,
    query_hst, download_hst,
    query_swift, download_swift
)
from image_processing import (
    get_pixel_scale, reproject_to_common_grid,
    
)
from plotting import InteractiveRGBPanel, InteractiveMultiPanel
```

+++

## 1. Choosing the source to visualize and setting up directories

We start by specifying which astronomical object we want to visualize.

The default target is the Crab Nebula, but you can change this to any object resolvable 
by name through the SIMBAD or NED databases for instance:

- ***Suggestion 1 that we have vetted***
- ***Suggestion 2 ...***
- ***You get the idea...***

```{code-cell} ipython3
# Define the target
SOURCE_NAME = "Crab"

# Resolve coordinates from name
SOURCE_COORD = SkyCoord.from_name(SOURCE_NAME)

print(f"{SOURCE_NAME} Coordinate:".upper())
print(SOURCE_COORD.to_string())
```

Set up directories for downloaded data.
The data directory will store raw files from each archive, while the output directory will store processed images.

```{code-cell} ipython3
# Setting up the directory structure
ROOT_DATA_DIR = "data/"

# Separate directories for each mission
CHAN_DATA_DIR = os.path.join(ROOT_DATA_DIR, "HEASARC", "Chandra")
SPITZER_DATA_DIR = os.path.join(ROOT_DATA_DIR, "IRSA", "Spitzer")
HST_DATA_DIR = os.path.join(ROOT_DATA_DIR, "MAST", "Hubble")
SWIFT_DATA_DIR = os.path.join(ROOT_DATA_DIR, "MAST", "Swift")

for directory in [CHAN_DATA_DIR, SPITZER_DATA_DIR, HST_DATA_DIR, SWIFT_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)
    
# Now where any outputs will be stored
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

```{code-cell} python
CHANDRA_SEARCH_RAD = Quantity(3, 'arcmin')
SPITZER_SEARCH_RAD = Quantity(3, 'arcmin')

VETTED_OBS = {"crab": {'Chandra': "1994", "Hubble": "JC6801010", "Swift": "00030371012", "Spitzer": "50059401.50059401-10.IRAC"}, }
```

+++

## 2. Query archives for available data

### 2.1 Query HEASARC for Chandra X-ray observations

```{code-cell} python
all_chandra_obs = Heasarc.query_region(SOURCE_COORD, 'chanmaster', column_filters={"detector": ["ACIS-S", "ACIS-I"], "grating": "NONE"}, columns='*', radius=CHANDRA_SEARCH_RAD)
all_chandra_obs['time'] = Time(all_chandra_obs['time'], format='mjd').datetime
all_chandra_obs.sort('exposure', reverse=True)

all_chandra_obs
```

```{code-cell} python
# TODO THIS NEEDS TO HAVE EXCEPTION CATCHING

if SOURCE_NAME.lower() in VETTED_OBS and "Chandra" in VETTED_OBS[SOURCE_NAME.lower()]:
    chandra_obs_id = VETTED_OBS[SOURCE_NAME.lower()]["Chandra"]
else:
    chandra_obs_id = all_chandra_obs[0]['obsid']
    
sel_chandra_obs = all_chandra_obs[all_chandra_obs['obsid'] == int(chandra_obs_id)]

sel_chandra_datalink = Heasarc.locate_data(sel_chandra_obs)['aws']
```

```{code-cell} python
chandra_hdu = load_chandra_image(sel_chandra_datalink, preproc_cent_hi_res=True)
```


### 2.2 Query IRSA for infrared data

```{code-cell} python
all_spitzer_ims = Irsa.query_sia(pos=(SOURCE_COORD, SPITZER_SEARCH_RAD), facility="Spitzer Space Telescope", 
                             data_type="image", instrument='IRAC', res_format='image/fits', calib_level=3)
sel_spitzer_ims = all_spitzer_ims[all_spitzer_ims['dataproduct_subtype'] == 'science']
sel_spitzer_ims
```

```{code-cell} python
sel_spitzer_ims = sel_spitzer_ims[sel_spitzer_ims['s_resolution'] == sel_spitzer_ims['s_resolution'].min()]
```

```{code-cell} python
# Filter to mean mosaics (exclude short HDR exposures and median mosaics)
not_short_median_filt = (
    (~np.char.find(sel_spitzer_ims['access_url'].data.astype(str), 'short') > -1) &
    (~np.char.find(sel_spitzer_ims['access_url'].data.astype(str), 'median') > -1)
)

sel_spitzer_ims = sel_spitzer_ims[not_short_median_filt]

sel_spitzer_ims.sort('dist_to_point')
sel_spitzer_ims = sel_spitzer_ims[0]

sel_spitzer_ims
```

```{code-cell} ipython3
spitzer_hdu = load_spitzer_image(sel_spitzer_ims['cloud_access'])
```

### 2.3 Query MAST for optical data

Hubble's Advanced Camera for Surveys provides extremely high resolution optical imaging.
We search for ACS/WFC observations with the F550M filter, which is sensitive to green-yellow light.

```{code-cell} ipython3
print("Querying MAST for Hubble observations...")

hst_obs = query_hst(
    SOURCE_COORD,
    instrument="ACS",
    aperture="WFC",
    filter_spec="F550M;CLEAR2L",  # Both filter wheel elements
    min_exposure=1000
)

if hst_obs is not None and len(hst_obs) > 0:
    print(f"Found {len(hst_obs)} Hubble observations")

    # Select a specific observation for Crab
    if SOURCE_NAME.lower() == "crab":
        selected_hst_dataset = "JC6801010"
        print(f"Selected dataset {selected_hst_dataset}")
    else:
        selected_hst_dataset = hst_obs['sci_data_set_name'][0]
        print(f"Selected dataset {selected_hst_dataset}")
else:
    print("No Hubble data found for this target")
    selected_hst_dataset = None
```

### 2.4 Query MAST for ultraviolet data

Swift's UV/Optical Telescope provides ultraviolet imaging.
We search for observations with the UVW2 filter, which covers near-ultraviolet wavelengths.

```{code-cell} ipython3
print("Querying MAST for Swift observations...")

swift_obs = query_swift(SOURCE_COORD, filter_name='UVW2')

if swift_obs is not None and len(swift_obs) > 0:
    print(f"Found {len(swift_obs)} Swift observations")
    print(f"Longest exposure: {swift_obs['t_exptime'][0]:.0f} seconds")

    # Select observation
    if SOURCE_NAME.lower() == "crab":
        selected_swift_obsid = "00030371012"
        print(f"Selected ObsID {selected_swift_obsid}")
    else:
        selected_swift_obsid = swift_obs['obs_id'][0]
        print(f"Selected longest exposure (ObsID {selected_swift_obsid})")
else:
    print("No Swift data found for this target")
    selected_swift_obsid = None
```

+++

## 3. Download and load image data

Now that we have identified suitable observations, we download the data products.
Some archives provide direct cloud access, which is faster than traditional downloads.

### 3.1 Download Chandra data


### 3.2 Load Spitzer data from the IRSA S3 bucket


### 3.3 Download Hubble data

```{code-cell} ipython3
if selected_hst_dataset is not None:
    print("Downloading Hubble data...")
    hst_img_path = download_hst(hst_obs, HST_DATA_DIR, dataset_name=selected_hst_dataset)

    if hst_img_path and os.path.exists(hst_img_path):
        print(f"Hubble data downloaded to: {hst_img_path}")
        hst_hdu = fits.open(hst_img_path)
        print(f"Image shape: {hst_hdu['SCI'].data.shape}")
    else:
        print("Failed to download or locate Hubble data")
        hst_hdu = None
else:
    print("Skipping Hubble download (no data available)")
    hst_hdu = None
```

### 3.4 Download Swift data

```{code-cell} ipython3
if selected_swift_obsid is not None:
    print("Downloading Swift data...")
    swift_img_path = download_swift(swift_obs, SWIFT_DATA_DIR, obs_id=selected_swift_obsid)

    if swift_img_path and os.path.exists(swift_img_path):
        print(f"Swift data downloaded to: {swift_img_path}")
        swift_hdu = fits.open(swift_img_path)
        print(f"Image shape: {swift_hdu[1].data.shape}")
    else:
        print("Failed to download or locate Swift data")
        swift_hdu = None
else:
    print("Skipping Swift download (no data available)")
    swift_hdu = None
```

+++

## 4. Reproject images to a common coordinate grid

Different telescopes have vastly different spatial resolutions (pixel sizes).
Before we can combine images, we must reproject them all to a common coordinate grid.

We use the highest resolution image (typically Hubble) as our reference coordinate system.
Lower resolution images are upsampled to match, while the high resolution image remains at its native scale.

### 4.1 Compare pixel scales

```{code-cell} ipython3
# Dictionary mapping missions to their relevant HDU extensions
mission_hdus = {
    'Chandra': chandra_hdu['PRIMARY'] if chandra_hdu else None,
    'Hubble': hst_hdu['SCI'] if hst_hdu else None,
    'Swift': swift_hdu[1] if swift_hdu else None,
    'Spitzer': spitzer_hdu['PRIMARY'] if spitzer_hdu else None
}

pixel_scales = {miss: get_pixel_scale(miss_hdu) for miss, miss_hdu in mission_hdus.items() if miss_hdu is not None}
finest_pix_miss = min(pixel_scales, key=pixel_scales.get)

print("-" * 50)
print(f"{finest_pix_miss:<10}: {pixel_scales[finest_pix_miss].to('arcsec/pix'):.3f} [x1.00]")
tab_strs = {}
for cur_miss, cur_scale in pixel_scales.items():
    if cur_miss == finest_pix_miss:
        continue  
    coarse_factor = (cur_scale / pixel_scales[finest_pix_miss]).value.round(1)
    tab_strs[coarse_factor] = f"{cur_miss:<10}: {cur_scale.to('arcsec/pix'):.3f} [x{coarse_factor}]\n"
    
print("".join(dict(sorted(tab_strs.items())).values()))
print("-" * 50)
```

+++

### 4.2 Reproject all images

We have to reproject the images to a common coordinate grid, giving them a matching 
spatial resolution for when we make visualizations. 

If we were trying to do science with these images directly, we would likely not 
reproject them.

Here we fetch out the highest resolution image's FITS header, to provide the
reprojection grid for all images (note though that we will not reproject the donor
image):
```{code-cell} ipython3
donor_wcs_hdr = mission_hdus[finest_pix_miss].header.copy()
```


```{code-cell} ipython3
reproj_data_cov = {mn: {"data": (rp := reproject_to_common_grid((cur_hdu.data, cur_hdu.header), donor_wcs_hdr))[0], "cov": ((rp[1] > 0).sum() / rp[1].size)} for mn, cur_hdu in mission_hdus.items() if cur_hdu is not None}

for cur_miss, rp_info in reproj_data_cov.items():
    if rp_info['data'] is not None:
        print(f"{cur_miss}: {rp_info['data'].shape}, coverage = {rp_info['cov'] * 100:.1f}%")
```

+++

## 5. Visualize individual wavelength images



```{code-cell} ipython3
sep_reproj_im_cmaps = {
    'Hubble': 'Greens',
    'Swift': 'Blues',
    'Chandra': 'Purples',
    'Spitzer': 'YlOrBr'
}
```

```{code-cell} ipython3
sep_ims = InteractiveMultiPanel({mn: res['data'] for mn, res in reproj_data_cov.items()}, sep_reproj_im_cmaps)
sep_ims.view()
```

+++

## 6. Interactive multi-wavelength image


***IMPROVE ALL THIS***

Finally, we combine three wavelength bands into a single RGB composite image.
We use:
- Red channel: Spitzer infrared (cool dust and molecular gas)
- Green channel: Hubble optical (stellar emission and ionized gas)
- Blue channel: Chandra X-ray (extremely hot plasma and high-energy particles)


The interactive controls allow you to adjust:
- **Channel percentiles**: Control the brightness scaling for each color
- **Lupton Q**: Softening parameter (higher values show more faint detail)
- **Lupton stretch**: Overall brightness (higher values make the image brighter)

***Experiment with the sliders...***

```{code-cell} ipython3
# Extract the three channels
red_channel = reproj_data_cov['Spitzer']['data']
green_channel = reproj_data_cov['Hubble']['data']
blue_channel = reproj_data_cov['Chandra']['data']

multi_wav_im = InteractiveRGBPanel(red_channel, green_channel, blue_channel)
multi_wav_im.view()
```

+++

## About this Notebook

**Authors:** David J Turner

**Contact:** For help with this notebook, please open a topic in the [Fornax Helpdesk](https://discourse.fornax.sciencecloud.nasa.gov/c/helpdesk).

+++

### Acknowledgements

This notebook was created with assistance from:
- The Fornax team
- The High Energy Astrophysics Science Archive Research Center (HEASARC) team
- The NASA/IPAC Infrared Science Archive (IRSA) team
- The Barbara A. Mikulski Archive for Space Telescopes (MAST) team

AI: This notebook was created with assistance from Google's Gemini models and NASA ChatGSFC.

### References

This work made use of:
- [Chandra X-ray Observatory Data Archive](https://cxc.harvard.edu/cda/)
- [Spitzer Heritage Archive at IRSA](https://irsa.ipac.caltech.edu/Missions/spitzer.html)
- [Mikulski Archive for Space Telescopes (MAST)](https://archive.stsci.edu/)
- [AstroPy: A Community Python Library for Astronomy](https://www.astropy.org/)
- [AstroQuery: Access to Online Astronomical Data](https://astroquery.readthedocs.io/)
- [Reproject: Image Reprojection in Python](https://reproject.readthedocs.io/)
- [Lupton et al. 2004, PASP, 116, 133: "Preparing Red-Green-Blue Images from CCD Data"](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L)
- STScI style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
- Fornax tech and science review guidelines: https://github.com/nasa-fornax/fornax-demo-notebooks/blob/main/template/notebook_review_checklists.md
- The Turing Way Style Guide: https://book.the-turing-way.org/community-handbook/style
