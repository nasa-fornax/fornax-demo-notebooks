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

```{code-cell} python
# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_multiwavelength_images.txt
```

```{code-cell} python
import os
os.environ['KMP_WARNINGS'] = '0' # Silences the OpenMP warning
import sys

import panel as pn
import holoviews as hv
pn.extension(loading_spinner='dots', 
             loading_color='#00aa41', 
             comms='ipywidgets', 
             inline=True)
hv.extension('bokeh')

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from astroquery.ipac.irsa import Irsa
from astroquery.mast import Observations

import numpy as np

# Add local code directory to path
sys.path.append('code_src/')

# Import our custom functions
from archive_queries import (vetted_source_check, load_chandra_image, 
                             load_spitzer_image, load_hubble_image, load_swift_image)
from image_processing import get_pixel_scale, reproject_to_common_grid
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

```{code-cell} python
# Define the target
SOURCE_NAME = "Crab"

# Resolve coordinates from name
SOURCE_COORD = SkyCoord.from_name(SOURCE_NAME)

print(f"{SOURCE_NAME} Coordinate:".upper())
print(SOURCE_COORD.to_string())
```

```{code-cell} python
CHANDRA_SEARCH_RAD = Quantity(3, 'arcmin')
SPITZER_SEARCH_RAD = Quantity(3, 'arcmin')
SWIFT_SEARCH_RAD = Quantity(5, 'arcmin')
HUBBLE_SEARCH_RAD = Quantity(2, 'arcmin')
```

+++

## 2. Query archives for available data

### 2.1 Query HEASARC for Chandra X-ray observations

```{code-cell} python
chandra_obs_id = vetted_source_check(SOURCE_NAME, "Chandra")
chandra_obs_id
```

```{code-cell} python
all_chandra_obs = Heasarc.query_region(SOURCE_COORD, 'chanmaster', column_filters={"detector": ["ACIS-S", "ACIS-I"], "grating": "NONE"}, columns='*', radius=CHANDRA_SEARCH_RAD)
all_chandra_obs['time'] = Time(all_chandra_obs['time'], format='mjd').datetime
all_chandra_obs.sort('exposure', reverse=True)

all_chandra_obs
```

```{code-cell} python
# TODO THIS NEEDS TO HAVE EXCEPTION CATCHING
if chandra_obs_id is None:
    chandra_obs_id = all_chandra_obs[0]['obsid']

sel_chandra_obs = all_chandra_obs[all_chandra_obs['obsid'] == int(chandra_obs_id)]

sel_chandra_datalink = Heasarc.locate_data(sel_chandra_obs)['aws']
```

```{code-cell} python
chandra_hdu = load_chandra_image(sel_chandra_datalink, preproc_cent_hi_res=True)
```


### 2.2 Query IRSA for infrared data

```{code-cell} python
spitzer_obs_id = vetted_source_check(SOURCE_NAME, "Spitzer")
spitzer_obs_id
```

```{code-cell} python
all_spitzer_ims = Irsa.query_sia(pos=(SOURCE_COORD, SPITZER_SEARCH_RAD), facility="Spitzer Space Telescope", 
                             data_type="image", instrument='IRAC', res_format='image/fits', calib_level=3)
del all_spitzer_ims['s_region']
del all_spitzer_ims['proposal_title']

all_spitzer_ims = all_spitzer_ims[all_spitzer_ims['dataproduct_subtype'] == 'science']
all_spitzer_ims
```

```{code-cell} python
sel_spitzer_ims = all_spitzer_ims[all_spitzer_ims['s_resolution'] == all_spitzer_ims['s_resolution'].min()]

if spitzer_obs_id is not None:
    sel_spitzer_ims = sel_spitzer_ims[sel_spitzer_ims['obs_id'] == spitzer_obs_id]
    
sel_spitzer_ims
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

```{code-cell} python
spitzer_hdu = load_spitzer_image(sel_spitzer_ims)
```

### 2.3 Query MAST for optical data

```{code-cell} python
hubble_obs_id = vetted_source_check(SOURCE_NAME, "Hubble")
hubble_obs_id
```

```{code-cell} python
all_hubble_obs = Observations.query_criteria(
    coordinates=SOURCE_COORD,
    obs_collection='HST',
    dataproduct_type='image',
    obs_id="*" if hubble_obs_id is None else hubble_obs_id.lower(),
    wave_region="OPTICAL",
    intentType='science',
    dataRights='PUBLIC',
    radius=HUBBLE_SEARCH_RAD
)

del all_hubble_obs['s_region']

all_hubble_obs = all_hubble_obs[all_hubble_obs['t_exptime'] > 0]

all_hubble_obs.sort('t_exptime', reverse=True)

all_hubble_obs
```

```{code-cell} python
sel_hubble_obs = all_hubble_obs[0]
```

```{code-cell} python
sel_hubble_prods = Observations.get_unique_product_list(sel_hubble_obs)
sel_hubble_prods
```

```{code-cell} python
sel_hubble_im = Observations.filter_products(sel_hubble_prods, productType='science', productSubGroupDescription="DRC",  mrp_only=True)
sel_hubble_im
```

```{code-cell} python
hubble_hdu = load_hubble_image(sel_hubble_im)
```


### 2.4 Query MAST for ultraviolet data

Swift's UV/Optical Telescope provides ultraviolet imaging.

We search for observations with the UVW2 filter, which covers near-ultraviolet wavelengths.


```{code-cell} python
swift_obs_id = vetted_source_check(SOURCE_NAME, "Swift")
swift_obs_id
```

```{code-cell} python
all_swift_obs = Observations.query_criteria(
    coordinates=SOURCE_COORD,
    radius=SWIFT_SEARCH_RAD,
    obs_collection='SWIFT',
    filters=["UVW2", "UVM2", "UVW1],
    calib_level=2,
)

del all_swift_obs['s_region']
all_swift_obs.sort('t_exptime', reverse=True)

all_swift_obs
```

```{code-cell} python
if swift_obs_id is None:
    sel_swift_obs = all_swift_obs[0]
else:
    sel_swift_obs = all_swift_obs[all_swift_obs['obs_id'] == swift_obs_id]
sel_swift_obs
```

```{code-cell} python
swift_hdu = load_swift_image(sel_swift_obs)
```

+++

## 3. Reproject images to a common coordinate grid

Different telescopes have vastly different spatial resolutions (pixel sizes).
Before we can combine images, we must reproject them all to a common coordinate grid.

We use the highest resolution image (typically Hubble) as our reference coordinate system.
Lower resolution images are upsampled to match, while the high resolution image remains at its native scale.

### 3.1 Compare pixel scales

```{code-cell} python
# Dictionary mapping missions to their relevant HDU extensions
mission_hdus = {
    'Chandra': chandra_hdu['PRIMARY'] if chandra_hdu else None,
    'Hubble': hubble_hdu['SCI'] if hubble_hdu else None,
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

### 3.2 Reproject all images

We have to reproject the images to a common coordinate grid, giving them a matching 
spatial resolution for when we make visualizations. 

If we were trying to do science with these images directly, we would likely not 
reproject them.

Here we fetch out the highest resolution image's FITS header, to provide the
reprojection grid for all images (note though that we will not reproject the donor
image):
```{code-cell} python
donor_wcs_hdr = mission_hdus[finest_pix_miss].header.copy()
```


```{code-cell} python
reproj_data_cov = {mn: {"data": (rp := reproject_to_common_grid((cur_hdu.data, cur_hdu.header), donor_wcs_hdr))[0], "cov": ((rp[1] > 0).sum() / rp[1].size)} for mn, cur_hdu in mission_hdus.items() if cur_hdu is not None and mn != finest_pix_miss}

for cur_miss, rp_info in reproj_data_cov.items():
    if rp_info['data'] is not None:
        print(f"{cur_miss}: {rp_info['data'].shape}, coverage = {rp_info['cov'] * 100:.1f}%")
        
reproj_data_cov[finest_pix_miss] = {'data': mission_hdus[finest_pix_miss].data}
reproj_data_cov = {mn: reproj_data_cov[mn] for mn in mission_hdus if mission_hdus[mn] is not None}
```

+++

## 4. Visualize individual wavelength images



```{code-cell} python
sep_reproj_im_cmaps = {
    'Hubble': 'Greens',
    'Swift': 'Blues',
    'Chandra': 'Purples',
    'Spitzer': 'YlOrBr'
}
```

```{code-cell} python
sep_ims = InteractiveMultiPanel({mn: res['data'] for mn, res in reproj_data_cov.items()}, sep_reproj_im_cmaps)
sep_ims.view()
```

+++

## 5. Interactive multi-wavelength image


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

```{code-cell} python
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
