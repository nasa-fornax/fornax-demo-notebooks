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

By the end of this tutorial, you will be able to:
- Find Chandra, Hubble, Spitzer, and Swift observations of a named source.
- Reproject images from different missions to a common coordinate grid.
- Create interactive visualizations of individual reprojected images.
- Interactively explore how 'pretty' RGB images can be made from multi-wavelength observations.

## Introduction

This notebook is intended as a gentle introduction to the sort of things you can do 
on the Fornax Science Console, while also providing a demonstration of how you might
begin to make visually appealing multi-wavelength images of astronomical sources (though 
the author makes no claims as to having any _good_ sense of aesthetics – so you're on 
your own there).

We'll be searching the NASA astrophysics archives for observations of named 
sources, with a set of suggested objects to try out, and setting up interactive 
visualizations where you can adjust how different wavelengths are combined and
see what makes the nicest looking image.

Throughout this demonstration we make use of the Python module Astroquery to search and 
acquire data from the HEASARC, MAST, and IRSA archives, while the HoloViz, Panel, and 
Bokeh libraries are used to create interactive figures that are performant even with
the thousands-of-pixels-per-side images we will be dealing with.


### Input
- The name of the target source; *we provide a few suggestions in the first section of this notebook.*

### Output

- Interactive multi-wavelength images of the target source.

### Runtime

As of 21st April 2026, this notebook takes approximately 8 minutes to run to completion on Fornax using the small server with 8GB RAM/ 2 CPU, with the default settings.

This demonstration acquires data from remote services, and as such the runtime can vary depending on the state of those services and the speed of your internet connection (if running locally).

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

from IPython.display import display

from astropy import conf
conf.max_lines = 6

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

## 1. Choosing the object we want to visualize

We start by choosing the source for which we want to create a multi-wavelength image; as we are 
using the `SkyCoord.from_name(...)` method to initialize the SkyCoord object, you may any name that 
can be identified by the CDS name resolver service. Alternatively, you could manually set up 
a SkyCoord from the RA-Dec (or other coordinate system) of your source of interest.

To get started, we provide a few suggestions to try out:
- ***The Crab Nebula*** **[Crab; default]** – One of the most observed sources in the Milky Way, a favorite for calibrating space observatories, and famously visually striking.
- ***Messier 61*** – Also known as the 'Swelling Spiral Galaxy', M61 is one of the largest members of the Virgo galaxy cluster and has played host to 8 observed supernovae since 1926 (a considerable number). 
- ***Kepler's Supernova*** **[SN 1604]** – The remnant of the most recent supernova observed with the naked eye (in 1604).
- ***NGC 4753*** - A lenticular galaxy, discovered by William Herschel in 1784, with eye-catching dust lanes.
- ***Abell 370*** - A galaxy cluster known for several prominent strong-lensing arc features.

```{code-cell} python
# Define the target
SOURCE_NAME = "Crab"

# Other source suggestions - uncomment them (and comment out the others) to try them out.
# SOURCE_NAME = "M61"
# SOURCE_NAME = "SN1604"
# SOURCE_NAME = "NGC4753"
# SOURCE_NAME = "A370"

# Resolve coordinates from name
SOURCE_COORD = SkyCoord.from_name(SOURCE_NAME)

print(f"{SOURCE_NAME} Coordinate:".upper())
print(SOURCE_COORD.to_string())
```

```{code-cell} python


SWIFT_SEARCH_RAD = Quantity(5, 'arcmin')
```

+++

## 2. Searching NASA's astrophysical archives for images of our source

Our goal is to be able to compare and combine images of the same source, taken in 
different wavelength bands. Some space telescopes (Hubble, for instance) can take 
observations in several, usually adjacent, wavelength bands. However, for this 
notebook, and to explore a wider range of physical processes highlighted by different
wavelengths of light, we will search for observations taken by four **different** 
observatories:
- Chandra [X-ray photons]
- Swift [UV photons]
- Spitzer Infrared Telescope [Infrared photons]
- Hubble Space Telescope [Optical photons]

### 2.1 Query HEASARC for Chandra X-ray observations

Chandra has the highest angular resolution of any X-ray telescope, which is of particular
importance given that we want to make nice visualizations of our target, and that, as 
high-energy photons are much harder to focus than their lower-energy cousins, even 
Chandra's spatial resolution is significantly worse than that of Hubble.

The High Energy Astrophysics Science Archive Research Center (HEASARC) provides 
access to the full public set of Chandra observations, and we'll use the `Heasarc` 
object imported from the `astroquery.heasarc` module to search for observations of 
current target.

First, though, we can check if the current target source has been pre-vetted to identify
a good observation to use for this demonstration. If it has, then the `chandra_obs_id`
variable will be set to the relevant ObsID, and if not it will be set to `None`:

```{code-cell} python
chandra_obs_id = vetted_source_check(SOURCE_NAME, "Chandra")
chandra_obs_id
```

Now, we need to search for observations of our current target – note though that if 
there is a pre-vetted observation for the source, we won't consider any other 
observations that might be relevant to the source.

We set a search radius of $3^{\prime}$, which should help exclude any observations 
where the source is not on or near the CCD chip positioned at the aimpoint – you could
experiment with adjusting this radius if you aren't finding observations of your 
source:

```{code-cell} python
CHANDRA_SEARCH_RAD = Quantity(3, 'arcmin')
```

Our search then begins by defining filters, to specify which Chandra instruments we might
want to consider using for our visualizations:
- **ACIS-S/I** - Chandra's CCD spectro-imagers, these are much more commonly used than HRC-I/S (the other Chandra instruments).
- **No grating** – If a grating was deployed, then the resulting 'image' would actually be a dispersed spectrum trace, rather than an image of the source.

Note that pre-vetted ObsID is added to the filters, if available.

Following the definition of our filters, we pass the coordinate of the source (specified 
by `SOURCE_COORD`), the name of the Chandra 'master' catalog (_chanmaster_), our 
filters, a `columns='*'` argument to force return of all _chanmaster_ columns, and the
search radius.

Once we have the results of our search (though don't forget that there might not be 
_any_ observations of your source of interest), we sort them so that the first entries
in the table have the longest exposures. 

Exposure time is not necessarily a good metric for determining the quality of a Chandra
observation, particularly those taken by the ACIS detectors, which have degraded in
sensitivity significantly since the beginning of the mission, but it is the best
we have on hand.

```{code-cell} python
search_filt = {"detector": ["ACIS-S", "ACIS-I"], 
               "grating": "NONE"}

# If a pre-vetted observation exists, we'll add an extra filter to only select that
#  one observation.              
search_filt.update({} if chandra_obs_id is None else {"obsid": chandra_obs_id})

all_chandra_obs = Heasarc.query_region(SOURCE_COORD, 
                                       catalog='chanmaster', 
                                       column_filters=search_filt, 
                                       columns='*', 
                                       radius=CHANDRA_SEARCH_RAD)
                                       
all_chandra_obs.sort('exposure', reverse=True)

all_chandra_obs
```

If there are any observations of the target source, we will select the first entry, 
with the longest exposure time; you could also experiment with choosing other 
observations by filtering the `all_chandra_obs` table yourself.

From there, we use another `Heasarc` function to fetch the 'data link' for our chosen
observation, which will tell us where we can fetch the data files from:

```{code-cell} python
if len(all_chandra_obs) > 0:
    sel_chandra_datalink = Heasarc.locate_data(all_chandra_obs[0])
else:
    sel_chandra_datalink = None
```

Now we use a convenience function (see the `code_src/archive_queries.py` file for the 
definition) to use the data link to fetch and load the Chandra image:

```{code-cell} python
chandra_hdu = load_chandra_image(sel_chandra_datalink, preproc_cent_hi_res=True)
```


### 2.2 Query IRSA for infrared data

For infrared data, we turn to the Spitzer space telescope; Spitzer is one of NASA's 
'great observatories' (alongside Chandra, Hubble, and Compton) and provides infrared 
imaging (and spectroscopy, but that isn't relevant to this demonstration) in the 
mid/far-IR bands.

Unlike the Chandra and Hubble great observatories that we're also using in this 
tutorial, Spitzer is no longer active, but its data are all maintained and served by 
the NASA/IPAC Infrared Science Archive (IRSA). 

We can use the `Irsa` object imported from the `astroquery.ipac.irsa` submodule to 
find Spitzer images of our target source. Our approach here is a little different
to how we found the Chandra data, however – as each archive creates and maintains its 
own Astroquery submodule, the capabilities and interface tend to vary.

For IRSA, we're going to use a 'Simple Image Access' (SIA) function, which can search
for specific _image_ products, rather than whole observations. 

Once again, we check to see if a particular observation has been pre-vetted for the 
current source - `spitzer_obs_id` will be set to the relevant ObsID, if it exists, 
otherwise it will be set to `None`:

```{code-cell} python
spitzer_obs_id = vetted_source_check(SOURCE_NAME, "Spitzer")
spitzer_obs_id
```

We define a radius within which we will search for Spitzer images of our target source:

```{code-cell} python
SPITZER_SEARCH_RAD = Quantity(3, 'arcmin')
```

For simplicity, we're only going to search for images taken by one of Spitzer's 
instruments – the Infrared Array Camera (IRAC). We pass the source coordinate and
search radius, specify that we want image products, and particularly those that have
a calibration level of **2** (science ready calibrated image mosaics) or **3** (enhanced 
data products often made up of data from multiple visits):

```{code-cell} python
all_spitzer_ims = Irsa.query_sia(pos=(SOURCE_COORD, SPITZER_SEARCH_RAD), 
                                 facility="Spitzer Space Telescope", 
                                 data_type="image", 
                                 instrument='IRAC', 
                                 res_format='image/fits', 
                                 calib_level=[2, 3])

# These columns are removed to make the table look nicer when shown in the notebook
del all_spitzer_ims['s_region']
del all_spitzer_ims['proposal_title']

# If a pre-vetted observation exists for the target, we'll filter out all 
#  products related to other observations.
if spitzer_obs_id is not None:
    all_spitzer_ims = all_spitzer_ims[all_spitzer_ims['obs_id'] == spitzer_obs_id]

# Don't want any calibration products or any such, just science products
all_spitzer_ims = all_spitzer_ims[all_spitzer_ims['dataproduct_subtype'] == 'science']
all_spitzer_ims
```

If Spitzer images are available, we automatically select the one with the highest calibration level, and the 
highest spatial resolution, available, that is centered on a point closest to our target. Again, this isn't a 
particularly good way to determine which data to use, but it means we can make this tutorial work for
any target source:

```{code-cell} python
if len(all_spitzer_ims) > 0:
    # Select the highest calibration level available
    filt_spitzer_ims = all_spitzer_ims[all_spitzer_ims['calib_level'] == all_spitzer_ims['calib_level'].max()]
    
    # Select the highest resolution images available
    filt_spitzer_ims = filt_spitzer_ims[filt_spitzer_ims['s_resolution'] == filt_spitzer_ims['s_resolution'].min()]
    
    # Filter to mean mosaics (exclude short HDR exposures and median mosaics) - if the
    #  products aren't mosaics at all then this has no effect
    not_short_median_filt = (
        (~np.char.find(filt_spitzer_ims['access_url'].data.astype(str), 'short') > -1) &
        (~np.char.find(filt_spitzer_ims['access_url'].data.astype(str), 'median') > -1)
    )
    filt_spitzer_ims = filt_spitzer_ims[not_short_median_filt]
    
    # Also sort by the distance from the target (closest at the top of the table)
    filt_spitzer_ims.sort('dist_to_point')
else:
    filt_spitzer_ims = None

# Show the table of products
filt_spitzer_ims
```

We use another convenience function (see the `code_src/archive_queries.py` file for the 
definition) to load the Spitzer image:

```{code-cell} python
sel_spitzer_im = None if filt_spitzer_ims is None else filt_spitzer_ims[0]
spitzer_hdu = load_spitzer_image(sel_spitzer_im)
```

### 2.3 Query MAST for optical data

For optical observations, we turn to the Hubble Space Telescope (HST), the third great 
observatory of this demonstration! Of course, Hubble is more than just an optical
telescope, as it also has some near-infrared and UV imaging and spectroscopic capabilities. 

Even our specification of 'optical' observations is a bit fuzzy, as not only has 
Hubble hosted four generations of scientific instruments with differing sensitives 
within what we consider to be the 'optical' bandpass, but each instrument has often
included a number of different wide and narrow filters that limit the bandpass further.

As the purpose of this demonstration is just to make nice visualizations and to be
relatively robust to choices of target other than those we suggest, we aren't going
to be picky about exactly which instrument or filter we use. So long as the data
are ostensibly in the 'optical' range, we'll take them.

Once again, we check to see if a particular observation has been pre-vetted for the 
current source - `hubble_obs_id` will be set to the relevant ObsID, if it exists, 
otherwise it will be set to `None`:

```{code-cell} python
hubble_obs_id = vetted_source_check(SOURCE_NAME, "Hubble")
hubble_obs_id
```

We define a radius within which we will search for optical images of our target source:

```{code-cell} python
HUBBLE_SEARCH_RAD = Quantity(2, 'arcmin')
```

Now we proceed to search the Hubble archive, maintained and served by 
Barbara A. Mikulski Archive for Space Telescopes (MAST), using the `Observations` class
imported from the `astroquery.mast` submodule. This search for observations will be 
followed up by a search for **data products** related to the observation we end up choosing.

We set `project="HST"` to avoid identifying 'Hubble Advanced Product' (HAP) results, as 
though the multi-visit-mosaics would be ideal for our purposes, they tend to be 
very large (both in storage and in the number of pixels in the image), which can
cause issues with running out of memory.

After the initial search for observations with image data, we filter out any that were
taken using Hubble's STIS instrument. We don't want to specify a list of instruments
to search, because Hubble has (and has had) so many. However, we specifically **do** 
wish to exclude STIS images because they are often just for targeting prior to 
spectroscopic observations and not high enough quality for our purposes.

```{code-cell} python
all_hubble_obs = Observations.query_criteria(
    coordinates=SOURCE_COORD,
    obs_collection='HST',
    project="HST",
    dataproduct_type='image',
    obs_id="*" if hubble_obs_id is None else hubble_obs_id.lower(),
    wave_region="OPTICAL",
    intentType='science',
    dataRights='PUBLIC',
    radius=HUBBLE_SEARCH_RAD
)

# These columns are removed to make the table look nicer when shown in the notebook
del all_hubble_obs['s_region']
del all_hubble_obs['obs_title']

# Don't want to filter on specific instruments, because Hubble has had so many, but the
#  STIS images are often just targetting images before spectroscopic observations, and
#  aren't useful for our purpose.
all_hubble_obs = all_hubble_obs[np.char.find(all_hubble_obs['instrument_name'].data, 'STIS') != 0]
all_hubble_obs = all_hubble_obs[all_hubble_obs['t_exptime'] > 0]

# Sort by exposure time, longest first
all_hubble_obs.sort('t_exptime', reverse=True)

all_hubble_obs
```

If any candidate observations have been identified, we will automatically select the
first entry in the table (which will have the longest exposure time, as we sorted by
it in the previous cell). You could easily modify this to select an observation
based on some other criteria, as the exposure time is no guarantee of quality.

Once an observation has been chosen, we run an additional search to retrieve the
table of **data products** associated with the observation, and then a filtering 
operation to pare down the list to just science imaging products:

```{code-cell} python
if len(all_hubble_obs) > 0:
    sel_hubble_obs = all_hubble_obs[0]
    
    sel_hubble_prods = Observations.get_unique_product_list(sel_hubble_obs)
    display(sel_hubble_prods)
    
    sel_hubble_im = Observations.filter_products(sel_hubble_prods, 
                                                 productType='SCIENCE', 
                                                 productSubGroupDescription=["DRC", "DRZ"], 
                                                 mrp_only=True)
    
    if "DRC" in sel_hubble_im['productSubGroupDescription'].value:
        sel_hubble_im = sel_hubble_im[sel_hubble_im['productSubGroupDescription'] == "DRC"]
    else:
        sel_hubble_im = sel_hubble_im[sel_hubble_im['productSubGroupDescription'] == "DRZ"]
    
    display(sel_hubble_im)
else:
    sel_hubble_im = None
```

Now we can load the Hubble image – see the `code_src/archive_queries.py` file for the 
definition of the `load_hubble_image` convenience function:

```{code-cell} python
hubble_hdu = load_hubble_image(sel_hubble_im)
```


### 2.4 Query MAST for ultraviolet data

Swift's UV/Optical Telescope provides ultraviolet imaging.

We search for observations that were taken with the UVW2, UVM2, and UVW1 filters, which 
cover parts of the Far/Mid-UV wavelength ranges. Swift's UVOT instrument also provides
filters with higher wavelength, optical, band passes, but we specifically want UV data here.


```{code-cell} python
swift_obs_id = vetted_source_check(SOURCE_NAME, "Swift")
swift_obs_id
```

```{code-cell} python
all_swift_obs = Observations.query_criteria(
    coordinates=SOURCE_COORD,
    radius=SWIFT_SEARCH_RAD,
    obs_collection='SWIFT',
    obs_id="*" if swift_obs_id is None else swift_obs_id,
    filters=["UVW2", "UVM2", "UVW1"],
    calib_level=2,
)

del all_swift_obs['s_region']
all_swift_obs.sort('t_exptime', reverse=True)

all_swift_obs
```

```{code-cell} python
if len(all_swift_obs) > 0:
    sel_swift_obs = all_swift_obs[0]
else:
    sel_swift_obs = None

swift_hdu = load_swift_image(sel_swift_obs)
```
+++

## 3. Reproject images to a common coordinate grid

The various telescopes whose images we've just acquired have, in some cases, wildly different 
angular (spatial) resolutions. Indeed, for several of these telescopes, the different instruments
also have disparate spatial resolutions, driven by the available technology and the instrument's
scientific purpose.

As such, we can't just plot the images directly over the top of each other and expect 
everything to line up properly. Of course, we can't expect that every feature we see in
one image will be present in the others, as the whole point of observing in different
wavelengths is to understand _different_ astrophysical processes going on in a source.

Here we use the image with the highest spatial resolution (almost certain to be the 
Hubble image if we managed to find one for the current source) as the reference 
coordinate grid. We'll refer to it as the 'donor' image.

That means that all other images will be aligned to the 'donor', and will have to be 
upscaled to match the donor's higher spatial resolution.


:::{note}
We would not necessarily recommend this interpolating and upscaling for images that 
are going to be used for scientific analyses, but given we're just making nice 
visualizations, it is fine to do it here.
:::

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

### 3.2 Reprojecting images to a common coordinate grid

We have to reproject the images to a common coordinate grid, giving them a matching 
spatial resolution for when we make visualizations. 

If we were trying to do science with these images directly, we would likely not 
reproject them.

Here we fetch out the highest resolution image's FITS header, from which we can fetch 
the information that will act as the common coordinate grid for all images (note though 
that we will not reproject the donor image):
```{code-cell} python
donor_wcs_hdr = mission_hdus[finest_pix_miss].header.copy()
```

```{code-cell} python
reproj_data_cov = {mn: {"data": (rp := reproject_to_common_grid((cur_hdu.data, cur_hdu.header), donor_wcs_hdr))[0], "cov": ((rp[1] > 0).sum() / rp[1].size)} for mn, cur_hdu in mission_hdus.items() if cur_hdu is not None and mn != finest_pix_miss}

for cur_miss, rp_info in reproj_data_cov.items():
    if rp_info['data'] is not None:
        print(f"{cur_miss}: {rp_info['data'].shape}, coverage = {rp_info['cov'] * 100:.1f}%")
        
        # While we're here, we also remove the original data for the current 
        #  mission from memory, as we have the reprojected data now.
        del mission_hdus[cur_miss].data
        
reproj_data_cov[finest_pix_miss] = {'data': mission_hdus[finest_pix_miss].data, 'cov': 1.0}
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
sep_ims = InteractiveMultiPanel({mn: res['data'] for mn, res in reproj_data_cov.items() if res['cov'] > 0.1}, sep_reproj_im_cmaps)
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

We note that NGC 4753 has poor Chandra coverage, so for that source we use the UV 
observation from Swift as the blue channel instead.

```{code-cell} python

red_chan = reproj_data_cov['Spitzer']['data']
green_chan = reproj_data_cov['Hubble']['data']
if SOURCE_NAME == 'NGC 4753':
    blue_chan = reproj_data_cov['Swift']['data']
else:
    blue_chan = reproj_data_cov['Chandra']['data']

multi_wav_im = InteractiveRGBPanel(red_data=red_chan,
                                   green_data=green_chan,
                                   blue_data=blue_chan)
multi_wav_im.view()
```

+++

## About this Notebook

**Authors:** David J Turner

**Contact:** For help with this notebook, please open a topic in the [Fornax Helpdesk](https://discourse.fornax.sciencecloud.nasa.gov/c/helpdesk).

+++

### Acknowledgements

This notebook was created with assistance from:
- The Fornax team.
- The High Energy Astrophysics Science Archive Research Center (HEASARC) team.
- The NASA/IPAC Infrared Science Archive (IRSA) team.
- The Barbara A. Mikulski Archive for Space Telescopes (MAST) team.

AI: This notebook was created with assistance from Google's Gemini models and NASA ChatGSFC.

### References

This work made use of:
- [AstroPy: A Community Python Library for Astronomy](https://www.astropy.org/)
- [AstroQuery: Access to Online Astronomical Data](https://astroquery.readthedocs.io/)
- [Reproject: Image Reprojection in Python](https://reproject.readthedocs.io/)
- [Lupton et al. 2004, PASP, 116, 133: "Preparing Red-Green-Blue Images from CCD Data"](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L)
