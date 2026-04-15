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

# Create interactively multi-wavelength images of astronomical sources

## Learning Goals

***NEED MORE ENTRIES***

By the end of this tutorial, you will be able to:
- Reproject images from different missions to a common coordinate grid.
- Create interactive visualizations of individual reprojected images.

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

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity

from astropy.visualization import ImageNormalize, PercentileInterval, LogStretch
import numpy as np

# Add local code directory to path
sys.path.append('code_src/')

# Import our custom functions
from archive_queries import (
    query_chandra, download_chandra,
    query_spitzer, get_spitzer_s3_path,
    query_hst, download_hst,
    query_swift, download_swift
)
from image_processing import (
    get_pixel_scale, reproject_to_common_grid,
    load_fits_from_s3
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

+++

## 2. Query archives for available data

We now query each archive to find observations covering our target.
Each archive has different data organization and search interfaces.

### 2.1 Query HEASARC for X-ray data

Chandra provides the highest resolution X-ray images available.
We search for ACIS-S observations without a grating, as these produce the best imaging data.

```{code-cell} ipython3
print("Querying HEASARC for Chandra observations...")

chandra_obs = query_chandra(SOURCE_COORD, detector="ACIS-S", grating="NONE")

if chandra_obs is not None and len(chandra_obs) > 0:
    print(f"Found {len(chandra_obs)} Chandra observations")
    print(f"Longest exposure: {chandra_obs['exposure'][0]:.0f} seconds")

    # Select a specific high-quality observation for Crab
    # For other targets, you may want to use the longest exposure (first row)
    if SOURCE_NAME.lower() == "crab":
        selected_chandra = chandra_obs[chandra_obs['obsid'] == 1994]
        print(f"Selected ObsID 1994 (well-known deep observation)")
    else:
        selected_chandra = chandra_obs[0]
        print(f"Selected longest exposure (ObsID {selected_chandra['obsid'][0]})")
else:
    print("No Chandra data found for this target")
    selected_chandra = None
```

### 2.2 Query IRSA for infrared data

Spitzer's IRAC instrument provides infrared imaging.
We search for processed mosaics at the 3.6 micron band, which typically has good sensitivity.

```{code-cell} ipython3
print("Querying IRSA for Spitzer observations...")

spitzer_obs = query_spitzer(SOURCE_COORD, radius_arcmin=3.0)

if spitzer_obs is not None:
    print("Found Spitzer IRAC mosaic")
    print(f"Distance from target: {spitzer_obs['dist_to_point']:.4f} degrees")
else:
    print("No Spitzer data found for this target")
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

```{code-cell} ipython3
if selected_chandra is not None:
    print("Downloading Chandra data...")
    chandra_img_path = download_chandra(selected_chandra, CHAN_DATA_DIR, obsid=1994)

    if chandra_img_path and os.path.exists(chandra_img_path):
        print(f"Chandra data downloaded to: {chandra_img_path}")
        chandra_hdu = fits.open(chandra_img_path)
        print(f"Image shape: {chandra_hdu[0].data.shape}")
    else:
        print("Failed to download or locate Chandra data")
        chandra_hdu = None
else:
    print("Skipping Chandra download (no data available)")
    chandra_hdu = None
```

### 3.2 Load Spitzer data from the IRSA S3 bucket

Spitzer data can be accessed directly from AWS S3 without downloading.
This is much faster and more efficient than traditional file downloads.

```{code-cell} ipython3
if spitzer_obs is not None:
    print("Loading Spitzer data from S3...")
    spitzer_s3_path = get_spitzer_s3_path(spitzer_obs)

    if spitzer_s3_path:
        print(f"S3 path: {spitzer_s3_path}")
        spitzer_hdu = load_fits_from_s3(spitzer_s3_path)
        print(f"Image shape: {spitzer_hdu[0].data.shape}")
    else:
        print("Failed to get Spitzer S3 path")
        spitzer_hdu = None
else:
    print("Skipping Spitzer load (no data available)")
    spitzer_hdu = None
```

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
print("Comparing pixel scales:")
print("-" * 50)

# Dictionary mapping missions to their relevant HDUs/extensions
mission_hdus = {
    'Chandra': chandra_hdu[0] if chandra_hdu else None,
    'HST': hst_hdu['SCI'] if hst_hdu else None,
    'Swift': swift_hdu[1] if swift_hdu else None,
    'Spitzer': spitzer_hdu[0] if spitzer_hdu else None
}

pixel_scales = {}
for mission, hdu in mission_hdus.items():
    if hdu is not None:
        scale = get_pixel_scale(hdu)
        pixel_scales[mission] = scale
        print(f"{mission:<10}: {scale.to('arcsec/pix'):.3f}")

print("-" * 50)

# Hubble is typically much higher resolution than others
if 'HST' in pixel_scales:
    print(f"\nHubble has the highest resolution")
    if 'Chandra' in pixel_scales:
        ratio = (pixel_scales['Chandra'] / pixel_scales['HST']).decompose()
        print(f"Chandra pixels are ~{ratio:.1f}x larger than HST pixels")
    print("We will use HST resolution as the common grid")
```

+++

### 4.2 Choosing the coordinate grid for reprojection

We select the highest resolution image's coordinate system as our target grid.

```{code-cell} ipython3
# Find highest resolution image
print("Selecting common WCS grid...")

# Identify which image has the finest pixel scale
finest_mission = min(pixel_scales, key=pixel_scales.get)
print(f"Using {finest_mission} as reference (highest resolution)")

# Get the reference header
if finest_mission == 'Chandra':
    donor_wcs_hdr = chandra_hdu[0].header.copy()
elif finest_mission == 'HST':
    donor_wcs_hdr = hst_hdu['SCI'].header.copy()
elif finest_mission == 'Swift':
    donor_wcs_hdr = swift_hdu[1].header.copy()
elif finest_mission == 'Spitzer':
    donor_wcs_hdr = spitzer_hdu[0].header.copy()

print(f"Common grid size: {donor_wcs_hdr['NAXIS1']} x {donor_wcs_hdr['NAXIS2']} pixels")

# Calculate pixel scale of final grid
final_pixel_scale = get_pixel_scale(donor_wcs_hdr)
print(f"Final pixel scale: {final_pixel_scale.to('arcsec/pix'):.3f}")
```

### 4.3 Reproject all images

This step performs the coordinate transformation for each image.
It may take a minute or two per image depending on size and complexity.

```{code-cell} ipython3
# Prepare images dictionary
images_dict = {}

if chandra_hdu is not None:
    images_dict['Chandra'] = chandra_hdu[0]

if hst_hdu is not None:
    images_dict['HST'] = hst_hdu['SCI']

if swift_hdu is not None:
    images_dict['Swift'] = swift_hdu[1]

if spitzer_hdu is not None:
    # Special handling for S3-loaded files
    images_dict['Spitzer'] = (spitzer_hdu[0].data, spitzer_hdu[0].header)

# Reproject to common grid
print("Reprojecting images to common grid...")
print("(This may take a few minutes)")
print()

reprojected_images = reproject_to_common_grid(images_dict, donor_wcs_hdr)

print()
print("Reprojection complete!")
print()

# Extract reprojected data arrays
reprojected_data = {}
for name, result in reprojected_images.items():
    if result is not None:
        data, footprint = result
        reprojected_data[name] = data
        print(f"{name}: {data.shape}, coverage = {(footprint > 0).sum() / footprint.size * 100:.1f}%")
```

+++

## 5. Visualize individual wavelength images

Now we create an interactive four-panel plot showing each wavelength band separately.
This allows us to see what each telescope captured and understand the contribution of each band to the final composite.

The plot includes linked zooming across all four panels.
Select a region in any panel to zoom all panels to that area.

```{code-cell} ipython3
sep_reproj_im_cmaps = {
    'HST': 'Greens',
    'Swift': 'Blues',
    'Chandra': 'Purples',
    'Spitzer': 'YlOrBr'
}
```

```{code-cell} ipython3
thing_ui = InteractiveMultiPanel(reprojected_data, sep_reproj_im_cmaps)
thing_ui.view()
```

Each panel shows the same region at a different wavelength.
The interactive plot allows you to zoom and pan simultaneously across all panels.
Different wavelengths reveal different physical processes and components of the source.

+++

## 6. Interactive multi-wavelength image

Finally, we combine three wavelength bands into a single RGB composite image.
We use:
- Red channel: Spitzer infrared (cool dust and molecular gas)
- Green channel: Hubble optical (stellar emission and ionized gas)
- Blue channel: Chandra X-ray (extremely hot plasma and high-energy particles)

This composite uses the Lupton algorithm, which is designed specifically for astronomical images with high dynamic range.

The interactive controls allow you to adjust:
- **Channel percentiles**: Control the brightness scaling for each color
- **Lupton Q**: Softening parameter (higher values show more faint detail)
- **Lupton stretch**: Overall brightness (higher values make the image brighter)

Experiment with the sliders to highlight different physical components and create striking visualizations.

```{code-cell} ipython3
# Extract the three channels
red_channel = reprojected_data['Spitzer']
green_channel = reprojected_data['HST']
blue_channel = reprojected_data['Chandra']

ui = InteractiveRGBPanel(red_channel, green_channel, blue_channel)
ui.view()
```

By adjusting the controls, you can create different visualizations that emphasize various aspects of the source's physics and structure.

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
