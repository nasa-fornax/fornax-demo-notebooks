---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Beyond metadata: analytical data search in the cloud with JWST spectral cubes

+++
## Runtime
For this notebook, we recommend using the [Fornax science platform](https://nasa-fornax.github.io/fornax-demo-notebooks/documentation/README.html) with Server Type = "Large - 64GB RAM/16 CPU" and Environment = "Default Astrophysics". As of June 2025, running this notebook once on that server will take about 10 minutes.

We assume that you are running this notebook in Fornax, but it will also still work (in order of preference) in another AWS-based cloud science platform, a non-AWS cloud science platform, and even on your local computer — perhaps at slower speeds, depending on your computing resources and internet connection.

+++

## Learning goals

The overarching intent of this notebook is to learn about how to answer data-intensive archival search queries in astronomical data archives like [MAST](https://archive.stsci.edu/). We'll have to think about:
* when to parallelize our data processing,
* the order of operations for performing cone searches on large numbers of targets,
* when to load data into memory versus downloading to storage,
* the advantages and constraints of a cloud science platform,
* and when to load data from cloud storage instead of on-premise storage.

## Introduction

As you read through a recent paper by [Assani et al. 2025](https://arxiv.org/pdf/2504.02136), imagine that you find yourself consumed by a single thought:

> I want to find more JWST spectral image cubes containing Fe II emission from jets launched by young stellar objects (YSOs).

You may now be experiencing some despair. "This observation contains a JWST spectral cube" is certainly something readily available from MAST's metadata tables, but "this file contains Fe II transitions" is certainly not. So instead, you need to dig beyond the metadata and into the data of thousands of files. That may sound difficult, but it can be done!

## Table of Contents

More granularly, this notebook is comprised of the following sections:

- [Imports](#Imports)
- [1. Finding JWST spectral cubes of YSOs](#1.-Finding-JWST-spectral-cubes-of-YSOs)
    - [1.1 Querying SIMBAD for YSOs](#1.1-Querying-SIMBAD-for-YSOs): *Use `astroquery` to ask SIMBAD for all the YSOs it knows about.*
    - [1.2 Querying MAST for all JWST spectral cube observations](#1.2-Querying-MAST-for-all-JWST-spectral-cube-observations): *Use `astroquery` to ask MAST for all JWST spectral cube observations.*
    - [1.3 Crossmatching YSO coordinates to MAST footprints](#1.3-Crossmatching-YSO-coordinates-to-MAST-footprints): *Write custom functions to crossmatch the SIMBAD YSOs to the footprints of the MAST observations, bypassing an astroquery service call that would take to long for this number of targets.*
    - [1.4 Refining and parallelizing the footprint crossmatch](#1.4-Parallelizing-the-footprint-crossmatch-with-dask): *Improve and parallelize the crossmatch technique, with gnomonic projection and dask.*
- [2. Finding Fe II emission jets](#2.-Finding-Fe-II-emission-jets)
    - [2.1 Retrieving sample data from AWS](#2.1-Retrieving-sample-data-from-AWS): *Retrieve MAST data from the AWS cloud into memory.*
    - [2.2 Searching for emission lines in spectral cubes](#2.2-Searching-for-emission-lines-in-spectral-cubes): *Write custom functions to algorithmically analyze thousands of spectral cube data files, and assess which of these are likely to exhibit extended Fe II emission.*
    - [2.3 Parallelizing the search for emission lines in spectral cubes](#2.3-Parallelizing-the-search-for-emission-lines-in-spectral-cubes): *Parallelize the spectral line search and apply it to thousands of data files.*
- [3. Postscript: Transposing to IRSA missions](#3.-Postscript:-Transposing-to-IRSA-missions): *Exercise for transposing these techniques to SOFIA data.*
- [About this notebook](#About-this-notebook)

+++

## Imports

+++

<span style="color: purple;">Temporary note: pending incorporation of a recent `astroquery` development version into the default Fornax kernel, we need to install the latest development version of astroquery. We can suppress the voluminous output here with `%%capture`, which will store the cell's output in a variable `captured` (accessed through `captured.stdout` and `captured.stderr`). Uncomment and run this cell:</span>

```{code-cell} ipython3
# %%capture captured
# %pip install git+https://github.com/astropy/astroquery.git
```

**You may need to restart your kernel now for that to take effect.**

+++

Otherwise, all dependencies are included in the Fornax "Default Astrophysics" environment. If you are not working in Fornax, you can uncomment and run the following cell to install these dependencies into your environment. You may need to restart your kernel after pip-installing new packages, before running the imports.

```{code-cell} ipython3
# !pip install -r requirements_analytical_data_search.txt
```

```{code-cell} ipython3
# Query databases
from astroquery.simbad import Simbad
from astroquery.mast import Observations
from astroquery.ipac.irsa import Irsa

# Crossmatch coordinates to footprint polygons
from matplotlib.path import Path
from astropy.wcs import WCS

# Create plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm

# Wrangle data 
from astropy.table import Table, Column, Row
from astropy.io import fits
import pandas as pd
import numpy as np

# Wrangle units
from astropy.coordinates import SkyCoord
import astropy.units as u

# Detect extended emission in specific wavebins
from scipy.signal import find_peaks
from scipy.ndimage import label
import bisect

# Utility functions
import os
import sys
import copy

# Parallelize
import dask
import dask.dataframe as dd
```

We'll also need a couple local functions written specifically for this notebook. These are located in the `code_src` directory, and you can read them there. However, they're pretty long, and their detailed implementation isn't critical to understand. As long as you understand their inputs and outputs, which we'll discuss later, you can write better functions to replace them, tailored to your own science case.

```{code-cell} ipython3
# Local code imports
sys.path.append('code_src/')

from extension_vs_wavebin import extension_vs_wavebin
from detect_spikes import detect_spikes
```

Let's also turn on `enable_cloud_dataset` for `astroquery.mast.Observations`. This will allow us to fetch the cloud locations for data products and access data directly from the cloud, instead of retrieving data from MAST's on-premise servers.

```{code-cell} ipython3
Observations.enable_cloud_dataset()
```

Later, we'll be evaluating boolean conditionals on values that could be `None`, the pandas equivalent `pandas.NA`, or a normal value, and we'll also need this little function. It's pretty uninteresting, so let's get it out of the way now!

```{code-cell} ipython3
# Check that x is not None and is not pd.NA
def exists(x):
    try:
        return x is not None and not pd.isna(x)
    except TypeError:
        # For types that can't be checked with pd.isna
        return x is not None
```

## 1. Finding JWST spectral cubes of YSOs

+++

Let's ignore the emission line issue and tackle the easiest parts of our driving question first: I want to find more JWST spectral cubes of young stellar objects.

+++

### 1.1 Querying SIMBAD for YSOs

+++

At this time, MAST doesn't support reliable object classification search. So let's go to the experts in astronomical object cataloging, SIMBAD, which we can access through `astroquery.simbad.Simbad.query_tap`.

In the cell below, we search for all SIMBAD-catalogued objects labeled as YSOs (`otype='Y*O'`) or as any of the descendant sub-concepts of YSOs, like T Tauri stars (`otype='Y*O..'` to retrieve both explicitly labeled YSOs and their subtypes).

```{code-cell} ipython3
# This will take a minute or two.
yso_table = Simbad.query_tap("SELECT * FROM basic WHERE otype='Y*O..'", maxrec=1000000)
```

```{code-cell} ipython3
print(f'We found {len(yso_table)} YSOs.')
```

Let's make a list of their sky coordinates:

```{code-cell} ipython3
yso_coords = []  # List of tuples, each tuple a coordinate.

for row in yso_table:
    # Check if the coordinates exist
    if isinstance(row['ra'], float) and isinstance(row['dec'], float):
        # Append the coordinates to the list
        yso_coords.append((row['ra'], row['dec']))
    
print(f'{len(yso_coords)} of these YSOs have good coordinates.')
```

### 1.2 Querying MAST for all JWST spectral cube observations

+++

Now we want to get all the JWST spectral cube observations in MAST whose sky footprints overlap with any of these YSO coordinates. The natural tool is the `astroquery.mast.Observations` class, which gives programmatic access to MAST's multi-mission archive.

However, `astroquery.mast` is not currently set up to quickly run a multi-target query for tens of thousands of sky coordinates. So in order to get things done in a reasonable amount of time, we need to think about the order of operations here.

It turns out that, at the time of writing, there are only a few tens of thousands of JWST spectral cubes across the whole sky. So let's retrieve *all* of those first, excluding only those with a `calib_level` of `-1` (planned observations that haven't yet executed) and non-public datasets.

```{code-cell} ipython3
# This will typically take anywhere from a few seconds to a minute or so,
# depending on how many people are using MAST right now.
# Rarely, if MAST is overloaded, it may take several minutes or time out.

jwst_obstable = Observations.query_criteria(dataproduct_type='cube',
                                                 obs_collection='JWST',
                                                 dataRights='PUBLIC',  # Limit to public data
                                                 calib_level = [0, 1, 2, 3, 4])  # Exclude planned observations
```

```{code-cell} ipython3
print(f'We found {len(jwst_obstable)} JWST spectral cube observations.')
```

`Astroquery` has retrieved a wide variety of metadata for each of these observations. Let's look at the first couple rows:

```{code-cell} ipython3
jwst_obstable[0:2]
```

### 1.3 Crossmatching YSO coordinates to MAST footprints

+++

Now, with that table as our starting point, let's match our YSO coordinates to the sky footprints of all these JWST observations. We already have everything from MAST that we need to do this, so we don't need to run another query. The footprint of an observation is in the `s_region` column of the observations table output by `astroquery`:

```{code-cell} ipython3
# For example...
print(jwst_obstable[0]['s_region'])
```

This `s_region` denotes the vertices of a polygon in right ascension and declination. We can make this string more parsable with this function:

```{code-cell} ipython3
def parse_polygon(s_region):
    """
    Extract coordinates from an s_region string.

    Input parameters
    ----------
    s_region : string - s_region from cross-mission archive
               As written, this s_region must be a single polygon,
               which is the case for all JWST and SOFIA observations,
               but not all missions in the Fornax archives.

    Returns
    ----------
    coord_array : numpy array - parsed representation of the s_region
    """

    if 'POLYGON ICRS' in s_region:
        coords = list(map(float, s_region.replace("POLYGON ICRS", "").strip().split()))
    elif 'POLYGON' in s_region:
        coords = list(map(float, s_region.replace("POLYGON", "").strip().split()))

    # Check if the polygon is closed, with the last vertex identical to the first vertex.
    # If not, append a copy of the first vertex to the end of the list.
    if (coords[-2], coords[-1]) != (coords[0], coords[1]):
        coords.append(coords[0])
        coords.append(coords[1])

    # Create a numpy array listing a coordinate tuple for each vertex
    coord_array = np.array([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])

    return(coord_array)
```

```{code-cell} ipython3
# For example...
print(parse_polygon(jwst_obstable[0]['s_region']))
```

Now, we'll parse our polygons and use `Matplotlib`'s `Path` class to test whether any of our YSO coordinates are in each observation's sky footprint.

Doing this without parallelizing, as in the following cell, would take about 6 or 7 minutes. On Fornax, if you uncomment this cell and open the "Kernel usage" tab to your right while it runs, you'll notice that you're using only a small percentage of the total CPU capacity that you have access to (e.g., less than 10% if you selected the 16-CPU server option when you launched Fornax). Feel free to click the `Interrupt the kernel` stop button on your notebook, because there's a better way!

```{code-cell} ipython3
# Since this cell is an example of what not to do,
# we'll also omit the nuance of gnomonic projection: see the next section.

# yso_jwst_obs = []  # Instantiate a list of observations whose footprints overlap with any of our YSO coordinates.

# for row in jwst_obstable:  # Looping through all JWST spectral cube observations...
#     polygon = parse_polygon(row['s_region'])  # Parse the polygon

#     polygon_path = Path(polygon, closed=True)  # Convert to matplotlib Path
#     is_inside = polygon_path.contains_points(yso_coords).any()  # Test whether any coordinate from yso_coord is inside this observation's polygon

#     if is_inside:
#         yso_jwst_obs.append(row)

# yso_jwst_obstable = Table(rows=yso_jwst_obs, names=jwst_obstable.colnames)  # Convert back to an astropy table.

# print(f'We found {len(yso_jwst_obstable)} JWST spectral cube observations whose footprints overlap with YSO coordinates.')
```

### 1.4 Refining and parallelizing the footprint crossmatch

+++

That cell would take a long time to run, so let's parallelize this code to save time! First we'll need to define a function that does what we did above, for each observation, so that we can pass this function into our parallelization tool.

While we're at it, we'll also add in a subtle nuance; *RA* and *dec* are spherical coordinates, so the segments of a footprint polygon should be great circles on the celestial sphere, not straight lines. However, they can be treated as straight lines if we project into a 2D cartesian coordinate system, to an excellent approximation in the vicinity of the center of the projection. An appropriate choice is the gnomonic (`TAN`) projection, which is the projection used in the JWST FITS images from which `s_region` values were derived. Before checking which YSO coordinates are inside a particular observation's footprint, we'll convert both the footprint's polygon vertices and the YSO coordinates to a gnomonic projection centered on that specific observation's coordinates.

We mentioned that the projection issue is a subtle nuance; in fact, at the time of writing, you could omit the projection step and still get the same number of observations matched. YSO coordinates tend not to be at the very edges of the observation footprints, after all. Nevertheless, to avoid any future problems, let's do things properly! On the Fornax server recommended above, this will only add about 30 seconds to our parallelized processing.

Note that we are not accounting for proper motions—and neither would `astroquery.mast` cone search, if we were using it. See [High PM Stars in MAST](https://outerspace.stsci.edu/display/MASTDOCS/High+PM+Stars+in+MAST) if this is a concern for your science case.

```{code-cell} ipython3
def check_points_in_polygon(row, coordinates_to_test):
    """
    Check if coordinates_to_test are in the s_region polygon for an observation.

    Input parameters
    ----------
    row : a row of an astropy table of observations from astroquery
    coordinates_to_test: python list of celestial coordinate tuples

    Returns
    ----------
    Either `row` (same as input) if a match was found,
    or an empty row of NaNs if a match was not found.
    """

    # Make WCS for a gnomonic (TAN) projection
    # with a projection center at the observation's coordinates
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0., 0.]  # Put center of the projection at projected pixel coords (0, 0)
    wcs.wcs.cdelt = [1., 1.]  # Arbitrary 1 degree per pixel (doesn't matter)
    wcs.wcs.crval = [row['s_ra'], row['s_dec']]  # Projection center is the observation's coordinates
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Gnomonic projection

    # Convert coordinates_to_test to the gnomonic projection
    proj_coordinates_to_test = wcs.wcs_world2pix(coordinates_to_test, 0)

    # Make footprint polygon path for this observation
    polygon = parse_polygon(row['s_region'])  # Parse s_region value
    proj_polygon = wcs.wcs_world2pix(polygon, 0)  # Convert to gnomonic projection
    proj_polygon_path = Path(proj_polygon, closed=True) # Convert to matplotlib Path

    # Test whether any coordinate from yso_coord is inside this observation's polygon_path.
    # Return the table row if the match is successful,
    # otherwise return an empty row full of NaNs.
    if proj_polygon_path.contains_points(proj_coordinates_to_test).any():
        return row
    else:
        return pd.Series([np.nan] * len(row), index=row.index)    
```

`dask` is a python module for parallelizing your Python code. Let's set it to use multiple CPU cores to make full use of the computational resources we have in Fornax:

```{code-cell} ipython3
dask.config.set(scheduler='processes')
```

For this task, we've found that it's best to partition the Observations table into about 4 times as many partitions as there are CPU cores that you have access to. With 4 cores, the next cell will take about 4 or 5 minutes, a slight improvement over the non-parallel approach. With 16 cores, it'll typically take less than 2 minutes.

```{code-cell} ipython3
num_cores = os.cpu_count()
print(f'We have access to {num_cores} cores.')
```

```{code-cell} ipython3
# Convert jwst_table into a partitioned Dask dataframe,
# so that the cores can split the partitions amongst themselves.
jwst_obsddf = dd.from_pandas(jwst_obstable.to_pandas(), npartitions=num_cores*4)  # Adjust npartitions to taste

# Set up to apply the check_point_in_polygon function in parallel.
meta = pd.DataFrame(columns=jwst_obstable.to_pandas().columns)
jwst_job = jwst_obsddf.map_partitions(lambda df: df.apply(lambda row: check_points_in_polygon(row, yso_coords), axis=1), meta=meta)
```

If you examine the `Kernel usage` tab to your right while running this next cell, you'll see that you're now using about 70% of your multi-CPU capacity.

```{code-cell} ipython3
# Execute
results = jwst_job.compute()
```

```{code-cell} ipython3
# Remove empty rows, where check_points_in_polygon reached the "else" part of the return statement.
results = results.dropna(how='all')

# Convert back to an astropy table.
yso_jwst_obstable = Table.from_pandas(results)

print(f'We found {len(yso_jwst_obstable)} JWST spectral cube observations whose footprints overlap with YSO coordinates.')
```

## 2. Finding Fe II emission jets

+++

We're looking for a way to identify jets exhibiting Fe II emission, without human intervention and without knowing much about the data quality or target source in each image. Let's first devise our approach by looking at a case example, and then we'll try it out on the full set of YSO JWST spectral cubes.

+++

### 2.1 Retrieving sample data from AWS

+++

From [Assani et al. 2025](https://arxiv.org/pdf/2504.02136), we know that the YSO called TMC1A features an extended jet with strong Fe II emission. Let's grab its coordinates from SIMBAD...

```{code-cell} ipython3
tmc1a_meta = Simbad.query_object('TMC1A')
tmc1a_ra = float(tmc1a_meta['ra'])  # ICRS RA of TMC1A in degrees
tmc1a_dec = float(tmc1a_meta['dec'])  # ICRS Dec of TMC1A in degrees
```

...and retrieve its observations from our table of JWST spectral cube observations of YSOs:

```{code-cell} ipython3
tmc1a_jwst = []

# As before...
for row in yso_jwst_obstable:
    polygon = parse_polygon(row['s_region'])
    polygon_path = Path(polygon, closed=True)
    # We'll use matplotlib.Path.contains_point (singular) this time,
    # instead of .contains_points (plural)
    is_inside = polygon_path.contains_point((tmc1a_ra, tmc1a_dec))
    if is_inside:
        tmc1a_jwst.append(row)

# Convert to an astropy Table
tmc1a_jwst_obstable = Table(rows=tmc1a_jwst, names=jwst_obstable.colnames)

print(f'We found {len(tmc1a_jwst_obstable)} JWST spectral cube observations of TMC1A.')
```

Let's get the AWS cloud URIs for these spectral cube files. Downloading files from AWS (via a cloud URI) isn't *always faster* than downloading from MAST's on-premise servers (via a `dataURI` converted into an HTTP URL), even when the destination for your downloads is an AWS cloud platform like Fornax, but importantly AWS is more *robust* for large data downloads. By retrieving files from AWS instead of MAST, you avoid the risk of overwhelming MAST's on-premise servers and causing problems for other users, and the substantial risk that MAST's on-premise servers may be overwhelmed by requests from other users in the moment that you want to get your data. AWS is, we believe, less likely to be overwhelmed.

Regardless, the volume of available astronomical data is growing faster than the MAST archive's capacity to store on-premise and transmit through HTTP links to the on-premise files. For example, the daily downlinked data volume from the Roman Space Telescope will be about 500 times that of HST. So eventually, for some data, cloud downloads are expected to be the only option.

Following usual MAST practice, we'd ordinarily use `astroquery.mast.Observations` to get all individual data products associated with our tabulated observations...

```{code-cell} ipython3
# Get products contained in these observations
tmc1a_jwst_products = Observations.get_product_list(tmc1a_jwst_obstable)

# Filter to select only the high-level spectral cube products ("S3D").
tmc1a_jwst_products_filtered = Observations.filter_products(tmc1a_jwst_products,
                                                                      mrp_only=True,
                                                                      productSubGroupDescription='S3D')

# Let's look at the first couple rows:

tmc1a_jwst_products_filtered[0:2]
```

...and we could feed this table into `astroquery.Observations.get_cloud_uris` to get the corresponding AWS URIs. However, later we're going to have to do this for *thousands* of JWST observations, each of which can have a remarkable number of underlying products. In that case, `astroquery.mast.Observations.get_product_list` would take quite a long time to run.

Luckily, the high-level "s3d" spectral cube products we're interested in are the primary data product of each observation, which means that the MAST `dataURI`s for these files are tabulated in the `dataURL` column of the observations table:

```{code-cell} ipython3
# For example:
print(tmc1a_jwst_obstable[0]['dataURL'])
```

So we can save ourselves quite a lot of time by feeding a python list of these `dataURL`s directly into `astroquery.Observations.get_cloud_uris`, which will convert each `dataURL` into a cloud URI.  By setting `return_uri_map=True`, we'll also ensure that we know which observation each cloud URI corresponds to. The output `cloud_uri_map` is a python dictionary mapping `dataURL` values to cloud URIs.

We'll also set `verbose=False` to suppress warnings about data products that are temporarily unavailable in the cloud; be sure to turn `verbose` back on if your science case requires complete information, or add code later on to handle the situation where a cloud URI is not available.

```{code-cell} ipython3
cloud_uri_map = Observations.get_cloud_uris(list(tmc1a_jwst_obstable['dataURL']), return_uri_map=True, verbose=False)
```

We could just look at the first observation in the table, but for demonstration purposes and to ensure that the images we plot in this tutorial are predictable, let's retrieve the observation corresponding to the first row in this table meeting the following criteria: MIRI instrument, CH1-SHORT dispersion element, proposal ID 1290, and has a valid cloud URI.

```{code-cell} ipython3
# Retrieve all observations for TMC1A with MIRI CH1-SHORT and proposal ID 1290
temp_observations = tmc1a_jwst_obstable[
    (tmc1a_jwst_obstable['instrument_name']=='MIRI/IFU') &
    (tmc1a_jwst_obstable['filters']=='CH1-SHORT') &
    (tmc1a_jwst_obstable['proposal_id']=='1290')
]

# Get the first valid cloud URI found
for row in temp_observations:
    cloud_uri = cloud_uri_map[row['dataURL']]
    if cloud_uri:
        break  # When a valid URI is found, exit the loop

print('Cloud URI: ', cloud_uri)
```

MAST cloud data is in the [Registry of Open Data on AWS](https://registry.opendata.aws/), so it's configured to allow non-credentialed anonymous access. We can load this file into `astropy.fits` by passing in the cloud URI in as if it were a normal URL, but we need to tell `astropy.fits` to access the file anonymously. More information is available [in the astropy documentation](https://docs.astropy.org/en/stable/io/fits/usage/cloud.html).

```{code-cell} ipython3
with fits.open(cloud_uri, cache=False, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
    header = hdul['SCI'].header
    data = hdul['SCI'].data

# Note: we didn't actually need to manually set use_fsspec=True.
# This is turned on by default when using an AWS URI.

# Note: instead of using fsspec inside an astropy.fits.open() to handle the cloud connection,
# you could equivalently use the s3fs module explicitly, as follows.
# Results are typically similar.

# import s3fs
# fs = s3fs.S3FileSystem(anon=True)
# with fs.open(cloud_uri, 'rb') as cloud_file:
#     with fits.open(cloud_file, 'readonly') as hdul:
#         header = hdul['SCI'].header
#         data = hdul['SCI'].data
```

Let's turn this into a convenience function that we can use later on:

```{code-cell} ipython3
def load_cloud(cloud_uri, extension='SCI'):
    """
    Load from a cloud URI into memory, without downloading to storage.

    Input parameters
    ----------
    cloud_uri : string
        An AWS cloud URI for a FITS file.
    extension : string
        The FITS extension in which the desired data is found. Defaults to 'SCI'.

    Returns
    ----------
    header : astropy.io.fits.header.Header
    data : numpy.ndarray
    """
    with fits.open(cloud_uri, cache=False, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        header = hdul[extension].header
        data = hdul[extension].data
    return header, data
```

Remember that you have a limited storage allocation in your Fornax account: 10 GB at the time of writing. In our case, when we analyze the full set of YSOs, we expect to need at least about 100 GB worth of data net across many small ~10 MB files. Thus, it is important that we are using `cache=False` and `use_fsspec=True` (which are the default settings when passing a cloud URI into `astropy.fits`) to load the data into memory (RAM) rather than into storage. This can similarly be enforced for downloads from on-premise HTTP links, if cloud URIs are not available, by explicitly passing both `cache=False` and `use_fsspec=True` into the `open` function.

+++

### 2.2 Searching for emission lines in spectral cubes

+++

Let's copy down some of the Fe II wavelengths from [Assani et al. 2025](https://arxiv.org/pdf/2504.02136):

```{code-cell} ipython3
transitions = [25.988,24.519,17.936,5.340,1.644,1.600,1.257]  # in microns
```

For the purposes of demonstration, we'll use five functions that together look at a JWST spectral cube file, and aim to characterize whether or not that spectral cube exhibits spatially-extended Fe II emission at each of these transition wavelengths.

Our first function, `extension_vs_wavebin`, measures the number of pixels above some flux threshold in each wavebin of a cube. We imported it earlier from the local `code_src` directory for this notebook, and you can take a look at it there.

That said, you don't need to read through the whole function. You can and almost certainly should replace it with [your personal favorite image segmentation and/or machine learning algorithm](https://ui.adsabs.harvard.edu/abs/2024A%26C....4800838X/abstract).

The important thing to know is that `extension_vs_wavebin()` takes as input a spectral cube as a `numpy.ndarray` (with first axis wavelength), and outputs a 1-dimensional python list of length equal to the number of wavebins, in which each list element corresponds to the size of bright emission or some parameterization of the probability that an extended jet-like feature is seen in that wavebin. If you write your own function that does all that, then it should be possible to integrate into this workflow.

+++

We can test `extension_vs_wavebin` out using the data cube that we've already downloaded into memory:

```{code-cell} ipython3
test_blobs = extension_vs_wavebin(data)
```

```{code-cell} ipython3
plt.plot(test_blobs, color='k')
plt.xlabel('Wavelength bin index')
plt.ylabel('Fraction of non-NaN pixels above threshold')
plt.show()
```

> Figure 1 image description: a line plot of "Fraction of non-NaN pixels above threshold" versus "Wavelength bin index", the latter ranging from around 0 to 1000. The plotted values have substantial scatter and features, but one feature stands out by a factor of two above all others: a narrow, abrupt spike near wavelength bin index 548.

+++

We see a variety of messy features, perhaps corresponding to noise or to the brightness of the central star, but we also see a dramatic single-bin spike at around the middle wavebin of the cube. Let's take a look at that slice of the cube, where we see an extended jet emerging from a YSO point source. This is the basis for one of the Fe II line images from Fig. 1 of [Assani et al. 2025](https://arxiv.org/pdf/2504.02136).

```{code-cell} ipython3
plt.imshow(data[548], norm='log')
plt.xlabel('y')
plt.ylabel('x')
plt.show()
```

> Figure 2 image description: 2-dimensional astronomical image in pixel coordinates, with rough boundaries of null values near the edges. From a bright circular source near the top edge, a faint hint of jet emission extends downwards and slightly to the right, nearly all the way to the lower edge of the image.

+++

Our next function, `detect_spikes`, was likewise imported earlier from the local `code_src` directory for this notebook, and you can look at it there. It takes as input the 1D list returned by `extension_vs_wavebin`, looks for peaks of extended emission size/probability in the vicinity of a specific wavebin, and returns `True` if and only if a peak is found within 2 wavebins of the input wavebin.

Here again, you can and should substitute your own more sophisticated algorithm. The important things to know are that `detect_spikes` takes the following as input...
- the list returned by `extension_vs_wavebin`,
- an integer wavebin index `pos`,
- an optional float scaling factor `sig` used to set the detection threshold,
- an optional integer `detect_halfwidth` to set the range of wavebins over which the peak finder algorithm is run (defaults to +/-10 pixels, which works well for JWST data),
- and an optional integer `match_halfwidth` to set the distance from `pos` in wavebins that a peak can be found and still count as a match (defaults to +/- 2 pixels)

...and returns `True` if and only if it finds an abrupt increase in extended emission size/probability very close to the specified input wavebin.

+++

For example, `detect_spikes` returns true when we give it the wavebin index with jet emission that we just looked at:

```{code-cell} ipython3
print(detect_spikes(test_blobs, 548, 3.0))
```

And False if we give it an unremarkable wavebin index:

```{code-cell} ipython3
print(detect_spikes(test_blobs, 540, 3.0))
```

Our third function is simple, and retrieves the wavelength array from a JWST spectral cube file's FITS header:

```{code-cell} ipython3
def get_jwst_wave_array(header, cloud_uri):
    """
    Input parameters
    ----------
    header : the header of a JWST FITS file in which the third axis is wavelength.

    Returns
    ----------
    wave : numpy array of wavelengths in units of CUNIT3 (for JWST, microns).
           Each bin is the wavelength of the corresponding slice of the cube.
    """
    try:
        # Populate wavelength array from the WCS
        wave = header['CRVAL3'] + header['CDELT3'] * np.arange(header['NAXIS3'])
    except:
        # Multi-channel cubes do not have CDELT3,
        # and instead store a nonlinear wavelength solution in extension 'WCS-TABLE'
        _, wavetable = load_cloud(cloud_uri, extension='WCS-TABLE')
        wave = np.asarray(wavetable['wavelength'].flatten())

    # Note: we could more elegantly have used jwst.datamodels in either case,
    # or spectral_cube.SpectralCube.read(file, hdu='SCI').spectral_axis in the first case.
    # But those options would read the whole file into memory,
    # which would increase our resource cost.

    return wave
```

Our fourth function checks certain metadata for each observation to determine if the analysis should proceed:

```{code-cell} ipython3
def check_conditions(row, cloud_uri_map, transitions):
    """
    Input parameters
    ----------
    row : row of an astropy table, or a pandas series representing a row of a dask dataframe
    cloud_uri_map : python dictionary mapping the relevant MAST URIs to AWS cloud URIs
    transitions : python list of spectral line wavelengths to look for, in microns

    Returns
    ----------
    conditions_met : boolean True or False for the union of all conditions
    """
    # Set condition: one of our Fe II wavelengths is in the Observation's spectral range.
    # The cross-mission MAST database that astroquery used gives this range in nanometers.
    condition_1 = any(row['em_min'] <= line*1e3 <= row['em_max'] for line in transitions)

    # Set condition: check that the data is exists and is accessible in the cloud.
    condition_2 = exists(row['dataURL']) and exists(cloud_uri_map[row['dataURL']])

    # Set condition: exclude stale observations with deprecated obs_id pattern 'shortmediumlong'.
    # These products are no longer available due to a change in the JWST pipeline
    condition_3 = ('shortmediumlong' not in row['obs_id'])

    # Apply all our database metadata criteria
    conditions_met = condition_1 and condition_2 and condition_3

    return conditions_met
```

Our fifth and final function serves as a wrapper for our other functions. This is the function that, later, we'll feed directly into a parallelization routine. For each input row (from `tmc1a_jwst_table` in our case), it will check certain metadata to determine if the analysis needs to proceed, then load the spectral cube file into memory, run `extension_vs_wavebin` on the whole data cube, and run `detect_spikes` around the wavebin of each Fe II transition wavelength of interest.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def line_search(row, cloud_uri_map, transitions, plot=False):
    """
    Input parameters
    ----------
    row : row of an astropy table, or a pandas series representing a row of a dask dataframe
    cloud_uri_map : python dictionary mapping the relevant MAST URIs to AWS cloud URIs
    transitions : python list of spectral line wavelengths to look for, in microns

    Returns
    ----------
    plot : boolean determining whether this function generates a summary plot for each "blob" spectrum
    row : an edited row with new column 'detected_feii_lines' populated
    """
    # Initialize a list of Fe II jet emission lines detected in this cube:
    detected_feii_lines = []

    # First, let's check_conditions
    conditions_met = check_conditions(row=row, cloud_uri_map=cloud_uri_map, transitions=transitions)

    # Apply all our database metadata criteria:
    if conditions_met:

        # Load the Observation's spectral cube data file into memory,
        # without downloading a copy to storage.
        cloud_uri = cloud_uri_map[row['dataURL']]
        header, data = load_cloud(cloud_uri)

        # Get the wavelength array for this cube
        wave = get_jwst_wave_array(header, cloud_uri)

        # Run our algorithm to look for extended emission in each slice of the cube
        blobs = extension_vs_wavebin(data)

        # And now our algorithm to check for extended emission peaks at Fe II wavelengths
        for line in transitions:  # For each Fe II wavelength of interest...
            if np.min(wave) <= line <= np.max(wave):  # If the Fe II wavelength is within the bounds of this spectral cube...
                # Find the approximate wavebin index matching the wavelength.
                pos = bisect.bisect_left(wave, line)

                # Look for a spike in the blobs array near this wavelength.
                detected = detect_spikes(blobs, pos, sig=3.0)

                if detected:  # If we found a spike at this wavelength...
                    # ...then this cube may contain extended emission from this Fe+ line.
                    plt.axvline(x=line, color='r', linestyle='--')
                    detected_feii_lines.append(line)

                else:  # Otherwise, probably no Fe+ line extended emission here.
                    plt.axvline(x=line, color='b', linestyle='--')
            if plot:
                plt.plot(wave, blobs, color='k')

    # If plot=True and at least one Fe II line detected,
    # display the corresponding blob detection plot.
    if plot:
        if len(detected_feii_lines) > 0:
            print('obs_id: ' + row['obs_id'])
            print('Detected lines: ' + str(detected_feii_lines))
            print('Plot:')
            plt.xlabel('Wavelength (microns)')
            plt.ylabel('Fraction of non-NaN pixels above threshold')
            plt.show()

    # Return the input row with a new column holding the detected_feii_lines list
    row['detected_feii_lines'] = detected_feii_lines

    return row
```

Phew! Let's try this out on our TMC1A observations.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Make a copy of our observations table with a new column detected_feii_lines to hold results.
copy_tmc1a_jwst_obstable = copy.deepcopy(tmc1a_jwst_obstable)
new_column = Column(name='detected_feii_lines', dtype=object, length=len(tmc1a_jwst_obstable))
copy_tmc1a_jwst_obstable.add_column(new_column)

# Initialize list of astropy table rows to hold results.
lines_tmc1a_jwst = []

# For reach row in our observations table...
for row in copy_tmc1a_jwst_obstable:
    # Execute line_search.
    new_row = line_search(row, cloud_uri_map, transitions, plot=True)
    # Append results.
    lines_tmc1a_jwst.append(new_row)

# Convert results list to astropy table.
lines_tmc1a_jwst_obstable = Table(rows=lines_tmc1a_jwst, names=copy_tmc1a_jwst_obstable.colnames)
```

> Figure 3 image description: a number of line plots similar to Figure 1, except that wavelength in microns is plotted on the horizontal axes instead of wavebin index, and different line plots cover different wavelength ranges. Vertical dashed red lines mark the locations of spikes corresponding to detected Fe II emission. Depending on the data available at the time of plot creation, vertical dashed blue lines may mark the location of non-detected Fe II transitions without co-located spikes.

+++

If we grab all the unique values from inside the python lists tabulated in `lines_tmc1a_jwst_table['detected_feii_lines']`...

```{code-cell} ipython3
all_values = [line for rowlist in lines_tmc1a_jwst_obstable['detected_feii_lines'] for line in rowlist]

detected_lines = set(all_values)

print(detected_lines)
```

We see that we have reproduced the detection of multiple Fe II lines in the YSO TMC1A.

+++

### 2.3 Parallelizing the search for emission lines in spectral cubes

+++

Now, for the real test: let's put everything together and apply this technique to the full table of JWST spectral cube observations of SIMBAD-labeled YSOs.

We'll be working with the `yso_jwst_table` of observations:

```{code-cell} ipython3
print(f'We need to look at {len(yso_jwst_obstable)} observations.')

# Let's look at a couple rows:
yso_jwst_obstable[0:2]
```

As before, let's retrieve the cloud URIs corresponding to this table's `dataURL`s:

```{code-cell} ipython3
# This will take a minute or so.
cloud_uri_map = Observations.get_cloud_uris(list(yso_jwst_obstable['dataURL']), return_uri_map=True, verbose=False)
```

And make a copy of the table with a `detected_feii_lines` column to hold our results:

```{code-cell} ipython3
copy_yso_jwst_obstable = copy.deepcopy(yso_jwst_obstable)
new_column = Column(name='detected_feii_lines', dtype=object, length=len(yso_jwst_obstable))
copy_yso_jwst_obstable.add_column(new_column)
```

Finally, let's use the `dask` parallelizing technique we used before to split `yso_jwst_table` into partitions, and allow our CPUs to split those partitions amongst themselves. You can think of this like sending one spectral cube file to each core, applying `line_search` to the cube, and repeating that until we're done.

```{code-cell} ipython3
yso_jwst_ddf = dd.from_pandas(copy_yso_jwst_obstable.to_pandas(), npartitions=num_cores*4)  # Adjust npartitions to taste
meta = pd.DataFrame(columns=copy_yso_jwst_obstable.to_pandas().columns)
```

```{code-cell} ipython3
# Set up to apply line_search in parallel
yso_jwst_job = yso_jwst_ddf.map_partitions(lambda df: df.apply(lambda row: line_search(row, cloud_uri_map, transitions), axis=1), meta=meta)
```

```{code-cell} ipython3
# Execute. This will take a few minutes.
results = yso_jwst_job.compute()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Convert back to an astropy table.
lines_yso_jwst_obstable = Table.from_pandas(results)

# Remove rows where no Fe II lines were detected
# (where detected_feii_lines is an empty list).
lines_yso_jwst_obstable = lines_yso_jwst_obstable[lines_yso_jwst_obstable['detected_feii_lines'].astype(bool)]

print(f'We found {len(lines_yso_jwst_obstable)} JWST spectral cube observations of YSOs that exhibit possible Fe II emission from a candidate jet, narrowed down from an initial {len(jwst_obstable)} JWST spectral cube observations.')
```

Let's associate each observation to its closest YSO:

```{code-cell} ipython3
# Find the closest coordinates in yso_table for each row in lines_yso_jwst_table

# Drop NaNs from YSO catalog
condition_yso = ~np.isnan(yso_table['ra']) & ~np.isnan(yso_table['dec'])
yso_table_clean = yso_table[condition_yso]

# Get coordinates from observations table
coords_obs = SkyCoord(ra=lines_yso_jwst_obstable['s_ra'] * u.deg, dec=lines_yso_jwst_obstable['s_dec'] * u.deg)

# Get coordinates from YSO catalog
coords_yso = SkyCoord(ra=yso_table_clean['ra'], dec=yso_table_clean['dec'])

# Get closest matches
idx, d2d, d3d = coords_obs.match_to_catalog_sky(coords_yso)

print(idx.shape)
# idx lists the row indices of yso_table_clean that correspond to each of the observations
```

```{code-cell} ipython3
# Add a column to hold this information
new_column = Column(name='yso_index', data=idx)
lines_yso_jwst_obstable.add_column(new_column)
```

Now let's find out how many YSOs had at least 3 Fe II emission lines detected:

```{code-cell} ipython3
# Group the observations by YSO
grouped_lines_yso_jwst_obstable = lines_yso_jwst_obstable.group_by('yso_index')
```

```{code-cell} ipython3
# Instantiate list to hold results
index_linecount = []

# For each group of observations (corresponding to a YSO)...
for group in grouped_lines_yso_jwst_obstable.groups:
    # Get all detected Fe II lines for this YSO
    all_values = [line for rowlist in group['detected_feii_lines'] for line in rowlist]
    detected_lines = set(all_values)

    #Count the number of detected Fe II lines for this YSO
    count_lines_detected = len(detected_lines)

    # If at least 3 lines detected for this YSO...
    if count_lines_detected >= 3:
        # Store the YSO index and detected line count in index_linecount
        index_linecount.append((group['yso_index'][0], count_lines_detected))
```

```{code-cell} ipython3
print(f'Detected candidate jet emission from at least 3 lines of Fe II in {len(index_linecount)} YSOs.')
```

Let's take a quick look at a random YSO from this list. At the time of writing, this was the `yso_index` value corresponding to the first row of `results`:

```{code-cell} ipython3
# For demonstration purposes, so that the rest of this section knows what data to expect,
# get yso_index corresponding to target_name = 'PER-EMB-33'.
temp_yso_index = max(lines_yso_jwst_obstable['yso_index'][lines_yso_jwst_obstable['target_name']=='PER-EMB-33'])

# Show the first few rows of the observations corresponding to that yso_index
lines_yso_jwst_obstable[lines_yso_jwst_obstable['yso_index']==temp_yso_index][0:3]
```

We're already familiar with the 5.34 micron line, so let's take a look at that one of the cubes above that contains it:

```{code-cell} ipython3
# Retrieve all observations for this YSO with MIRI CH1-SHORT and proposal ID 1236
temp_observations = lines_yso_jwst_obstable[
    (lines_yso_jwst_obstable['yso_index']==temp_yso_index) &
    (lines_yso_jwst_obstable['instrument_name']=='MIRI/IFU') &
    (lines_yso_jwst_obstable['filters']=='CH1-SHORT') &
    (lines_yso_jwst_obstable['proposal_id']=='1236')
]

# Get the first valid cloud URI found
for row in temp_observations:
    cloud_uri = cloud_uri_map[row['dataURL']]
    if cloud_uri:
        break  # When a valid URI is found, exit the loop

print('Cloud URI: ', cloud_uri)
```

```{code-cell} ipython3
# Get the data
header, data = load_cloud(cloud_uri)
```

```{code-cell} ipython3
# Find the wavebin for the 5.34 micron line
wave = header['CRVAL3'] + header['CDELT3'] * np.arange(header['NAXIS3']) 
pos = bisect.bisect_left(wave, 5.340)  # Find the approximate wavebin index
print(pos)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[1].imshow(data[pos])  # wavebin for the 5.34 micron line
axes[1].set_xlabel('y')
axes[1].set_ylabel('x')
axes[1].set_title('5.34 micron line')

axes[0].imshow(data[pos-10])  # -10 slices away for comparison
axes[0].set_xlabel('y')
axes[0].set_ylabel('x')
axes[0].set_title('-10 wavebins away')

axes[2].imshow(data[pos+10])  # +10 slices away for comparison
axes[2].set_xlabel('y')
axes[2].set_ylabel('x')
axes[2].set_title('+10 wavebins away')

plt.tight_layout()
plt.show()
```

> Figure 4 image description: three astronomical images in pixel coordinates, side by side, display three different wavebin slices of a spectral cube. The left and right images are labeled "-10 wavebins away" and "+10 wavebins away", respectively, and feature a bright central source with faint hints of adjacent extention to the lower right. The middle image is labeled "5.34 micron line". In the middle image, the central source is much dimmer, emission extends all the way to the lower-right corner, and a blob in the lower-right corner is the brightest feature, in a region where the other two images exhibit no emission.

+++

Looks promising, at least! Let's check what YSO this is...

```{code-cell} ipython3
yso_table_clean[temp_yso_index]
```

Now let's take a look at a 5.34 micron slice from each of the other YSOs where we've assessed that there are at least three Fe II lines detected in extended emission, *and* specifically a 5.34 micron line detection. This will take some complicated `matplotlib` wrangling to pack the plots into a grid, but not much else of note.

```{code-cell} ipython3
# This cell will take up to a minute to run.

transition = 5.34  # Plotting 5.34 micron line

# Set up a grid of plots
num_plots = len(index_linecount)
cols = 4
maxrows = -(num_plots // -cols)  # (num_plots / cols) rounded up

fig = plt.figure(figsize=(15, 5 * maxrows))
gs = GridSpec(maxrows, cols, figure=fig)

i = 0  # instantiate subplot index
# For each YSO meeting our conditions...
for entry in index_linecount:
    temp_yso_index = entry[0]  # Get the yso_index
    yso_name = yso_table_clean['main_id'][temp_yso_index]  # get name from SIMBAD
    # For each observation...
    for row in lines_yso_jwst_obstable:
        #  If this observation contains a detected 5.34 micron slice from this YSO...
        if (row['yso_index'] == temp_yso_index) & (transition in row['detected_feii_lines']):
            cloud_uri = cloud_uri_map[row['dataURL']]  # get cloud URI

            # Open the file into memory
            header, data = load_cloud(cloud_uri)

            # Find the approximate slice
            wave = get_jwst_wave_array(header, cloud_uri)
            pos = bisect.bisect_left(wave, transition)  # Find the approximate wavebin index for the transition.

            # Get the brightest slice up to +/- 2 wavebins away
            stacked_images = np.stack((data[pos], data[pos-1], data[pos-2], data[pos+1], data[pos+2]), axis=0)
            bright_image = np.max(stacked_images, axis=0)

            # Plot the slice
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax.imshow(bright_image, norm=PowerNorm(gamma=0.4))
            ax.set_title(f'{yso_name}')
            ax.axis('off')
            i += 1  # go to next subplot index
            break  # If we got a plot for this YSO, skip to the next YSO

plt.tight_layout()
plt.show()
```

> Figure 5 image description: Tens of images plotted in a grid, with an object name labeling each image. Most of the images exhibit obvious extended emission in a wide variety of complex morphologies.

+++

At a glance, it looks like we've done a pretty good job of detecting candidate spatially-extended Fe II emission! Further investigation, validation, and improvement are left to the reader as an exercise.

+++

## 3. Postscript: Transposing to IRSA missions

+++

We note that the spectral cubes in [IRSA](https://irsa.ipac.caltech.edu/frontpage/) from Herschel and SOFIA would not be as easy to integrate into this method, largely due to the smaller field of view and the sharp variations in the number of image pixels between adjacent wavebin slices. But with a different set of algorithms in `detect_spikes` and `extension_vs_wavebin`, and a different set of emission lines of interest (say from [Karska et al. 2025](https://arxiv.org/pdf/2503.15059)), it might be possible.

That is left as an *extremely challenging* exercise for the reader, but we'll get you started retrieving spectral cube data for SOFIA...

```{code-cell} ipython3
# Astroquery call
sofia_temp = Irsa.query_sia(facility='SOFIA', calib_level=3, data_type='cube', format='cube/fits')

# Drop rows without footprints
sofia_obstable = sofia_temp[np.array(['POLYGON' in str(region) for region in sofia_temp['s_region']])]
```

...and crossmatching these footprints to the SIMBAD YSOs:

```{code-cell} ipython3
# Convert jwst_table into a partitioned Dask dataframe,
# so that the cores can split the partitions amongst themselves.
sofia_obsddf = dd.from_pandas(sofia_obstable.to_pandas(), npartitions=num_cores*4)  # Adjust npartitions to taste

# Apply the check_point_in_polygon function in parallel.
meta = pd.DataFrame(columns=sofia_obstable.to_pandas().columns)
sofia_job = sofia_obsddf.map_partitions(lambda df: df.apply(lambda row: check_points_in_polygon(row, yso_coords), axis=1), meta=meta)
results = sofia_job.compute()

# Remove empty rows, where check_points_in_polygon reached the "else" part of the return statement.
results = results.dropna(how='all')

# Convert back to an astropy table.
yso_sofia_obstable = Table.from_pandas(results)
```

```{code-cell} ipython3
# Let's take a look at the first row
yso_sofia_obstable[0]
```

Note a few other differences worth keeping in mind:

```{code-cell} ipython3
# Retrieve data through the link in the access_url column
print('HTTP URL: ' + yso_sofia_obstable['access_url'][0])

# or maybe in the future the cloud_access column, though this is not available for SOFIA right now
print('Cloud URI: ' + yso_sofia_obstable['cloud_access'][0])
```

```{code-cell} ipython3
# em_min and em_max are in meters
print(yso_sofia_obstable['em_min'][0], yso_sofia_obstable['em_max'][0])
```

```{code-cell} ipython3
# The data cubes are in each file's 'FLUX' extension
with fits.open(yso_sofia_obstable['access_url'][0], cache=False, use_fsspec=True) as hdul:
    header = hdul['FLUX'].header
    data = hdul['FLUX'].data
```

In the future, spectral cubes in IRSA from [the new SPHEREx mission](https://spherex.caltech.edu/page/data-products) might be more readily integrated into the workflow we've demonstrated in this notebook, so stay tuned!

+++

## About this notebook

+++

### Author

Adrian Lucy, MAST, alucy@stsci.edu

### Acknowledgements

MAST internal review: Thomas Dutkiewicz, Sam Bianco, Zach Claytor, Brian McLean, Jonathan Hargis

Fornax review: TBD

### Publication date

TBD

### References

This notebook relies on the following papers:
- Assani et al. 2025 ([2025arXiv250402136A](https://ui.adsabs.harvard.edu/abs/2025arXiv250402136A/abstract))
- Karska et al. 2025 ([2025A&A...697A.186K](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.186K/abstract))

And the following software:
- Astroquery; Ginsburg et al. 2019 (2019AJ….157…98G)
- Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013 (2022ApJ…935..167A, 2018AJ….156..123A, 2013A&A…558A..33A)
- Matplotlib; Hunter 2007 (2007CSE.....9...90H)
- SciPy; Pauli et al. 2020 (2020NatMe..17..261V)
- Dask: Library for dynamic task scheduling; Dask Development Team 2016, http://dask.pydata.org
