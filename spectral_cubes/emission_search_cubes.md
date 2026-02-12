---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Analytical data search in the cloud: finding jets in JWST spectral cubes

+++

## Learning goals

In this notebook, we'll learn about how to answer data-intensive archival search queries in astronomical data archives like [MAST](https://archive.stsci.edu/), [IRSA](https://irsa.ipac.caltech.edu/frontpage/), and the [HEASARC](https://heasarc.gsfc.nasa.gov/docs/archive.html). We'll have to think about:
* when to parallelize our data processing,
* the order of operations for performing cone searches on large numbers of targets,
* when to load data into memory versus downloading to storage,
* the advantages and constraints of a cloud science platform,
* and when to load data from cloud storage instead of on-premise storage.

## Introduction

Inspired by the jet in [Assani et al. 2025](https://arxiv.org/pdf/2504.02136), this notebook will consider **how to find more JWST spectral image cubes containing Fe II emission from jets launched by young stellar objects (YSOs).**

"This observation contains a JWST spectral cube" is certainly something readily available from MAST's metadata tables, but "this file contains Fe II transitions" is certainly not. So instead, we need to dig beyond the metadata and into the data of thousands of files.

### Runtime

As of December 2025, running this notebook once will take about 10 minutes on the Fornax "Large" server (64 GB RAM / 16 CPU), or about 25 minutes on the Fornax "Medium" server (16 GB RAM / 4 CPU).

+++

## Imports

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_emission_search_cubes.txt
```

```{code-cell} ipython3
import os
import sys
import copy

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

# Parallelize
from multiprocessing import Pool
import pickle
import importlib
```

We'll also need a couple local functions written specifically for this notebook. These are located in the `code_src` directory, and you can read them there. However, some of them are pretty long, and their detailed implementation isn't critical to understand. As long as you understand their inputs and outputs, which we'll discuss later, you can write better functions to replace them, tailored to your own science case.

```{code-cell} ipython3
# Local code imports
from code_src.parse_polygon import parse_polygon
from code_src.find_extended_emission import extension_vs_wavebin, detect_spikes, not_none
```

Let's also turn on `enable_cloud_dataset` for `astroquery.mast.Observations`. This will allow us to fetch the cloud locations for data products and access data directly from the cloud, instead of retrieving data from MAST's on-premise servers.

```{code-cell} ipython3
Observations.enable_cloud_dataset()
```

## 1. Finding JWST spectral cubes of YSOs

+++

Let's ignore the emission line issue and tackle the easiest parts of our driving question first: *I want to find more JWST spectral cubes of young stellar objects.*

+++

### 1.1 Querying SIMBAD for YSOs

+++

At this time, MAST doesn't support reliable object classification search. So let's go to the experts in astronomical object cataloging, SIMBAD, which we can access through `astroquery.simbad.Simbad.query_tap`.

In the cell below, we search for all SIMBAD-catalogued objects labeled as YSOs (`otype='Y*O'`) or as any of the descendant sub-concepts of YSOs, like T Tauri stars (`otype='Y*O..'` to retrieve both explicitly labeled YSOs and their subtypes). This cell will take a minute or two.

```{code-cell} ipython3
yso_table = Simbad.query_tap("SELECT * FROM basic WHERE otype='Y*O..'", maxrec=1000000)
```

```{code-cell} ipython3
print(f'We found {len(yso_table)} YSOs.')
```

Let's drop YSOs that lack coordinates:

```{code-cell} ipython3
# Drop NaNs from YSO catalog
condition_yso = ~np.isnan(yso_table['ra']) & ~np.isnan(yso_table['dec'])
yso_table_clean = yso_table[condition_yso]
print(f'{len(yso_table_clean)} of these YSOs have good coordinates.')
```

### 1.2 Querying MAST for all JWST spectral cube observations

+++

Now we want to get all the JWST spectral cube observations in MAST whose sky footprints overlap with any of these YSO coordinates. The natural tool is the [`astroquery.mast.Observations`](https://astroquery.readthedocs.io/en/latest/api/astroquery.mast.ObservationsClass.html#astroquery.mast.ObservationsClass) class, which gives programmatic access to MAST's multi-mission archive.

However, [`astroquery.mast`](https://astroquery.readthedocs.io/en/latest/mast/mast.html) is not currently set up to quickly run a multi-target query for tens of thousands of sky coordinates. So in order to get things done in a reasonable amount of time, we need to think about the order of operations here.

It turns out that, at the time of writing, there are only a few tens of thousands of JWST spectral cubes across the whole sky. So let's retrieve *all* of those first, excluding only those with a `calib_level` of `-1` (planned observations that haven't yet executed) and non-public datasets.

:::{note}
This cell will typically take anywhere from a few seconds to a minute or so, depending on how many people are using MAST right now. Rarely, if MAST is overloaded, it may take several minutes, or time out.
:::

```{code-cell} ipython3
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

This `s_region` denotes the vertices of a polygon in right ascension and declination. We can parse this string into an array of tuple coordinates with the custom `parse_polygon` function, which we imported from the `code_src` directory for this notebook.

:::{warning}
As currently written, the `s_region` passed into `parse_polygon` must be a single polygon. This is the case for all JWST and SOFIA spectral cube observations, but not all missions in the Fornax archives.
:::

:::{warning}
If you replace `parse_polygon` with another function, be aware that the array of polygon vertices we later pass into `matplotlib.path.Path` must be *closed*, meaning that the last vertex is identical to the first vertex.
:::

```{code-cell} ipython3
# For example...
print(parse_polygon(jwst_obstable[0]['s_region']))
```

In a moment, we'll parse our polygons and use `Matplotlib`'s `Path` class to test whether any of our YSO coordinates are in each observation's sky footprint. `Path.contains_points` expects an array of test coordinates in an *(N, 2)* shape, where N is the number of coordinates to test. Let's make an array of YSO coordinates:

```{code-cell} ipython3
yso_coords = np.transpose(np.asarray((yso_table_clean['ra'], yso_table_clean['dec'])))
print(yso_coords.shape)
```

Performing the test on all observation sky footprints without parallelizing would take quite a while; e.g., six or seven minutes on the Fornax 16-CPU server for the following cell, which implements a simplified version of the crossmatch algorithm that we'll use in the next section. On Fornax, if you uncomment this cell and open the "Kernel usage" tab to your right while it runs, you'll notice that you're using only a small percentage of the total CPU capacity that you have access to (e.g., less than 10% if you selected the 16-CPU server option when you launched Fornax). Feel free to click the `Interrupt the kernel` stop button on your notebook, because there's a better way!

```{code-cell} ipython3
# yso_jwst_obs = []  # Instantiate a list of observations whose footprints overlap with any of our YSO coordinates.

# for obs in jwst_obstable:  # Looping through all JWST spectral cube observations...
#     polygon = parse_polygon(obs['s_region'])  # Parse the polygon

#     polygon_path = Path(polygon, closed=True)  # Convert to matplotlib Path
#     is_inside = polygon_path.contains_points(yso_coords).any()  # Test whether any coordinate from yso_coord is inside this observation's polygon

#     if is_inside:
#         yso_jwst_obs.append(obs)

# yso_jwst_obstable = Table(rows=yso_jwst_obs, names=jwst_obstable.colnames)  # Convert back to an astropy table.

# print(f'We found {len(yso_jwst_obstable)} JWST spectral cube observations whose footprints overlap with YSO coordinates.')
```

### 1.4 Refining and parallelizing the footprint crossmatch

+++

That cell would take a long time to run, so let's parallelize this code to save time! First we'll need to define a function that does what we did above, for each observation, so that we can pass this function into our parallelization tool.

:::{tip}
While we're at it, we'll also add in a subtle nuance; *RA* and *dec* are spherical coordinates, so the segments of a footprint polygon should be great circles on the celestial sphere, not straight lines. However, they can be treated as straight lines if we project into a 2D cartesian coordinate system, to an excellent approximation in the vicinity of the center of the projection. An appropriate choice is the gnomonic (`TAN`) projection, which is the projection used in the JWST FITS images from which `s_region` values were derived. Before checking which YSO coordinates are inside a particular observation's footprint, we'll convert both the footprint's polygon vertices and the YSO coordinates to a gnomonic projection centered on that specific observation's coordinates.

We mentioned that the projection issue is a subtle nuance; in fact, at the time of writing, you could omit the projection step and still get the same number of observations matched. YSO coordinates tend not to be at the very edges of the observation footprints, after all. Nevertheless, to avoid any future problems, let's do things properly!

Note that we are not accounting for proper motions—and neither would `astroquery.mast` cone search, if we were using it. See [High PM Stars in MAST](https://outerspace.stsci.edu/display/MASTDOCS/High+PM+Stars+in+MAST) if this is a concern for your science case.
:::

On macOS and Windows operating systems, functions defined in a Jupyter notebook only exist in-memory in the main process, and don't propagate into the parallelized child processes created by `multiprocessing`. This notebook is written to work on any operating system. So instead of directly executing function definition below, we'll use `%%writefile` to save it (and the imports on which it depends) to a file, then import the function from that file.

```{code-cell} ipython3
%%writefile code_src/temp_check_points_in_polygon.py

from matplotlib.path import Path
from astropy.wcs import WCS
import pandas as pd
import numpy as np
import pickle
from .parse_polygon import parse_polygon

def check_points_in_polygon(obs, coordinates_to_test=None):
    """
    Check if coordinates_to_test are in the s_region polygon for an observation.

    Parameters
    ----------
    obs : pandas.Series
        A row of a pandas dataframe, converted from an astroquery-flavored astropy Table
        Columns must at least include s_ra (RA in deg), s_dec (Declination in deg),
        and s_region (STC-S formatted string).
    coordinates_to_test: np.ndarray, optional
        Array of celestial coordinates shaped like (N, 2), where N is the number of targets.
        If not provided, this array must be serialized to a file coords.pkl available in the
        local directory.

    Returns
    ----------
    pandas.Series
        Either `obs` (same as input) if a match was found, or an empty row of NaNs if a match
        was not found.
    """

    # If coordinates_to_test is not provided (e.g. if multiprocessing),
    # assume it can be read from a local file named coords.pkl.
    if coordinates_to_test is None:
        with open("coords.pkl", "rb") as f:
            coordinates_to_test = pickle.load(f)

    # Make WCS for a gnomonic (TAN) projection
    # with a projection center at the observation's coordinates
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0., 0.]  # Put center of the projection at projected pixel coords (0, 0)
    wcs.wcs.cdelt = [1., 1.]  # Arbitrary 1 degree per pixel (doesn't matter)
    wcs.wcs.crval = [obs['s_ra'], obs['s_dec']]  # Projection center is the observation's coordinates
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Gnomonic projection

    # One option now would be to convert to astropy SkyCoords
    # and used astropy PolygonSkyRegion.contains(coordinates_to_test, wcs),
    # but the following technique yields >2x faster performance:

    # Convert coordinates_to_test to the gnomonic projection
    proj_coordinates_to_test = wcs.wcs_world2pix(coordinates_to_test, 0)

    # Make footprint polygon matplotlib Path for this observation
    polygon = parse_polygon(obs['s_region'])  # Parse s_region value
    proj_polygon = wcs.wcs_world2pix(polygon, 0)  # Convert to gnomonic projection
    proj_polygon_path = Path(proj_polygon, closed=True) # Convert to matplotlib Path

    # Test whether any coordinate from yso_coord is inside this observation's polygon_path.
    if proj_polygon_path.contains_points(proj_coordinates_to_test).any():
        # Return the table row if the match is successful.
        return obs
    else:
        # otherwise return an empty row full of NaNs.
        return pd.Series([np.nan] * len(obs), index=obs.index)
```

When importing the function we just wrote, it's important to `reload` first in case you make changes to the saved function while you work with this notebook.

```{code-cell} ipython3
try:
    importlib.reload(sys.modules['code_src.temp_check_points_in_polygon'])
except KeyError:
    pass

from code_src.temp_check_points_in_polygon import check_points_in_polygon
```

`Multiprocessing` is a python module for parallelizing your code across multiple CPU cores. For this task, we'll set the number of workers to the number of cores...

```{code-cell} ipython3
num_cores = os.cpu_count()
num_workers = num_cores
print(f'We have access to {num_cores} cores, setting to {num_workers} workers.')
```

...and run `check_points_in_polygon` on the rows of our observations table in parallel. To do that, we'll use `multiprocessing`'s `map` method, which allows us to apply a function in parallel to each item (in this case, a row of `jwst_obstable` representing an observation) in an iterable. To improve performance, we'll first convert `jwst_obstable` to a `pandas` dataframe, and use the latter's `iterrows` method to create the iterable.

```{code-cell} ipython3
jwst_obstable_iterable = jwst_obstable.to_pandas().iterrows()
```

Additionally, because yso_coords will be the same for each row of the observations table, we get better performance if we don't pass it in every time, but instead [pickle](https://docs.python.org/3/library/pickle.html) it, and unpickle it into memory in every child process. We already set this up with the following section from `check_points_in_polygon`...
```
    if coordinates_to_test is None:
        with open("coords.pkl", "rb") as f:
            coordinates_to_test = pickle.load(f)
```
...so now we just need to serialize `yso_coords` as the pickle file `coords.pkl` which `check_points_in_polygon` expects:

```{code-cell} ipython3
with open("coords.pkl", "wb") as f:
    pickle.dump(yso_coords, f)
```

If you examine the `Kernel usage` tab to your right while running the next cell, you'll see that you're now using nearly 100% of your multi-CPU capacity!

```{code-cell} ipython3
# Apply check_points_in_polygon to each item in the iterable, in parallel
with Pool(num_workers) as pool:
    results = pool.map(check_points_in_polygon, [obs for _, obs in jwst_obstable_iterable])
```

```{code-cell} ipython3
# Convert back to a pandas dataframe
results = pd.DataFrame(results)

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
for obs in yso_jwst_obstable:
    polygon = parse_polygon(obs['s_region'])
    polygon_path = Path(polygon, closed=True)
    # We'll use matplotlib.Path.contains_point (singular) this time,
    # instead of .contains_points (plural)
    is_inside = polygon_path.contains_point((tmc1a_ra, tmc1a_dec))
    if is_inside:
        tmc1a_jwst.append(obs)

# Convert to an astropy Table
tmc1a_jwst_obstable = Table(rows=tmc1a_jwst, names=jwst_obstable.colnames)

print(f'We found {len(tmc1a_jwst_obstable)} JWST spectral cube observations of TMC1A.')
```

Let's get the AWS cloud URIs for these spectral cube files. Downloading files from AWS (via a cloud URI) isn't always *faster* than downloading from MAST's on-premise servers (via a `dataURI` converted into an HTTP URL), even when the destination for your downloads is an AWS cloud platform like Fornax. But importantly, AWS is more *robust* for large data downloads. By retrieving files from AWS instead of MAST, you avoid the risk of overwhelming MAST's on-premise servers and causing problems for other users, and the substantial risk that MAST's on-premise servers may be overwhelmed by requests from other users in the moment that you want to get your data. AWS is, we believe, less likely to be overwhelmed.

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
cloud_uri_map = Observations.get_cloud_uris(tmc1a_jwst_obstable['dataURL'], return_uri_map=True, verbose=False)
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
for obs in temp_observations:
    cloud_uri = cloud_uri_map[obs['dataURL']]
    if cloud_uri:
        break  # When a valid URI is found, exit the loop

print('Cloud URI: ', cloud_uri)
```

MAST cloud data is in the [Registry of Open Data on AWS](https://registry.opendata.aws/collab/stsci/), so it's configured to allow non-credentialed anonymous access. We can load this file into `astropy.fits` by passing in the cloud URI in as if it were a normal URL, but we need to tell `astropy.fits` to access the file anonymously. More information is available [in the astropy documentation](https://docs.astropy.org/en/stable/io/fits/usage/cloud.html).

```{code-cell} ipython3
with fits.open(cloud_uri, cache=False, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
    header = hdul['SCI'].header
    data = hdul['SCI'].data
```

Let's turn this into a convenience function that we can use later on:

```{code-cell} ipython3
%%writefile code_src/temp_load_cloud_data.py

from astropy.io import fits

def load_cloud_data(cloud_uri, extension='SCI'):
    """
    Load from a cloud URI into memory, without downloading to storage.

    Parameters
    ----------
    cloud_uri : str
        An AWS cloud URI for a FITS file.
    extension : str, optional
        The FITS extension in which the desired data is found. Defaults to 'SCI'.

    Returns
    ----------
    header : astropy.io.fits.header.Header
        Header of the specified FITS extension
    data : np.ndarray
        Data array of the specified FITS extension
    """
    with fits.open(cloud_uri, cache=False, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        header = hdul[extension].header
        data = hdul[extension].data
    return header, data
```

```{code-cell} ipython3
try:
    importlib.reload(sys.modules['code_src.temp_load_cloud_data'])
except KeyError:
    pass

from code_src.temp_load_cloud_data import load_cloud_data
```

Remember that you have a [limited storage allocation in your Fornax account](https://docs.fornax.sciencecloud.nasa.gov/user-resource-allotments-and-costs/). In our case, when we analyze the full set of YSOs, we expect to need at least about 100 GB worth of data net across many small ~10 MB files. Thus, it is important that we are using `cache=False` and `use_fsspec=True` (which are the default settings when passing a cloud URI into `astropy.fits`) to load the data into memory (RAM) rather than into storage. This can similarly be enforced for downloads from on-premise HTTP links, if cloud URIs are not available, by explicitly passing both `cache=False` and `use_fsspec=True` into the `open` function.

+++

### 2.2 Searching for emission lines in spectral cubes

+++

Let's copy down some of the Fe II wavelengths from [Assani et al. 2025](https://arxiv.org/pdf/2504.02136):

```{code-cell} ipython3
transitions = [25.988, 24.519, 17.936, 5.340, 1.644, 1.600, 1.257]*u.um
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

> Figure 1 alt text: a line plot of "Fraction of non-NaN pixels above threshold" versus "Wavelength bin index", the latter ranging from around 0 to 1000. The plotted values have substantial scatter and features, but one feature stands out by a factor of two above all others: a narrow, abrupt spike near wavelength bin index 548.

+++

We see a variety of messy features, perhaps corresponding to noise or to the brightness of the central star, but we also see a dramatic single-bin spike at around the middle wavebin of the cube. Let's take a look at that slice of the cube, where we see an extended jet emerging from a YSO point source. This is the basis for one of the Fe II line images from Fig. 1 of [Assani et al. 2025](https://arxiv.org/pdf/2504.02136).

```{code-cell} ipython3
plt.imshow(data[548], norm='log')
plt.xlabel('y')
plt.ylabel('x')
plt.show()
```

> Figure 2 alt text: 2-dimensional astronomical image in pixel coordinates, with rough boundaries of null values near the edges. From a bright circular source near the top edge, a faint hint of jet emission extends downwards and slightly to the right, nearly all the way to the lower edge of the image.

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
%%writefile code_src/temp_get_jwst_wave_array.py

import astropy.units as u
import numpy as np
from .temp_load_cloud_data import load_cloud_data

def get_jwst_wave_array(header, cloud_uri):
    """
    Get wavelength coordinates array from a JWST spectral cube FITS file.
    
    Parameters
    ----------
    header : astropy.io.fits.header.Header
        The header of a JWST FITS file in which the third axis is wavelength.
    cloud_uri: str
        An AWS cloud URI for a FITS file.

    Returns
    ----------
    wave : astropy.units.quantity.Quantity
        Quantity array of wavelengths in units of CUNIT3 (for JWST, microns).
        Each bin is the wavelength of the corresponding slice of the cube.
    """
    # Note: we could more elegantly have used jwst.datamodels in both cases below,
    # or spectral_cube.SpectralCube.read(file, hdu='SCI').spectral_axis in the first case.
    # But those options would read the whole file into memory,
    # which would increase our resource cost.
    
    try:
        # Populate wavelength array from the WCS
        wave = header['CRVAL3'] + header['CDELT3'] * np.arange(header['NAXIS3'])
    except KeyError:
        # Multi-channel cubes do not have CDELT3,
        # and instead store a nonlinear wavelength solution in extension 'WCS-TABLE'
        _, wavetable = load_cloud_data(cloud_uri, extension='WCS-TABLE')
        wave = np.asarray(wavetable['wavelength'].flatten())

    return wave * u.Unit(header['CUNIT3'])
```

Our fourth function checks certain metadata for each observation to determine if the analysis should proceed:

```{code-cell} ipython3
%%writefile code_src/temp_check_conditions.py

import astropy.units as u
import pandas as pd
from .find_extended_emission import not_none

def check_conditions(obs, cloud_uri_map, transitions):
    """
    Check an observation's metadata for a set of conditions.

    Parameters
    ----------
    obs : astropy.table.Row, pandas.Series
        Row of an astropy or pandas table of observations
        Columns must at least include dataURL (MAST URI for main product in an observation),
        obs_id (MAST observation ID), and em_min + em_max (wavelength bounds in nm).
    cloud_uri_map : dict[str, str]
        Python dictionary mapping the relevant MAST URIs to AWS cloud URIs
    transitions : astropy.units.quantity.Quantity
        List of spectral line wavelengths to look for, as astropy Quantities

    Returns
    ----------
    conditions_met : bool
        Boolean True or False for the union of all conditions
    """
    # Set condition: one of our Fe II wavelengths is in the Observation's spectral range.
    # The cross-mission MAST database that astroquery used gives this range in nanometers.
    condition_1 = any(obs['em_min']*u.nm <= line <= obs['em_max']*u.nm for line in transitions)

    # Set condition: check that the data is exists and is accessible in the cloud.
    # not_none is a custom function we imported earlier in the notebook,
    # and checks for None and pandas.NA
    condition_2 = not_none(obs['dataURL']) and not_none(cloud_uri_map[obs['dataURL']])

    # Set condition: exclude stale observations with deprecated obs_id pattern 'shortmediumlong'.
    # These products are no longer available due to a change in the JWST pipeline
    condition_3 = ('shortmediumlong' not in obs['obs_id'])

    # Apply all our database metadata criteria
    conditions_met = condition_1 and condition_2 and condition_3

    return conditions_met
```

Our fifth and final function serves as a wrapper for our other functions. This is the function that, later, we'll feed directly into a parallelization routine. For each input row (from `tmc1a_jwst_table` in our case), it will check certain metadata to determine if the analysis needs to proceed, then load the spectral cube file into memory, run `extension_vs_wavebin` on the whole data cube, and run `detect_spikes` around the wavebin of each Fe II transition wavelength of interest.

```{code-cell} ipython3
%%writefile code_src/temp_line_search.py

import pickle
import numpy as np
import bisect
import matplotlib.pyplot as plt
import astropy.units as u
from .find_extended_emission import extension_vs_wavebin, detect_spikes
from .temp_load_cloud_data import load_cloud_data
from .temp_get_jwst_wave_array import get_jwst_wave_array
from .temp_check_conditions import check_conditions

def line_search(obs, cloud_uri_map=None, transitions=None, plot=False):
    """
    Search a spectral cube observation for emission lines.

    Parameters
    ----------
    obs : astropy.table.Row, pandas.Series
        Row of an astropy or pandas table of observations
        Columns must at least include dataURL (MAST URI for main product in an observation)
        and obs_id (MAST observation ID).
    cloud_uri_map : dict[str, str], optional
        Python dictionary mapping the relevant MAST URIs to AWS cloud URIs.
        If not provided, defaults to looking for cloud_uri_map.pkl in local directory.
    transitions : astropy.units.quantity.Quantity, optional
        List of spectral line wavelengths to look for, as astropy Quantities
        If not provided, defaults to looking for transitions.pkl in local directory.

    Returns
    ----------
    plot : bool
        Boolean determining whether this function generates a summary plot for each "blob" spectrum
    obs : astropy.table.Row, pandas.Series
        An edited row of the observations table with new column 'detected_feii_lines' populated
    """

    # If cloud_uri_map and transitions not provided,
    # assume that they can be read from pickle files 
    if cloud_uri_map is None:
        with open("cloud_uri_map.pkl", "rb") as f:
            cloud_uri_map = pickle.load(f)
    if transitions is None:
        with open("transitions.pkl", "rb") as f:
            transitions = pickle.load(f)
    
    # Initialize a list of Fe II jet emission lines detected in this cube:
    detected_feii_lines = []

    # First, let's check_conditions
    conditions_met = check_conditions(obs=obs, cloud_uri_map=cloud_uri_map, transitions=transitions)

    # Apply all our database metadata criteria:
    if conditions_met:

        # Load the Observation's spectral cube data file into memory,
        # without downloading a copy to storage.
        cloud_uri = cloud_uri_map[obs['dataURL']]
        header, data = load_cloud_data(cloud_uri)

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
                    plt.axvline(x=line.value, color='r', linestyle='--')
                    detected_feii_lines.append(line)

                else:  # Otherwise, probably no Fe+ line extended emission here.
                    plt.axvline(x=line.value, color='b', linestyle='--')
            if plot:
                plt.plot(wave, blobs, color='k')

    # If plot=True and at least one Fe II line detected,
    # display the corresponding blob detection plot.
    if plot:
        if len(detected_feii_lines) > 0:
            print('obs_id: ' + obs['obs_id'])
            print('Detected lines: ', u.Quantity(detected_feii_lines))
            print('Plot:')
            plt.xlabel('Wavelength (microns)')
            plt.ylabel('Fraction of non-NaN pixels above threshold')
            plt.show()

    # Return the input row with a new column holding the detected_feii_lines list
    obs['detected_feii_lines'] = detected_feii_lines

    return obs
```

```{code-cell} ipython3
try:
    importlib.reload(sys.modules['code_src.temp_get_jwst_wave_array'])
    importlib.reload(sys.modules['code_src.temp_check_conditions'])
    importlib.reload(sys.modules['code_src.temp_line_search'])
except KeyError:
    pass

from code_src.temp_get_jwst_wave_array import get_jwst_wave_array
from code_src.temp_check_conditions import check_conditions
from code_src.temp_line_search import line_search
```

Let's try this out on our TMC1A observations:

```{code-cell} ipython3
# Make a copy of our observations table with a new column detected_feii_lines to hold results.
copy_tmc1a_jwst_obstable = copy.deepcopy(tmc1a_jwst_obstable)
new_column = Column(name='detected_feii_lines', dtype=object, length=len(tmc1a_jwst_obstable))
copy_tmc1a_jwst_obstable.add_column(new_column)

# Initialize list of astropy table rows to hold results.
lines_tmc1a_jwst = []

# For reach row in our observations table...
for obs in copy_tmc1a_jwst_obstable:
    # Execute line_search.
    new_row = line_search(obs, cloud_uri_map, transitions, plot=True)
    # Append results.
    lines_tmc1a_jwst.append(new_row)

# Convert results list to astropy table.
lines_tmc1a_jwst_obstable = Table(rows=lines_tmc1a_jwst, names=copy_tmc1a_jwst_obstable.colnames)
```

> Figure 3 alt text: a number of line plots similar to Figure 1, except that wavelength in microns is plotted on the horizontal axes instead of wavebin index, and different line plots cover different wavelength ranges. Vertical dashed red lines mark the locations of spikes corresponding to detected Fe II emission. Depending on the data available at the time of plot creation, vertical dashed blue lines may mark the location of non-detected Fe II transitions without co-located spikes.

+++

If we grab all the unique values from inside the python lists tabulated in `lines_tmc1a_jwst_table['detected_feii_lines']`...

```{code-cell} ipython3
all_values = [line for linelist in lines_tmc1a_jwst_obstable['detected_feii_lines'] for line in linelist]

detected_lines = set(all_values)

print(u.Quantity(list(detected_lines)))
```

We see that we have reproduced the detection of multiple Fe II lines in the YSO TMC1A.

+++

### 2.3 Parallelizing the search for emission lines in spectral cubes

+++

Now, for the real test: let's put everything together and apply this technique to the full table of JWST spectral cube observations of SIMBAD-labeled YSOs.

We'll be working with the `yso_jwst_table` of observations:

```{code-cell} ipython3
print(f'We need to look at {len(yso_jwst_obstable)} observations.')

# Let's look at a couple observations:
yso_jwst_obstable[0:2]
```

As before, let's retrieve the cloud URIs corresponding to this table's `dataURL`s. This cell will typically take about a minute or so:

```{code-cell} ipython3
cloud_uri_map = Observations.get_cloud_uris(yso_jwst_obstable['dataURL'], return_uri_map=True, verbose=False)
```

And make a copy of the table with a `detected_feii_lines` column to hold our results:

```{code-cell} ipython3
copy_yso_jwst_obstable = copy.deepcopy(yso_jwst_obstable)
new_column = Column(name='detected_feii_lines', dtype=object, length=len(yso_jwst_obstable))
copy_yso_jwst_obstable.add_column(new_column)
```

Finally, let's use the parallelizing technique we used before to apply `line_search` to the rows (observations) of `yso_jwst_obstable` in parallel across our CPU cores.

Once again, for objects that stay constant as a function of iteration, we improve performance by using pickle files to feed them into our algorithm.

```{code-cell} ipython3
with open("cloud_uri_map.pkl", "wb") as f:
    pickle.dump(cloud_uri_map, f)

with open("transitions.pkl", "wb") as f:
    pickle.dump(transitions, f)
```

We will make one minor modification to the parallelization approach that we used before. Empirically, we can get mildly better performance in this case (using nearly 100% of our available CPU capacity instead of 70-80%) by having slightly more workers than cores. This is probably because reading data from FITS files means that there's an I/O-bound component to the `line_search` function. When a core is waiting on an I/O task (like reading a FITS file), another worker can use that core for computations.

```{code-cell} ipython3
# Use pandas to make the iterable
copy_yso_jwst_obstable_iterable = copy_yso_jwst_obstable.to_pandas().iterrows()

# Increase the number of workers to compensate for I/O bound tasks
if num_cores<=4:
    num_workers = round(num_cores*2)
else:
    num_workers = round(num_cores*1.4)

# Execute
with Pool(num_workers) as pool:
    results = pool.map(line_search,
                       [obs for _, obs in copy_yso_jwst_obstable_iterable])
```

```{code-cell} ipython3
# Convert back to a pandas dataframe
results = pd.DataFrame(results)

# Convert back to an astropy table.
lines_yso_jwst_obstable = Table.from_pandas(results)

# Remove observations where no Fe II lines were detected
# (where detected_feii_lines is an empty list).
lines_yso_jwst_obstable = lines_yso_jwst_obstable[lines_yso_jwst_obstable['detected_feii_lines'].astype(bool)]

print(f'We found {len(lines_yso_jwst_obstable)} JWST spectral cube observations of YSOs that exhibit possible Fe II emission from a candidate jet, narrowed down from an initial {len(jwst_obstable)} JWST spectral cube observations.')
```

Let's associate each observation to its closest YSO:

```{code-cell} ipython3
# Find the closest coordinates in yso_table_clean for each observation in lines_yso_jwst_table

# Get coordinates from observations table
coords_obs = SkyCoord(ra=lines_yso_jwst_obstable['s_ra'] * u.deg, dec=lines_yso_jwst_obstable['s_dec'] * u.deg)

# Get coordinates from YSO catalog
coords_yso = SkyCoord(ra=yso_table_clean['ra'], dec=yso_table_clean['dec'])

# Get closest matches
idx, d2d, d3d = coords_obs.match_to_catalog_sky(coords_yso)

print(idx.shape)
# idx lists the table row indices in yso_table_clean that correspond to each of the observations
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
    all_values = [line for linelist in group['detected_feii_lines'] for line in linelist]
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

# Show the first few observations corresponding to that yso_index
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
for obs in temp_observations:
    cloud_uri = cloud_uri_map[obs['dataURL']]
    if cloud_uri:
        break  # When a valid URI is found, exit the loop

print('Cloud URI: ', cloud_uri)
```

```{code-cell} ipython3
# Get the data
header, data = load_cloud_data(cloud_uri)
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

> Figure 4 alt text: three astronomical images in pixel coordinates, side by side, display three different wavebin slices of a spectral cube. The left and right images are labeled "-10 wavebins away" and "+10 wavebins away", respectively, and feature a bright central source with faint hints of adjacent extention to the lower right. The middle image is labeled "5.34 micron line". In the middle image, the central source is much dimmer, emission extends all the way to the lower-right corner, and a blob in the lower-right corner is the brightest feature, in a region where the other two images exhibit no emission.

+++

Looks promising, at least! Let's check what YSO this is...

```{code-cell} ipython3
yso_table_clean[temp_yso_index]
```

Now let's take a look at a 5.34 micron slice from each of the other YSOs where we've assessed that there are at least three Fe II lines detected in extended emission, *and* specifically a 5.34 micron line detection. This will take some complicated `matplotlib` wrangling to pack the plots into a grid, but not much else of note.

```{code-cell} ipython3
# This cell will take up to a minute to run.

transition = 5.34*u.um  # Plotting 5.34 micron line

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
    for obs in lines_yso_jwst_obstable:
        #  If this observation contains a detected 5.34 micron slice from this YSO...
        if (obs['yso_index'] == temp_yso_index) & (transition in obs['detected_feii_lines']):
            cloud_uri = cloud_uri_map[obs['dataURL']]  # get cloud URI

            # Open the file into memory
            header, data = load_cloud_data(cloud_uri)

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

> Figure 5 alt text: Tens of images plotted in a grid, with an object name labeling each image. Most of the images exhibit obvious extended emission in a wide variety of complex morphologies.

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

# Drop observations without footprints
sofia_obstable = sofia_temp[np.array(['POLYGON' in str(region) for region in sofia_temp['s_region']])]
```

...and crossmatching these footprints to the SIMBAD YSOs:

```{code-cell} ipython3
# Create an iterable with pandas
sofia_obstable_iterable = sofia_obstable.to_pandas().iterrows()

# Apply the check_points_in_polygon function in parallel.
num_workers = num_cores
with Pool(num_workers) as pool:
    results = pool.map(check_points_in_polygon, [obs for _, obs in sofia_obstable_iterable])

# Convert back to a pandas dataframe
results = pd.DataFrame(results)

# Remove empty rows in the resulting table of observations,
# where check_points_in_polygon reached the "else" part of the return statement.
results = results.dropna(how='all')

# Convert back to an astropy table.
yso_sofia_obstable = Table.from_pandas(results)
```

```{code-cell} ipython3
# Let's take a look at the first observation
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

In the future, all-sky spectral cubes in IRSA from [the new SPHEREx mission](https://spherex.caltech.edu/page/data-products) might be more readily integrated into the line-search workflow we've demonstrated in this notebook, so stay tuned!

+++

## About this notebook

+++

- **Author**: Adrian Lucy (MAST/STScI, alucy@stsci.edu)
- **Contact:** For help with this notebook, please open a topic in the [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov/) "Support" category.

### Acknowledgements
- MAST internal review: Thomas Dutkiewicz, Sam Bianco, Zach Claytor, Brian McLean, Jonathan Hargis
- Fornax review: Troy Raen, Jessica Krick, Brigitta Sipőcz

### References

This notebook relies on the following papers:
- Assani et al. 2025 ([2025arXiv250402136A](https://ui.adsabs.harvard.edu/abs/2025arXiv250402136A/abstract))
- Karska et al. 2025 ([2025A&A...697A.186K](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.186K/abstract))

And the following software:
- Astroquery; Ginsburg et al. 2019 (2019AJ….157…98G)
- Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013 (2022ApJ…935..167A, 2018AJ….156..123A, 2013A&A…558A..33A)
- Matplotlib; Hunter 2007 (2007CSE.....9...90H)
- SciPy; Pauli et al. 2020 (2020NatMe..17..261V)
- NumPy; Harris et al. 2020 (doi:10.1038/s41586-020-2649-2)
- pandas; McKinney et al. 2010 (doi:10.25080/Majora-92bf1922-00a)
