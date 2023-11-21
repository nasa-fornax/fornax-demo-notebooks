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

# Make multiwavelength light curves using archival data
***

## Learning Goals    
By the end of this tutorial, you will be able to:
 - automatically load a catalog of sources
 - automatically search NASA and non-NASA resources for light curves
 - store light curves in a Pandas multiindex dataframe
 - plot all light curves on the same plot
 
 
## Introduction:
 - A user has a sample of interesting targets for which they would like to see a plot of available archival light curves.  We start with a small set of changing look AGN from Yang et al., 2018, which are automatically downloaded. Changing look AGN are cases where the broad emission lines appear or disappear (and not just that the flux is variable). 
 - We model light curve plots after van Velzen et al. 2021.  We search through a curated list of time-domain NASA holdings as well as non-NASA sources.  HEASARC catalogs used are Fermi and Beppo-Sax, IRSA catalogs used are ZTF and WISE, and MAST catalogs used are Pan-Starrs, TESS, Kepler, and K2.  Non-NASA sources are Gaia and IceCube. This list is generalized enough to include many types of targets to make this notebook interesting for many types of science.  All of these time-domain archives are searched in an automated fashion using astroquery or APIs.
 - Light curve data storage is a tricky problem.  Currently we are using a multi-index Pandas dataframe, as the best existing choice for right now.  One downside is that we need to manually track the units of flux and time instead of relying on an astropy storage scheme which would be able to do some of the units worrying for us (even astropy can't do all magnitude to flux conversions).  Astropy does not currently have a good option for multi-band light curve storage.
 - We intend to explore a ML classifier for these changing look AGN light curves.
 
## Input:
 - choose from a list of known changing look AGN from the literature
 
  OR - 
 - input your own sample

## Output:
 - an archival optical + IR + neutrino light curve
 
## Non-standard Imports:
- `acstools` to work with HST magnitude to flux conversion
- `astropy` to work with coordinates/units and data structures
- `astroquery` to interface with archives APIs
- `hpgeom` to locate coordinates in HEALPix space
- `lightkurve` to search TESSS, Kepler, and K2 archives
- `pyarrow` to work with Parquet files for WISE and ZTF
- `s3fs` to connect to AWS S3 buckets
- `urllib` to handle archive searches with website interface

## Authors:
Jessica Krick, Shoubaneh Hemmati, Andreas Faisst, Troy Raen, Brigitta Sipőcz, Dave Shupe

## Acknowledgements:
Suvi Gezari, Antara Basu-zych,Stephanie LaMassa\
MAST, HEASARC, & IRSA Fornax teams

```{code-cell} ipython3
#ensure all dependencies are installed
!pip install -r requirements.txt
```

```{code-cell} ipython3
import multiprocessing as mp
import sys
import time
import warnings

import astropy.units as u
import pandas as pd
from astropy.table import Table

warnings.filterwarnings('ignore')

# local code imports
sys.path.append('code_src/')
from data_structures import MultiIndexDFObject
from gaia_functions import Gaia_get_lightcurve
from HCV_functions import HCV_get_lightcurves
from heasarc_functions import HEASARC_get_lightcurves
from icecube_functions import Icecube_get_lightcurve
from panstarrs import Panstarrs_get_lightcurves
from plot_functions import create_figures
from sample_selection import (clean_sample, get_green_sample, get_hon_sample, get_lamassa_sample, get_lopeznavas_sample,
    get_lyu_sample, get_macleod16_sample, get_macleod19_sample, get_ruan_sample, get_SDSS_sample, get_sheng_sample, get_yang_sample)
from TESS_Kepler_functions import TESS_Kepler_get_lightcurves
# Note: WISE and ZTF data are temporarily located in a non-public AWS S3 bucket. It is automatically
# available from the Fornax SMCE, but will require user credentials for access outside the SMCE.
from WISE_functions import WISE_get_lightcurves
from ztf_functions import ZTF_get_lightcurve
```

## 1. Define the Sample
 We define here a "gold" sample of spectroscopically confirmed changing look AGN and quasars. This sample includes both objects which change from type 1 to type 2 and also the opposite.  Future studies may want to treat these as seperate objects or seperate QSOs from AGN.
 
 Bibcodes for the samples used are listed next to their functions for reference.  
 
 Functions used to grab the samples from the papers use Astroquery, NED, SIMBAD, Vizier, and in a few cases grab the tables from the html versions of the paper.

```{code-cell} ipython3
#build up the sample
coords =[]
labels = []

#choose your own adventure:

#get_lamassa_sample(coords, labels)  #2015ApJ...800..144L
#get_macleod16_sample(coords, labels) #2016MNRAS.457..389M
#get_ruan_sample(coords, labels) #2016ApJ...826..188R
#get_macleod19_sample(coords, labels)  #2019ApJ...874....8M
#get_sheng_sample(coords, labels)  #2020ApJ...889...46S
#get_green_sample(coords, labels)  #2022ApJ...933..180G
#get_lyu_sample(coords, labels)  #z32022ApJ...927..227L
#get_lopeznavas_sample(coords, labels)  #2022MNRAS.513L..57L
#get_hon_sample(coords, labels)  #2022MNRAS.511...54H
get_yang_sample(coords, labels)   #2018ApJ...862..109Y

#now get some "normal" QSOs for use in the classifier
#there are ~500K of these, so choose the number based on
#a balance between speed of running the light curves and whatever 
#the ML algorithms would like to have

#num_normal_QSO = 5000
#get_SDSS_sample(coords, labels, num_normal_QSO)

# remove duplicates and attach an objectid to the coords
sample_table = clean_sample(coords, labels)
```

### 1.1 Build your own Sample

To build your own sample, you can follow the examples of functions above to grab coordinates from your favorite literature resource, 

or

You can use [astropy's read](https://docs.astropy.org/en/stable/io/ascii/read.html) function to read in an input table
and then convert that table into a list of [skycoords](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html)

+++

### 1.2 Write out your sample to disk

At this point you may wish to write out your sample to disk and reuse that in future work sessions, instead of creating it from scratch again.

We would suggest to choose from various formats that fully supports the astropy objects, such as SkyCoord, in the so-called Mixin columns. E.g  Enhanced Character-Separated Values or 'ecsv' is one such format: https://docs.astropy.org/en/stable/io/ascii/ecsv.html

```{code-cell} ipython3
sample_table.write('data/input_sample.ecsv', format='ascii.ecsv', overwrite = True)
```

### 1.3 Load the sample table from disk

Do only this step from this section when you have a previously generated sample table

```{code-cell} ipython3
sample_table = Table.read('data/input_sample.ecsv', format='ascii.ecsv')
```

## 2. Find light curves for these targets in NASA catalogs
  - We search a curated list of time-domain catalogs from all NASA astrophysics archives

```{code-cell} ipython3
### Initialize Pandas MultiIndex data frame for storing the light curves
df_lc = MultiIndexDFObject()
```

### 2.1 HEASARC: FERMI & Beppo SAX

```{code-cell} ipython3
start_serial = time.time()

#what is the size of error_radius for the fermi catalog that we will accept for our cross-matching?
#in degrees; chosen based on histogram of all values for these catalogs
max_fermi_error_radius = str(1.0)  
max_sax_error_radius = str(3.0)

#list of missions to query and their corresponding error radii
heasarc_cat = ["FERMIGTRIG", "SAXGRBMGRB"]
error_radius = [max_fermi_error_radius , max_sax_error_radius]


#go out and find all light curves in the above curated list which match our target positions
df_lc_fermi = HEASARC_get_lightcurves(sample_table, heasarc_cat, error_radius)
df_lc.append(df_lc_fermi)
    
```

### 2.2 IRSA: ZTF

```{code-cell} ipython3
# use the nworkers arg to control the amount of parallelization in the data loading step
df_lc_ZTF = ZTF_get_lightcurve(sample_table, nworkers=6)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_ZTF)
```

### 2.3 IRSA: WISE

- use the unWISE light curves catalog which ties together all WISE & NEOWISE 2010 - 2020 epochs.  Specifically it combined all observations at a single epoch to achieve deeper mag limits than individual observations alone.
- [Meisner et al., 2023, 2023AJ....165...36M](https://ui.adsabs.harvard.edu/abs/2023AJ....165...36M/abstract)

```{code-cell} ipython3
bandlist = ['W1', 'W2']
WISE_radius = 1.0 * u.arcsec

df_lc_WISE = WISE_get_lightcurves(sample_table, WISE_radius, bandlist)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_WISE)
```

### 2.4 MAST: Pan-STARRS
Query the Pan-STARRS API; based on this [example](https://ps1images.stsci.edu/ps1_dr2_api.html)

```{code-cell} ipython3
#Do a panstarrs search
panstarrs_radius = 1.0/3600.0    # search radius = 1 arcsec
df_lc_panstarrs = Panstarrs_get_lightcurves(sample_table, panstarrs_radius)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_panstarrs)
```

### 2.5 MAST: Asteroid Terrestrial-impact Last Alert System (ATLAS)
 - All-sky stellar reference catalog 
 -  MAST hosts this catalog but there are three barriers to using it
     1. it is unclear if the MAST [holdings]( https://archive.stsci.edu/hlsp/atlas-refcat2#section-a737bc3e-2d56-4827-9ab4-838fbf8d67c1) include the individual epoch photometry and 
     2. it is only accessible with casjobs, not through python notebooks.  
     3. magnitude range (g, r, i) < 19mag makes it not relevant for this use case
 
One path forward if this catalog becomes scientifically interesting is to put in a MAST helpdesk ticket to see if 1) they do have the light curves, and 2) they could switch the catalog to a searchable with python version.  There are some ways of [accessing casjobs with python](<https://github.com/spacetelescope/notebooks/blob/master/notebooks/MAST/HSC/HCV_CASJOBS/HCV_casjobs_demo.ipynb), but not this particular catalog.

+++

### 2.6 MAST: TESS, Kepler and K2
 - use [`lightKurve`](https://docs.lightkurve.org/index.html) to search all 3 missions and download light curves

```{code-cell} ipython3
#go get the lightcurves using lightkurve
TESS_radius = 1.0  #arcseconds
df_lc_TESS = TESS_Kepler_get_lightcurves(sample_table, TESS_radius)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_TESS)

#LightKurve will return an "Error" when it doesn't find a match for a target
#These are not real errors and can be safely ignored.
```

### 2.7 MAST: HCV
 - [hubble catalog of variables](https://archive.stsci.edu/hlsp/hcv) 
 - using [this notebook](https://archive.stsci.edu/hst/hsc/help/HCV/HCV_API_demo.html) as a reference to search and download light curves via API

```{code-cell} ipython3
#Do an HCV search
HCV_radius = 1.0/3600.0 # radius = 1 arcsec
df_lc_HCV = HCV_get_lightcurves(sample_table, HCV_radius)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_HCV)
```

## 3. Find light curves for these targets in relevant, non-NASA catalogs

+++

### 3.1 Gaia

```{code-cell} ipython3
gaiastarttime = time.time()
df_lc_gaia = Gaia_get_lightcurve(sample_table, 1/3600., 0)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_gaia)

print('gaia search took:', time.time() - gaiastarttime, 's')
```

### 3.2 ASAS-SN (all sky automated survey for supernovae) 
- Has a [website](https://asas-sn.osu.edu/photometry) that can be manually searched; but no API which would allow automatic searches from within this notebook
- Magnitude range of this survey is not consistent with the magnitude range of our CLAGN.  If this catalog becomes scientifically interesting, one path forward would be to ask ASAS-SN team about implementing an API

+++

### 3.3 Icecube Neutrinos

There are several [catalogs](https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018) (basically one for each year of IceCube data from 2008 - 2018). The following code creates a large catalog by combining
all the yearly catalogs.
The IceCube catalog contains Neutrino detections with associated energy and time and approximate direction (which is uncertain by half-degree scales....). Usually, for active events only one or two Neutrinos are detected, which makes matching quite different compared to "photons". For our purpose, we will list the top 3 events in energy that are within a given distance to the target.

This time series (time vs. neutrino energy) information is similar to photometry. We choose to storing time and energy in our data structure, leaving error = 0. What is __not__ stored in this format is the distance or angular uncertainty of the event direction.

```{code-cell} ipython3
df_lc_icecube = Icecube_get_lightcurve(sample_table ,
                                   icecube_select_topN = 3)

#add the resulting dataframe to all other archives
df_lc.append(df_lc_icecube)
end_serial = time.time()
```

```{code-cell} ipython3
#benchmarking
print('total time for serial archive calls is ', end_serial - start_serial, 's')
```

## 4. Parallel Processing the archive calls

```{code-cell} ipython3
# define some variables in case the above serial cells are not run
max_fermi_error_radius = str(1.0)  
max_sax_error_radius = str(3.0)
heasarc_cat = ["FERMIGTRIG", "SAXGRBMGRB"]
error_radius = [max_fermi_error_radius , max_sax_error_radius]
bandlist = ["W1", "W2"]
wise_radius = 1.0 * u.arcsec
panstarrs_radius = 1.0 / 3600.0  # search radius = 1 arcsec
lk_radius = 1.0  # arcseconds
hcv_radius = 1.0 / 3600.0  # radius = 1 arcsec
```

```{code-cell} ipython3
# number of workers to use in the parallel processing pool
# this should equal the total number of archives called
n_workers = 8

# "spawn" new processes because it uses less memory and is thread safe
# in particular, this is required for pd.read_parquet (used by ZTF_get_lightcurve)
# https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
mp.set_start_method("spawn", force=True)
```

```{code-cell} ipython3
# the ZTF call can be parallelized internally, separate from the pool launched below.
# these parallelizations are mutually exclusive, so we must turn off the internal parallelization.
ztf_nworkers = None

# note that the ZTF call is relatively slow compared to other archives.
# if you want to query for a large number of objects, it will be faster to call ZTF individually
# (code above) and use the internal parallelization. try 8-12 workers.
```

```{code-cell} ipython3
parallel_starttime = time.time()

# start a multiprocessing pool and run all the archive queries
parallel_df_lc = MultiIndexDFObject()  # to collect the results
callback = parallel_df_lc.append  # will be called once on the result returned by each archive
with mp.Pool(processes=n_workers) as pool:

    # start the processes that call the archives
    pool.apply_async(
        Gaia_get_lightcurve, (sample_table, 1/3600., 0), callback=callback
    )
    pool.apply_async(
        HEASARC_get_lightcurves, (sample_table, heasarc_cat, error_radius), callback=callback
    )
    pool.apply_async(
        HCV_get_lightcurves, (sample_table, hcv_radius), callback=callback
    )
    pool.apply_async(
        Icecube_get_lightcurve, (sample_table , 3), callback=callback
    )
    pool.apply_async(
        Panstarrs_get_lightcurves, (sample_table, panstarrs_radius), callback=callback
    )
    pool.apply_async(
        TESS_Kepler_get_lightcurves, (sample_table, lk_radius), callback=callback
    )
    pool.apply_async(
        WISE_get_lightcurves, (sample_table,  wise_radius, bandlist), callback=callback
    )
    pool.apply_async(
        ZTF_get_lightcurve, (sample_table, ztf_nworkers), callback=callback
    )

    pool.close()  # signal that no more jobs will be submitted to the pool
    pool.join()  # wait for all jobs to complete, including the callback

parallel_endtime = time.time()
```

```{code-cell} ipython3
# How long did parallel processing take?
# and look at the results
print('parallel processing took', parallel_endtime - parallel_starttime, 's')
parallel_df_lc.data
```

```{code-cell} ipython3
# Save the data for future use with ML notebook
#parquet_savename = 'output/df_lc_090723_yang.parquet'
#parallel_df_lc.data.to_parquet(parquet_savename)
#print("file saved!")
```

```{code-cell} ipython3
# could load a previously saved file in order to plot
#parquet_loadname = 'output/df_lc_090723_yang.parquet'
#parallel_df_lc = MultiIndexDFObject()
#parallel_df_lc.data = pd.read_parquet(parquet_loadname)
#print("file loaded!")
```

## 5. Make plots of luminosity as a function of time
Model plots after [van Velzen et al., 2021](https://arxiv.org/pdf/2111.09391.pdf).

__Note__ that in the following, we can either plot the results from `df_lc` (from the serial call) or `parallel_df_lc` (from the parallel call). By default (see next cell) the output of the parallel call is used.

```{code-cell} ipython3
_ = create_figures(sample_table ,
                   df_lc = parallel_df_lc, # either df_lc (serial call) or parallel_df_lc (parallel call)
                   show_nbr_figures = 5,
                   save_output = True ,
                  )
```

## References

This work made use of:

- Astroquery; Ginsburg et al., 2019, 2019AJ....157...98G

- Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013, 2022ApJ...935..167A, 2018AJ....156..123A, 2013A&A...558A..33A

- Lightkurve; Lightkurve Collaboration 2018, 2018ascl.soft12013L

- acstools; https://zenodo.org/record/7406933#.ZBH1HS-B0eY

- unWISE light curves; Meisner et al., 2023, 2023AJ....165...36M

- Alerce; Forster et al., 2021, 2021AJ....161..242F
