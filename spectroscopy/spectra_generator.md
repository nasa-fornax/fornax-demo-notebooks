---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Extract Multi-Wavelength Spectroscopy from Archival Data
***

## Learning Goals    
By the end of this tutorial, you will be able to:

 &bull; automatically load a catalog of sources
 
 &bull; search NASA and non-NASA resources for fully reduced spectra and load them using specutils
 
 &bull; store the spectra in a Pandas multiindex dataframe
 
 &bull; plot all the spectra of a given source
 
 
## Introduction:

### Motivation
A user has a source (or a sample of sources) for which they want to obtain spectra covering ranges of wavelengths from the UV to the far-IR. The large amount of spectra available enables multi-wavelength spectroscopic studies, which is crucial to understand the physics of stars, galaxies, and AGN. However, gathering and analysing spectra is a difficult endeavor as the spectra are distributed over different archives and in addition they have different formats which complicates their handling. This notebook showcases a tool for the user to conveniently query the spectral archives and collect the spectra for a set of objects in a format that can be read in using common software such as the Python `specutils` package. For simplicity, we limit the tool to query already reduced and calibrated spectra. 
The notebook may focus on the COSMOS field for now, which has a large overlap of spectroscopic surveys such as with SDSS, DESI, Keck, HST, JWST, Spitzer, and Herschel. In addition, the tool enables the capability to search and ingest spectra from Euclid and SPHEREx in the feature. For this to work, the `specutils` functions may have to be update or a wrapper has to be implemented. 


### List of Spectroscopic Archives and Status


| Archive | Spectra | Description | Access point | Status |
| ------- | ------- | ----------- | ------------ | ------ |
| IRSA    | Keck    | About 10,000 spectra on the COSMOS field from [Hasinger et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...858...77H/abstract) | [IRSA Archive](https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-scan?projshort=COSMOS) | Implemented with `astroquery.ipac.irsa`. (Table gives URLs to spectrum FITS files.) Note: only implemented for absolute calibrated spectra. |
| IRSA    | Spitzer IRS | ~17,000 merged low-resolution IRS spectra | [IRS Enhanced Product](https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=irs_enhv211) | Implemented with `astroquery.ipac.irsa`. (Table gives URLs to spectrum IPAC tables.) |
| IRSA    | Herschel*    | Some spectra, need to check reduction stage | | |
| IRSA    | Euclid      | Spectra hosted at IRSA in FY25 -> preparation for ingestion | | Will use mock spectra with correct format for testing |
| IRSA    | SPHEREx     | Spectra/cubes will be hosted at IRSA, first release in FY25 -> preparation for ingestion | | Will use mock spectra with correct format for testing |
| MAST    | HST*         | Slitless spectra would need reduction and extraction. There are some reduced slit spectra from COS in the Hubble Archive | `astroquery.mast` | Implemented using `astroquery.mast` |
| MAST    | JWST*        | Reduced slit MSA and Slit spectra that can be queried | `astroquery.mast` | Implemented using `astroquery.mast` |
| SDSS    | SDSS optical| Optical spectra that are reduced | [Sky Server](https://skyserver.sdss.org/dr18/SearchTools) or `astroquery.sdss` (preferred) | Implemented using `astroquery.sdss`. |
| DESI    | DESI*        | Optical spectra | [DESI public data release](https://data.desi.lbl.gov/public/) | Implemented with `SPARCL` library |
| BOSS    | BOSS*        | Optical spectra | [BOSS webpage (part of SDSS)](https://www.sdss4.org/surveys/boss/) | Implemented with `SPARCL` library together with DESI |
| HEASARC | None        | Could link to Chandra observations to check AGN occurrence. | `astroquery.heasarc` | More thoughts on how to include scientifically.   |

The ones with an asterisk (*) are the challenging ones.

## Input:

 &bull; Coordinates for a single source or a sample on the COSMOS field
 


## Output:
 
 &bull; A Pandas data frame including the spectra from different facilities
 
 &bull; A plot comparing the different spectra extracted for each source
 
## Non-standard Imports:

&bull; ...

## Authors:
Andreas Faisst, Jessica Krick, Shoubaneh Hemmati, Troy Raen, Brigitta Sipőcz, Dave Shupe

## Acknowledgements:
...

## Open Issues:

&bull; Implement queries for: Herschel, Euclid (use mock data), SPHEREx (use mock data)
&bull; Match to HEASARC
&bull; Make more efficient (especially MAST searches)


<!-- #endregion -->

### Datasets that were considered but didn't end up being used:
#### IRTF: 
    - https://irsa.ipac.caltech.edu/Missions/irtf.html \
    - The IRTF is a 3.2 meter telescope, optimized for infrared observations, and located at the summit of Mauna Kea, Hawaiʻi. \
    - large library of stellar spectra \
    - Not included here because the data are not currently available in an easily accessible, searchable format
    

```python
# Ensure all dependencies are installed

!pip install -r requirements.txt
```

```python
## IMPORTS
import sys, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl


from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

sys.path.append('code_src/')
from data_structures_spec import MultiIndexDFObject
from sample_selection import clean_sample
from desi_functions import DESIBOSS_get_spec
from spitzer_functions import SpitzerIRS_get_spec
from sdss_functions import SDSS_get_spec
from mast_functions import HST_get_spec, JWST_get_spec
from keck_functions import KeckDEIMOS_get_spec
from plot_functions import create_figures
```

## 1. Define the sample

Here we will define the sample of galaxies. For now, we just enter some "random" coordinates to test the code.

```python
coords = []
labels = []

coords.append(SkyCoord("{} {}".format("09 54 49.40" , "+09 16 15.9"), unit=(u.hourangle, u.deg) ))
labels.append("NGC3049")

coords.append(SkyCoord("{} {}".format("12 45 17.44 " , "27 07 31.8"), unit=(u.hourangle, u.deg) ))
labels.append("NGC4670")

coords.append(SkyCoord("{} {}".format("14 01 19.92" , "−33 04 10.7"), unit=(u.hourangle, u.deg) ))
labels.append("Tol_89")

coords.append(SkyCoord(233.73856 , 23.50321, unit=u.deg ))
labels.append("Arp220")

coords.append(SkyCoord( 150.091 , 2.2745833, unit=u.deg ))
labels.append("COSMOS1")

coords.append(SkyCoord( 150.1024475 , 2.2815559, unit=u.deg ))
labels.append("COSMOS2")

coords.append(SkyCoord("{} {}".format("150.000" , "+2.00"), unit=(u.deg, u.deg) ))
labels.append("COSMOS3")

coords.append(SkyCoord("{} {}".format("+53.15508" , "-27.80178"), unit=(u.deg, u.deg) ))
labels.append("JADESGS-z7-01-QU")

coords.append(SkyCoord("{} {}".format("+53.15398", "-27.80095"), unit=(u.deg, u.deg) ))
labels.append("TestJWST")


sample_table = clean_sample(coords, labels, precision=2.0* u.arcsecond , verbose=1)

```

### 1.2 Write out your sample to disk

At this point you may wish to write out your sample to disk and reuse that in future work sessions, instead of creating it from scratch again. Note that we first check if the `data` directory exists and if not, we will create one.

For the format of the save file, we would suggest to choose from various formats that fully support astropy objects(eg., SkyCoord).  One example that works is Enhanced Character-Separated Values or ['ecsv'](https://docs.astropy.org/en/stable/io/ascii/ecsv.html)

```python
if not os.path.exists("./data"):
    os.mkdir("./data")
sample_table.write('data/input_sample.ecsv', format='ascii.ecsv', overwrite = True)
```

### 1.3 Load the sample table from disk

Do only this step from this section when you have a previously generated sample table

```python
sample_table = Table.read('data/input_sample.ecsv', format='ascii.ecsv')
```

### 1.4 Initialize data structure to hold the spectra
Here, we initialize the MultiIndex data structure that will hold the spectra.

```python
df_spec = MultiIndexDFObject()
```

## 2. Find spectra for these targets in NASA and other ancillary catalogs

We search a curated list of NASA astrophysics archives.  Because each archive is different, and in many cases each catalog is different, each function to access a catalog is necesarily specialized to the location and format of that particular catalog.


### 2.1 IRSA Archive

This archive includes spectra taken by 

 &bull; Keck
 
 &bull; Spitzer/IRS
 
 &bull; Herschel (not implemented, yet)


```python
%%time
## Get Keck Spectra (COSMOS only)
df_spec_DEIMOS = KeckDEIMOS_get_spec(sample_table = sample_table, search_radius_arcsec=1)
df_spec.append(df_spec_DEIMOS)
```

```python
%%time
## Get Spitzer IRS Spectra
df_spec_IRS = SpitzerIRS_get_spec(sample_table, search_radius_arcsec=1 , COMBINESPEC=False)
df_spec.append(df_spec_IRS)
```

### 2.2 MAST Archive

This archive includes spectra taken by 

 &bull; HST (including slit spectroscopy)
 
 &bull; JWST (including MSA and slit spectroscopy)


```python
%%time
## Get Spectra for HST
df_spec_HST = HST_get_spec(sample_table , search_radius_arcsec = 0.5, datadir = "./data/", verbose = False)
df_spec.append(df_spec_HST)
```

```python
%%time
## Get Spectra for JWST
df_jwst = JWST_get_spec(sample_table , search_radius_arcsec = 0.5, datadir = "./data/", verbose = False)
df_spec.append(df_jwst)
```

### 2.3 SDSS Archive

This includes SDSS spectra.

```python
%%time
## Get SDSS Spectra
df_spec_SDSS = SDSS_get_spec(sample_table , search_radius_arcsec=5, data_release=17)
df_spec.append(df_spec_SDSS)
```

### 2.4 DESI Archive

This includes DESI spectra. Here, we use the `SPARCL` query. Note that this can also be used
for SDSS searches, however, according to the SPARCL webpage, only up to DR16 is included. Therefore, we will not include SDSS DR16 here (this is treated in the SDSS search above).

```python
%%time
## Get DESI and BOSS spectra with SPARCL
df_spec_DESIBOSS = DESIBOSS_get_spec(sample_table, search_radius_arcsec=5)
df_spec.append(df_spec_DESIBOSS)
```

## 3. Make plots of luminosity as a function of time
We show flux in mJy as a function of time for all available bands for each object. `show_nbr_figures` controls how many plots are actually generated and returned to the screen.  If you choose to save the plots with `save_output`, they will be put in the output directory and labelled by sample number.



```python
### Plotting ####
create_figures(df_spec = df_spec,
             bin_factor=5,
             show_nbr_figures = 10,
             save_output = False,
             )
```

<!-- #raw -->

<!-- #endraw -->
