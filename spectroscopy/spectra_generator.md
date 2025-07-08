---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: py-spectra_generator
  display_name: py-spectra_generator
  language: python
---

# Extract Multi-Wavelength Spectroscopy from Archival Data


## Learning Goals
By the end of this tutorial, you will be able to:

 &bull; automatically load a catalog of sources

 &bull; search NASA and non-NASA resources for fully reduced spectra and load them using specutils

 &bull; store the spectra in a Pandas multiindex dataframe

 &bull; plot all the spectra of a given source


## Introduction:

### Motivation
A user has a source (or a sample of sources) for which they want to obtain spectra covering ranges
of wavelengths from the UV to the far-IR. The large amount of spectra available enables
multi-wavelength spectroscopic studies, which is crucial to understand the physics of stars,
galaxies, and AGN. However, gathering and analysing spectra is a difficult endeavor as the spectra
are distributed over different archives and in addition they have different formats which
complicates their handling. This notebook showcases a tool for the user to conveniently query the
spectral archives and collect the spectra for a set of objects in a format that can be read in
using common software such as the Python `specutils` package. For simplicity, we limit the tool to
query already reduced and calibrated spectra.
The notebook may focus on the COSMOS field for now, which has a large overlap of spectroscopic
surveys such as with SDSS, DESI, Keck, HST, JWST, Spitzer, and Herschel. In addition, the tool
enables the capability to search and ingest spectra from Euclid and SPHEREx in the feature. For
this to work, the `specutils` functions may have to be update or a wrapper has to be implemented.


### List of Spectroscopic Archives and Status


| Archive | Spectra | Description | Access point | Status |
| ------- | ------- | ----------- | ------------ | ------ |
| IRSA    | Keck    | About 10,000 spectra on the COSMOS field from [Hasinger et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...858...77H/abstract) | [IRSA Archive](https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-scan?projshort=COSMOS) | Implemented with `astroquery.ipac.irsa`. (Table gives URLs to spectrum FITS files.) Note: only implemented for absolute calibrated spectra. |
| IRSA    | Spitzer IRS | ~17,000 merged low-resolution IRS spectra | [IRS Enhanced Product](https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=irs_enhv211) | Implemented with `astroquery.ipac.irsa`. (Table gives URLs to spectrum IPAC tables.) |
| IRSA    | IRTF*        | Large library of stellar spectra | | does `astroquery.ipac.irsa` work?? |
| ESA    | Herschel*    | Some spectra | astroquery.esa.hsa | implemented with [astroquery](https://astroquery.readthedocs.io/en/latest/esa/hsa/hsa.html) |
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
## Runtime

As of 2024 December, this notebook takes about 17 minutes to run to completion on Fornax using
Server Type: 'Standard - 8GB RAM/4 CPU' and Environment: 'Default Astrophysics' (image).

## Authors:
Andreas Faisst, Jessica Krick, Shoubaneh Hemmati, Troy Raen, Brigitta Sipőcz, David Shupe

## Acknowledgements:

AI: This notebook was created with assistance from OpenAI’s ChatGPT o4-mini-high model.

+++

### Datasets that were considered but didn't end up being used:
#### IRTF:
- https://irsa.ipac.caltech.edu/Missions/irtf.html
- The IRTF is a 3.2 meter telescope, optimized for infrared observations, and located at the summit
  of Mauna Kea, Hawaiʻi.
- large library of stellar spectra
- Not included here because the data are not currently available in an easily accessible,
  searchable format


## Imports

This cell will install them if needed:

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# !pip install -r requirements_spectra_generator.txt
# !pip install --upgrade --pre astroquery  # >=0.4.8.dev9474 needed for mast_functions
```

```{code-cell} ipython3
%pip install -U --pre astroquery[all]
```

```{code-cell} ipython3
%pip install fsspec boto3 
```

```{code-cell} ipython3
%pip install s3fs
```

```{code-cell} ipython3
import os
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

sys.path.append('code_src/')
from data_structures_spec import MultiIndexDFObject
#from desi_functions import DESIBOSS_get_spec
from herschel_functions import Herschel_get_spec
from keck_functions import KeckDEIMOS_get_spec
#from mast_functions import HST_get_spec, JWST_get_spec
from plot_functions import create_figures
from sample_selection import clean_sample
from sdss_functions import SDSS_get_spec
from spitzer_functions import SpitzerIRS_get_spec
```

## 1. Define the sample

Here we will define the sample of galaxies. For now, we just enter some "random" coordinates to
test the code.

```{code-cell} ipython3
coords = []
labels = []

coords.append(SkyCoord("{} {}".format("09 54 49.40", "+09 16 15.9"), unit=(u.hourangle, u.deg)))
labels.append("NGC3049")

coords.append(SkyCoord("{} {}".format("12 45 17.44 ", "27 07 31.8"), unit=(u.hourangle, u.deg)))
labels.append("NGC4670")

coords.append(SkyCoord("{} {}".format("14 01 19.92", "−33 04 10.7"), unit=(u.hourangle, u.deg)))
labels.append("Tol_89")

coords.append(SkyCoord(233.73856, 23.50321, unit=u.deg))
labels.append("Arp220")

coords.append(SkyCoord(150.091, 2.2745833, unit=u.deg))
labels.append("COSMOS1")

coords.append(SkyCoord(150.1024475, 2.2815559, unit=u.deg))
labels.append("COSMOS2")

#coords.append(SkyCoord("{} {}".format("150.000", "+2.00"), unit=(u.deg, u.deg)))
#labels.append("COSMOS3")

#coords.append(SkyCoord("{} {}".format("+53.15508", "-27.80178"), unit=(u.deg, u.deg)))
#labels.append("JADESGS-z7-01-QU")

#coords.append(SkyCoord("{} {}".format("+53.15398", "-27.80095"), unit=(u.deg, u.deg)))
#labels.append("TestJWST")

#coords.append(SkyCoord("{} {}".format("+150.33622", "+55.89878"), unit=(u.deg, u.deg)))
#labels.append("Twin Quasar")

sample_table = clean_sample(coords, labels, precision=2.0 * u.arcsecond, verbose=1)
```

### 1.2 Write out your sample to disk

At this point you may wish to write out your sample to disk and reuse that in future work sessions,
instead of creating it from scratch again. Note that we first check if the `data` directory exists
and if not, we will create one.

For the format of the save file, we would suggest to choose from various formats that fully support
astropy objects(eg., SkyCoord).  One example that works is Enhanced Character-Separated Values or
['ecsv'](https://docs.astropy.org/en/stable/io/ascii/ecsv.html)

```{code-cell} ipython3
if not os.path.exists("./data"):
    os.mkdir("./data")
sample_table.write('data/input_sample.ecsv', format='ascii.ecsv', overwrite=True)
```

### 1.3 Load the sample table from disk

Do only this step from this section when you have a previously generated sample table

```{code-cell} ipython3
sample_table = Table.read('data/input_sample.ecsv', format='ascii.ecsv')
```

### 1.4 Initialize data structure to hold the spectra
Here, we initialize the MultiIndex data structure that will hold the spectra.

```{code-cell} ipython3
df_spec = MultiIndexDFObject()
```

## 2. Find spectra for these targets in NASA and other ancillary catalogs

We search a curated list of NASA astrophysics archives.  Because each archive is different, and in
many cases each catalog is different, each function to access a catalog is necesarily specialized
to the location and format of that particular catalog.

+++

### 2.1 IRSA Archive

This archive includes spectra taken by

 &bull; Keck

 &bull; Spitzer/IRS

```{code-cell} ipython3
%%time
# Get Keck Spectra (COSMOS only)
df_spec_DEIMOS = KeckDEIMOS_get_spec(sample_table=sample_table, search_radius_arcsec=1)
df_spec.append(df_spec_DEIMOS)
```

```{code-cell} ipython3
%%time
# Get Spitzer IRS Spectra
df_spec_IRS = SpitzerIRS_get_spec(sample_table, search_radius_arcsec=1, COMBINESPEC=False)
df_spec.append(df_spec_IRS)
```

### 2.2 MAST Archive

This archive includes spectra taken by

 &bull; HST (including slit spectroscopy)

 &bull; JWST (including MSA and slit spectroscopy)

```{code-cell} ipython3
%%time
# Get Spectra for HST
df_spec_HST = HST_get_spec(
    sample_table,
    search_radius_arcsec=0.5,
    datadir="./data/",
    verbose=False,
    delete_downloaded_data=True
)
df_spec.append(df_spec_HST)
```

```{code-cell} ipython3
import os
import shutil
import warnings
import fsspec

import astropy.constants as const
import astropy.units as u
from astropy.io import fits
import astroquery.exceptions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.table import vstack
from astroquery.mast import Observations
from specutils import Spectrum1D

from data_structures_spec import MultiIndexDFObject
            

def JWST_get_spec(sample_table, search_radius_arcsec, verbose):
    """
    Retrieve JWST spectra for a list of sources and groups/stacks them.
    This main function runs two sub-functions:
    - `JWST_get_spec_helper()` which searches and retrieves the spectra from
      the cloud.
    - `JWST_group_spectra()` which groups and stacks the spectra.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    verbose : bool
        Verbosity level. Set to True for extra talking.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Get the spectra
    print("Searching Spectra in the cloud... ")
    df_jwst_all = JWST_get_spec_helper(
        sample_table, search_radius_arcsec, verbose)
    print("done")

    # Group
    print("Grouping Spectra... ")
    df_jwst_group = JWST_group_spectra(df_jwst_all, verbose=verbose, quickplot=False)
    print("done")

    return df_jwst_group


def JWST_get_spec_helper(sample_table, search_radius_arcsec, verbose):
    """
    Retrieve JWST spectra for a list of sources.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    verbose : bool
        Verbosity level. Set to True for extra talking.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Enable cloud data access for MAST once
    Observations.enable_cloud_dataset()


    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        print("Processing source {}".format(stab["label"]))

        # Query results
        search_coords = stab["coord"]
        # If no results are found, this will raise a warning. We explicitly handle the no-results
        # case below, so let's suppress the warning to avoid confusing notebook users.
        warnings.filterwarnings("ignore", message='Query returned no results.',
                                category=astroquery.exceptions.NoResultsWarning,
                                module="astroquery.mast.discovery_portal")
        query_results = Observations.query_criteria(
            coordinates=search_coords, radius=search_radius_arcsec * u.arcsec,
            dataproduct_type=["spectrum"], obs_collection=["JWST"], intentType="science",
            calib_level=[2, 3, 4], instrument_name=['NIRSPEC/MSA', 'NIRSPEC/SLIT'],
            dataRights=['PUBLIC'])
        print("Number of search results: {}".format(len(query_results)))

        if len(query_results) == 0:
            print("Source {} could not be found".format(stab["label"]))
            continue

        # Retrieve spectra
        data_products = [Observations.get_product_list(obs) for obs in query_results]
        data_products_list = vstack(data_products)
        
        # Filter
        data_products_list_filter = Observations.filter_products(
            data_products_list, productType=["SCIENCE"], extension="fits",
            calib_level=[2, 3, 4],  # only calibrated data
            productSubGroupDescription=["X1D"],  # only 1D spectra
            dataRights=['PUBLIC'])  # only public data
        print("Number of files found: {}".format(len(data_products_list_filter)))

        if len(data_products_list_filter) == 0:
            print("No spectra found for source {}.".format(stab["label"]))
            continue

        # Get cloud access URIs (returns a list of strings)
        cloud_uris_map = Observations.get_cloud_uris(data_products_list_filter, return_uri_map=True)

        # Create table with metadata from data_priducts_list_filter and cloud URIs
        keys = ["filters", "obs_collection", "instrument_name", "calib_level",
                "t_obs_release", "proposal_id", "obsid", "objID", "distance"]
        tab = Table(names=keys + ["productFilename", "clouduri"],
                    dtype=[str, str, str, int, float, int, int, int, float, str, str])

        for row in data_products_list_filter:
            filename = str(row["productFilename"])
            lookup_key = f"mast:JWST/product/{filename}"
            uri = cloud_uris_map.get(lookup_key)
            if uri is None:
                print(f"Skipping {filename}: not available in cloud.")
                continue

            obsid = row["parent_obsid"]
            idx_cross = np.where(query_results["obsid"] == obsid)[0]

            tmp = query_results[idx_cross][keys]
            tab.add_row(list(tmp[0]) + [filename, uri])

            
        # Create multi-index object
        for jj in range(len(tab)):

            # open spectrum directly from the cloud
            filepath = tab["clouduri"][jj]
            
            with fsspec.open(filepath, mode='rb', anon=True) as f:  # Open the file from S3
                with fits.open(f) as hdul:  # Open the FITS file
                    spec1d = Table(hdul[1].data)
                    columns = hdul[1].columns
                    print("flux: ", spec1d["FLUX"].data)
                    # Explicitly extract units
                    wave_unit = u.Unit(getattr(columns["WAVELENGTH"], 'unit', None))
                    flux_unit = u.Unit(getattr(columns["FLUX"], 'unit', None))
                    err_unit = u.Unit(getattr(columns["FLUX_ERROR"], 'unit', None))

            dfsingle = pd.DataFrame(dict(
                wave=[spec1d["WAVELENGTH"].data * wave_unit],
                flux=[spec1d["FLUX"].data * flux_unit],
                err=[spec1d["FLUX_ERROR"].data * err_unit],
                label=[stab["label"]],
                objectid=[stab["objectid"]],
                mission=[tab["obs_collection"][jj]],
                instrument=[tab["instrument_name"][jj]],
                filter=[tab["filters"][jj]],
            )).set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)


    return df_spec


def JWST_group_spectra(df, verbose, quickplot):
    """
    Group the JWST spectra and removes entries that have no spectra.
    Stack spectra that are similar and create a new DataFrame.

    Parameters
    ----------
    df : MultiIndexDFObject
        Raw JWST multi-index object (output from JWST_get_spec()).
    verbose : bool
        Flag for verbosity: True or False.
    quickplot : bool
        If True, quick plots are made for each spectral group.

    Returns
    -------
    MultiIndexDFObject
        Consolidated and grouped data structure storing the spectra.
    """

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    # Create data table from DF.
    tab = df.data.reset_index()

    # Get objects
    objects_unique = np.unique(tab["label"])

    for obj in objects_unique:
        print("Grouping object {}".format(obj))

        # Get filters
        filters_unique = np.unique(tab["filter"])
        if verbose:
            print("List of filters in data frame: {}".format(" | ".join(filters_unique)))

        for filt in filters_unique:
            if verbose:
                print("Processing {}: ".format(filt), end=" ")

            sel = np.where((tab["filter"] == filt) & (tab["label"] == obj))[0]
            tab_sel = tab.iloc[sel]
            if verbose:
                print("Number of items: {}".format(len(sel)), end=" | ")

            # get good ones
            sum_flux = np.asarray(
                [np.nansum(tab_sel.iloc[iii]["flux"]).value for iii in range(len(tab_sel))])
            idx_good = np.where(sum_flux > 0)[0]
            if verbose:
                print("Number of good spectra: {}".format(len(idx_good)))

            if len(idx_good) == 0:
                continue

            # Create wavelength grid
            wave_grid = tab_sel.iloc[idx_good[0]]["wave"]  # NEED TO BE MADE BETTER LATER

            # Interpolate fluxes
            fluxes_int = np.asarray(
                [np.interp(wave_grid, tab_sel.iloc[idx]["wave"], tab_sel.iloc[idx]["flux"]) for idx in idx_good])
            fluxes_units = [tab_sel.iloc[idx]["flux"].unit for idx in idx_good]

            # Sometimes fluxes are all NaN. We'll leave these in and ignore the RuntimeWarning.
            warnings.filterwarnings("ignore", message='All-NaN slice encountered', category=RuntimeWarning)
            fluxes_stack = np.nanmedian(fluxes_int, axis=0)
            if verbose:
                print("Units of fluxes for each spectrum: {}".format(
                    ",".join([str(tt) for tt in fluxes_units])))

            # Unit conversion to erg/s/cm2/A
            # (note fluxes are nominally in Jy. So have to do the step with dividing by lam^2)
            fluxes_stack_cgs = (fluxes_stack * fluxes_units[0]).to(u.erg / u.second / (
                u.centimeter**2) / u.hertz) * (const.c.to(u.angstrom/u.second)) / (wave_grid.to(u.angstrom)**2)
            fluxes_stack_cgs = fluxes_stack_cgs.to(
                u.erg / u.second / (u.centimeter**2) / u.angstrom)

            # Add to data frame
            dfsingle = pd.DataFrame(dict(
                wave=[wave_grid.to(u.micrometer)], flux=[fluxes_stack_cgs],
                err=[np.repeat(0, len(fluxes_stack_cgs))], label=[tab_sel["label"].iloc[0]],
                objectid=[tab_sel["objectid"].iloc[0]], mission=[tab_sel["mission"].iloc[0]],
                instrument=[tab_sel["instrument"].iloc[0]], filter=[tab_sel["filter"].iloc[0]]))
            dfsingle = dfsingle.set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)

            # Quick plot
            if quickplot:
                tmp = np.percentile(fluxes_stack, q=(1, 50, 99))
                plt.plot(wave_grid, fluxes_stack)
                plt.ylim(tmp[0], tmp[2])
                plt.xlabel(r"Observed Wavelength [$\mu$m]")
                plt.ylabel(r"Flux [Jy]")
                plt.show()

    return df_spec
```

```{code-cell} ipython3
%%time
# Get Spectra for JWST
df_jwst = JWST_get_spec(
    sample_table,
    search_radius_arcsec=0.5,
    verbose=False
)
df_spec.append(df_jwst)
```

```{code-cell} ipython3
df_jwst.data
```

```{code-cell} ipython3

```

### 2.3 ESA Archive

```{code-cell} ipython3
# Herschel PACS & SPIRE from ESA TAP using astroquery
# This search is fully functional, but is commented out because it takes
# ~4 hours to run to completion
herschel_radius = 1.1
herschel_download_directory = 'data/herschel'

# if not os.path.exists(herschel_download_directory):
#    os.makedirs(herschel_download_directory, exist_ok=True)
# df_spec_herschel =  Herschel_get_spec(sample_table, herschel_radius, herschel_download_directory, delete_downloaded_data=True)
# df_spec.append(df_spec_herschel)
```

### 2.4 SDSS Archive

```{code-cell} ipython3
%%time
# Get SDSS Spectra
df_spec_SDSS = SDSS_get_spec(sample_table, search_radius_arcsec=5, data_release=17)
df_spec.append(df_spec_SDSS)
```

### 2.5 DESI Archive

This includes DESI spectra. Here, we use the `SPARCL` query. Note that this can also be used
for SDSS searches, however, according to the SPARCL webpage, only up to DR16 is included.
Therefore, we will not include SDSS DR16 here (this is treated in the SDSS search above).

```{code-cell} ipython3
%%time
# Get DESI and BOSS spectra with SPARCL
df_spec_DESIBOSS = DESIBOSS_get_spec(sample_table, search_radius_arcsec=5)
df_spec.append(df_spec_DESIBOSS)
```

## 3. Make plots of luminosity as a function of time
We show flux in mJy as a function of time for all available bands for each object.
`show_nbr_figures` controls how many plots are actually generated and returned to the screen.
If you choose to save the plots with `save_output`, they will be put in the output directory and
labelled by sample number.

```{code-cell} ipython3
### Plotting ####
create_figures(df_spec=df_spec,
               bin_factor=5,
               show_nbr_figures=10,
               save_output=False,
               )
```
