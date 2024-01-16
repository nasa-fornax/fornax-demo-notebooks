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
| IRSA    | Keck    | About 10,000 spectra on the COSMOS field from [Hasinger et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...858...77H/abstract) | [IRSA Archive](https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-scan?projshort=COSMOS) | Straight forward to implement via IRSA API |
| IRSA    | Spitzer IRS | ~17,000 merged low-resolution IRS spectra | [IRS Enhanced Product](https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=irs_enhv211) | IRS Enhanced products can be searched through IRSA API and downloaded as IPAC table. Does `astroquery.ipac.irsa` work? |
| IRSA    | IRTF*        | Large library of stellar spectra | | does `astroquery.ipac.irsa` work?? |
| IRSA    | Herschel*    | Some spectra, need to check reduction stage | | |
| IRSA    | Euclid      | Spectra hosted at IRSA in FY25 -> preparation for ingestion | | Will use mock spectra with correct format for testing |
| IRSA    | SPHEREx     | Spectra/cubes will be hosted at IRSA, first release in FY25 -> preparation for ingestion | | Will use mock spectra with correct format for testing |
| MAST    | HST*         | Slitless spectra would need reduction and extraction. There are some reduced slit spectra from COS in the Hubble Archive | `astroquery.mast`? | Should be straight forward using `astroquery.mast` |
| MAST    | JWST*        | Reduced slit MSA spectra that can be queried | `astroquery.mast`? | Should be straight forward using `astroquery.mast` |
| SDSS    | SDSS optical| Optical spectra that are reduced | [Sky Server](https://skyserver.sdss.org/dr18/SearchTools) or `astroquery.sdss` (preferred) | (ALF has code to get spectra via skyserver). Need to look into `astroquery`. |
| DESI    | DESI*        | Optical spectra | [DESI public data release](https://data.desi.lbl.gov/public/) | No obvious API. `pyvo` might work, need to look into this. |
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

## Next Steps:

&bull; Start with HSt and JWST. Is there an easy way to download the spectra?

&bull; Contact IRSA folks (Anastasia) to ask whether Herschel and Spitzer IRS can be accessed via the new IRSA API (and `astroquery`?)


<!-- #endregion -->

```python
## test the HST spectrum retrieval.
# For this, take a galaxy for which we know HST spectroscopy exists
```

```python
## Your code here
```

```python

```
