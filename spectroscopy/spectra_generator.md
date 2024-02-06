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
| IRSA    | IRTF*        | Large library of stellar spectra | | does `astroquery.ipac.irsa` work?? |
| IRSA    | Herschel*    | Some spectra, need to check reduction stage | | |
| IRSA    | Euclid      | Spectra hosted at IRSA in FY25 -> preparation for ingestion | | Will use mock spectra with correct format for testing |
| IRSA    | SPHEREx     | Spectra/cubes will be hosted at IRSA, first release in FY25 -> preparation for ingestion | | Will use mock spectra with correct format for testing |
| MAST    | HST*         | Slitless spectra would need reduction and extraction. There are some reduced slit spectra from COS in the Hubble Archive | `astroquery.mast`? | Implemented using `astroquery.mast` |
| MAST    | JWST*        | Reduced slit MSA spectra that can be queried | `astroquery.mast`? | Should be straight forward using `astroquery.mast` |
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

## Next Steps:

&bull; Start with HSt and JWST. Is there an easy way to download the spectra?

&bull; Contact IRSA folks (Anastasia) to ask whether Herschel and Spitzer IRS can be accessed via the new IRSA API (and `astroquery`?)


<!-- #endregion -->

```python
## IMPORTS
import sys, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd

from astroquery.mast import Observations
from astroquery.sdss import SDSS
from astroquery.ipac.irsa import Irsa

from sparcl.client import SparclClient

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join, join_skycoord, unique
from astropy.nddata import StdDevUncertainty
import astropy.constants as const
#from astropy.nddata import InverseVariance
from astropy import nddata
#from astropy import cosmology

from specutils import Spectrum1D # had to pip install this one

sys.path.append('code_src/')
from data_structures_spec import MultiIndexDFObject
from sample_selection import clean_sample

!pip install sparclclient

## Plotting stuff
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['xtick.major.pad'] = 7
mpl.rcParams['ytick.major.pad'] = 7
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.minor.top'] = True
mpl.rcParams['xtick.minor.bottom'] = True
mpl.rcParams['ytick.minor.left'] = True
mpl.rcParams['ytick.minor.right'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['hatch.linewidth'] = 1

def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
```

```python
def DESIBOSS_get_spec(sample_table, search_radius_arcsec):
    '''
    Retrieves DESI and BOSS spectra for a list of sources.
    Note, that we can also retrieve SDSS-DR16 spectra here, which
    leads to similar results as SDSS_get_spec().

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds. Here its rather half a box size.

    Returns
    -------
    df_lc : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    
    ## Set up client
    client = SparclClient()
    #print(client.all_datasets) # print data sets
    
    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()
    
    
    for stab in sample_table:
    
        ## Search
        data_releases = ['DESI-EDR','BOSS-DR16']
        #data_releases = ['DESI-EDR','BOSS-DR16','SDSS-DR16']
        
        search_coords = stab["coord"]
        dra = (search_radius_arcsec*u.arcsec).to(u.degree)
        ddec = (search_radius_arcsec*u.arcsec).to(u.degree)
        out = ['sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'data_release', 'redshift_err']
        cons = {'spectype': ['GALAXY','STAR','QSO'],
                'data_release': data_releases,
                #'redshift': [0.5, 0.9],
                'ra' : [search_coords.ra.deg-dra.value  , search_coords.ra.deg+dra.value ],
                'dec' : [search_coords.dec.deg-ddec.value  , search_coords.dec.deg+ddec.value ]
               }
        found_I = client.find(outfields=out, constraints=cons, limit=20) # search
        #print(found_I)
        
        ## Extract nice table and the spectra
        if len(found_I.records) > 0:
            result_tab = Table(names=found_I.records[0].keys() , dtype=[ type(found_I.records[0][key]) for key in found_I.records[0].keys()])
            _ = [ result_tab.add_row([f[key] for key in f.keys()]) for f in found_I.records]
        
            sep = [search_coords.separation(SkyCoord(tt["ra"], tt["dec"], unit=u.deg, frame='icrs')).to(u.arcsecond).value for tt in result_tab]
            result_tab["separation"] = sep
            
            ## Retrieve Spectra
            inc = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                   'wavelength', 'model', 'ivar', 'mask', 'spectype', 'ra', 'dec']
            results_I = client.retrieve(uuid_list=found_I.ids, include=inc)
            specs = [Spectrum1D(spectral_axis = r.wavelength*u.AA,
                                flux = np.array(r.flux)* 10**-17 * u.Unit('erg cm-2 s-1 AA-1'),
                                uncertainty = nddata.InverseVariance(np.array(r.ivar)),
                                redshift = r.redshift,
                                mask = r.mask)
                    for r in results_I.records]
            
        
            ## Choose objects
            for dr in data_releases:
                
                sel = np.where(result_tab["data_release"] == dr)[0]
                if len(sel) > 0: # found entry
                    idx_closest = sel[ np.where(result_tab["separation"][sel] == np.nanmin(result_tab["separation"][sel]))[0][0] ]
            
                    # create MultiIndex Object
                    dfsingle = pd.DataFrame(dict(wave=[specs[idx_closest].spectral_axis] ,
                                                     flux=[specs[idx_closest].flux],
                                                     err=[np.sqrt(1/specs[idx_closest].uncertainty.quantity)],
                                                                         label=[stab["label"]],
                                                                         objectid=[stab["objectid"]],
                                                                         mission=[dr],
                                                                         instrument=[dr],
                                                                         filter=["optical"],
                                                                        )).set_index(["objectid", "label", "filter", "mission"])
                    df_spec.append(dfsingle)
                
    return(df_spec)

def SpitzerIRS_get_spec(sample_table, search_radius_arcsec , COMBINESPEC):
    '''
    Retrieves HST spectra for a list of sources.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds.
    COMBINESPEC : `bool`
        If set to `True`, then, if multiple spectra are found, the spectra are
        mean-combined. If `False`, the closest spectrum is chosen and returned.

    Returns
    -------
    df_lc : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()
    
    for stab in sample_table:
    
        print("Processing source {}".format(stab["label"]))
        
        ## Do search
        search_coords = stab["coord"]
        tab = Irsa.query_region(coordinates=search_coords, catalog="irs_enhv211", spatial="Cone",radius=search_radius_arcsec * u.arcsec)
        print("Number of entries found: {}".format(len(tab)))
    
        if len(tab) > 0: # found a source
        
            ## If multiple entries are found, pick the closest.
            # Or should we take the average instead??
            if len(tab) > 0:
                print("More than 1 entry found" , end="")
                if not COMBINESPEC:
                    print(" - pick the closest")
                    sep = [search_coords.separation(SkyCoord(tt["ra"], tt["dec"], unit=u.deg, frame='icrs')).to(u.arcsecond).value for tt in tab]
                    id_min = np.where(sep == np.nanmin(sep))[0]
                    tab_final = tab[id_min]
                else:
                    print(" - Combine spectra")
                    tab_final = tab.copy()
            else:
                tab_final = tab.copy()
            
            ## Now extract spectra and put all in one array
            specs = []
            for tt in tab:
                url = "https://irsa.ipac.caltech.edu{}".format(tt["xtable"].split("\"")[1])
                spec = Table.read(url , format="ipac") # flux_density in Jy
                specs.append(spec)
                
            ## Create final spectrum (combine if necesary)
            # Note that spectrum is automatically combined if longer than length=1
            lengths = np.asarray( [len(spec["wavelength"]) for spec in specs]) # get the lengths of the spectra
            id_orig = np.where(lengths == np.nanmax(lengths))[0] # pick the spectra with the most wavelength as template
            if len(id_orig) > 0: id_orig = id_orig[0]
            wave_orig = np.asarray(specs[id_orig]["wavelength"]) # original wavelength
            fluxes = np.asarray([np.interp(wave_orig , spec["wavelength"] , spec["flux_density"]) for spec in specs ])
            errors = np.asarray([np.interp(wave_orig , spec["wavelength"] , spec["error"]) for spec in specs ])
            flux_jy_mean = np.nanmean(fluxes , axis=0)
            errors_jy_combined = np.asarray( [ np.sqrt( np.nansum(errors[:,ii]**2) ) for ii in range(errors.shape[1])] )
    
            # Change units
            wave_orig_A = (wave_orig*u.micrometer).to(u.angstrom)
            flux_cgs_mean = (flux_jy_mean*u.jansky).to(u.erg/u.second/u.centimeter**2/u.hertz) * const.c.to(u.angstrom/u.second) / (wave_orig_A**2)
            errors_cgs_combined = (errors_jy_combined*u.jansky).to(u.erg/u.second/u.centimeter**2/u.hertz) * const.c.to(u.angstrom/u.second) / (wave_orig_A**2)
            
            # Create MultiIndex object
            dfsingle = pd.DataFrame(dict(wave=[wave_orig_A] ,
                                         flux=[flux_cgs_mean],
                                         err=[errors_cgs_combined],
                                                             label=[stab["label"]],
                                                             objectid=[stab["objectid"]],
                                                             mission=["Spitzer"],
                                                             instrument=["IRS"],
                                                             filter=["IR"],
                                                            )).set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)
    
    return(df_spec)


def SDSS_get_spec(sample_table, search_radius_arcsec):
    '''
    Retrieves SDSS spectra for a list of sources. Note that no data will
    be directly downloaded. All will be saved in cache.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds.

    Returns
    -------
    df_lc : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    
    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()
    
    for stab in sample_table:

        ## Get Spectra for SDSS
        search_coords = stab["coord"]
        
        xid = SDSS.query_region(search_coords, radius=search_radius_arcsec * u.arcsec, spectro=True)

        if str(type(xid)) != "<class 'NoneType'>":
            sp = SDSS.get_spectra(matches=xid, show_progress=True)
    
            ## Get data
            wave = 10**sp[0]["COADD"].data.loglam * u.angstrom # only one entry because we only search for one xid at a time. Could change that?
            flux = sp[0]["COADD"].data.flux*1e-17 * u.erg/u.second/u.centimeter**2/u.angstrom 
            err = np.sqrt(1/sp[0]["COADD"].data.ivar)*1e-17 * flux.unit
            
            ## Add to df_spec.
            dfsingle = pd.DataFrame(dict(wave=[wave] , flux=[flux], err=[err],
                                     label=[stab["label"]],
                                     objectid=[stab["objectid"]],
                                     mission=["SDSS"],
                                     instrument=["SDSS"],
                                     filter=["optical"],
                                    )).set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)

        else:
            print("Source {} could not be found".format(stab["label"]))


    return(df_spec)

def HST_get_spec(sample_table, search_radius_arcsec, datadir):
    '''
    Retrieves HST spectra for a list of sources.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds.
    datadir : `str`
        Data directory where to store the data. Each function will create a
        separate data directory (for example "[datadir]/HST/" for HST data).

    Returns
    -------
    df_lc : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    ## Create directory
    this_data_dir = os.path.join(datadir , "HST/")
    
    
    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()
    
    for stab in sample_table:

        print("Processing source {}".format(stab["label"]))

        ## Query results
        search_coords = stab["coord"]
        query_results = Observations.query_criteria(coordinates = search_coords, radius = search_radius_arcsec * u.arcsec,
                                                dataproduct_type=["spectrum"], obs_collection=["HST"], intentType="science", calib_level=[3,4],
                                               )
        print("Number of search results: {}".format(len(query_results)))

        if len(query_results) > 0: # found some spectra
            
            
            ## Retrieve spectra
            data_products_list = Observations.get_product_list(query_results)
            
            ## Filter
            data_products_list_filter = Observations.filter_products(data_products_list,
                                                    productType=["SCIENCE"],
                                                    extension="fits",
                                                    calib_level=[3,4], # only fully reduced or contributed
                                                    productSubGroupDescription=["SX1"] # only 1D spectra
                                                                    )
            print("Number of files to download: {}".format(len(data_products_list_filter)))

            if len(data_products_list_filter) > 0:
                
                ## Download
                download_results = Observations.download_products(data_products_list_filter, download_dir=this_data_dir)
            
                
                ## Create table
                keys = ["filters","obs_collection","instrument_name","calib_level","t_obs_release","proposal_id","obsid","objID","distance"]
                tab = Table(names=keys , dtype=[str,str,str,int,float,int,int,int,float])
                for jj in range(len(download_results)):
                    tmp = query_results[query_results["obsid"] == data_products_list_filter["obsID"][jj]][keys]
                    tab.add_row( list(tmp[0]) )
                
                ## Create multi-index object
                for jj in range(len(tab)):
                
                    # open spectrum
                    filepath = download_results[jj]["Local Path"]
                    print(filepath)
                    spec1d = Spectrum1D.read(filepath)  
                    
                    dfsingle = pd.DataFrame(dict(wave=[spec1d.spectral_axis] , flux=[spec1d.flux], err=[np.repeat(0,len(spec1d.flux))],
                                                 label=[stab["label"]],
                                                 objectid=[stab["objectid"]],
                                                 #objID=[tab["objID"][jj]],
                                                 #obsid=[tab["obsid"][jj]],
                                                 mission=[tab["obs_collection"][jj]],
                                                 instrument=[tab["instrument_name"][jj]],
                                                 filter=[tab["filters"][jj]],
                                                )).set_index(["objectid", "label", "filter", "mission"])
                    df_spec.append(dfsingle)
            
            else:
                print("Nothing to download for source {}.".format(stab["label"]))
        else:
            print("Source {} could not be found".format(stab["label"]))
        

    return(df_spec)


def KeckDEIMOS_get_spec(sample_table, search_radius_arcsec):
    '''
    Retrieves Keck DEIMOS on COSMOS spectra for a list of sources.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds.

    Returns
    -------
    df_lc : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()
    
    for stab in sample_table:

        search_coords = stab["coord"]
        tab = Irsa.query_region(coordinates=search_coords, catalog="cosmos_deimos", spatial="Cone",radius=search_radius_arcsec * u.arcsec)
        print("Number of entries found: {}".format(len(tab)))
        
        
        if len(tab) > 0: # found a source
        
            ## If multiple entries are found, pick the closest.
            if len(tab) > 1:
                print("More than 1 entry found - pick the closest. ")
                sep = [search_coords.separation(SkyCoord(tt["ra"], tt["dec"], unit=u.deg, frame='icrs')).to(u.arcsecond).value for tt in tab]
                id_min = np.where(sep == np.nanmin(sep))[0]
                tab_final = tab[id_min]
            else:
                tab_final = tab.copy()
        
        
            if "_acal" in tab_final["fits1d"][0]:
                ISCALIBRATED = True
            else:
                print("no calibration is available")
                ISCALIBRATED = False
        
        
            if ISCALIBRATED: # only if calibrated spectrum is available
            
                ## Now extract spectrum
                # ASCII 1d spectrum for file spec1d.cos0.034.VD_07891 
                # wavelength in Angstroms and observed-frame
                # flux (f_lambda) in 1e-17 erg/s/cm2/A if calibrated, else in counts
                # ivar (inverse variance) in 1e-17 erg/s/cm2/A if calibrated, else in counts
                # wavelength flux ivar 
                url = "https://irsa.ipac.caltech.edu{}".format(tab_final["fits1d"][0].split("\"")[1])
                spec = Table.read(url)
                
                # Prepare arrays
                wave = spec["LAMBDA"][0] * u.angstrom
                flux_cgs = spec["FLUX"][0] * 1e-17 * u.erg/u.second/u.centimeter**2/u.angstrom
                error_cgs = np.sqrt( 1 / spec["IVAR"][0]) * 1e-17 * u.erg/u.second/u.centimeter**2/u.angstrom
                
                # Create MultiIndex object
                dfsingle = pd.DataFrame(dict(wave=[wave] ,
                                             flux=[flux_cgs],
                                             err=[error_cgs],
                                                                 label=[stab["label"]],
                                                                 objectid=[stab["objectid"]],
                                                                 mission=["Keck"],
                                                                 instrument=["DEIMOS"],
                                                                 filter=["optical"],
                                                                )).set_index(["objectid", "label", "filter", "mission"])
                df_spec.append(dfsingle)

    return(df_spec)


def bin_spectra(wave,flux, bin_factor):
    '''
    Does a very crude median binning on a spectrum.

    Parameters
    ----------
    wave: `astropy.ndarray`
        Wavelength (can be any units)
    flux: `astropy.ndarray`
        Flux (can be any linear units)
    bin_factor: `float`
        Binning factor in terms of average wavelength resolution

    Returns
    -------
    A tuple (wave_bin , flux_bin , dwave) where
    wave_bin: `astropy.ndarray`
        Binned wavelength.
    flux_bin: `astropy.ndarray`
        Binned flux
    dwave: `float`
        The wavelength resolution used for the binning.
    
    '''

    dlam = np.nanmedian(np.diff(wave.value)) * bin_factor

    lam_bin = np.arange(np.nanmin(wave.value)+dlam/2, np.nanmax(wave.value)+dlam/2 + dlam , dlam)
    flux_bin = np.asarray( [ np.nanmedian(flux[(wave.value >= (ll-dlam/2)) & (wave.value < (ll+dlam/2) )].value) for ll in lam_bin ] )

    return(lam_bin , flux_bin, dlam)


def create_figures(df_spec, bin_factor, show_nbr_figures , save_output):
    '''
    Plots the spectra of the sources.

    Parameters
    ----------
    df_spec: MultiIndexDFObject
        The main data structure to store all spectra
    
    bin_factor: `float`
        Binning factor in terms of average wavelength resolution
    
    show_nbr_figures : int
        Number of figures to show inline. For example, `show_nbr_figures = 5' would
        show the first 5 figures inline.
        
    save_output: bool
        Whether to save the lightcurve figures. If saved, they will be in the "output" directory.
    
    '''

    
    for cc, (objectid, singleobj_df) in enumerate(df_spec.data.groupby('objectid')):
    
        fig = plt.figure(figsize=(9,6))
        ax1 = fig.add_subplot(1,1,1)

        this_label = list(singleobj_df.groupby('label'))[0][0]

        ## Logarithmic or linear x-axis?
        all_instruments = [ list(singleobj_df.groupby('instrument'))[ii][0] for ii in range(len(list(singleobj_df.groupby('instrument')))) ]
        if "IRS" in all_instruments:
            LOGX = True
        else:
            LOGX = False
        
        for ff, (filt,filt_df) in enumerate(singleobj_df.groupby('filter')):
    
            print("{} entries for a object {} and filter {}".format(len(filt_df.flux), objectid , filt))
            for ii in range(len(filt_df.flux)):
    
                wave = filt_df.reset_index().wave[ii]
                flux = filt_df.reset_index().flux[ii]
                err = filt_df.reset_index().err[ii]
                mask = np.where(np.isfinite(err))[0]
                wave = wave[mask]
                flux = flux[mask]
                err = err[mask]
                
                #ax1.plot(wave/1e4 , flux , "-" , label="{} ({})".format(filt, filt_df.reset_index().mission[0]))
                wave_bin , flux_bin, _ = bin_spectra(wave, flux, bin_factor=bin_factor)
                ax1.step(wave_bin/1e4 , flux_bin , "-" , label="{} ({})".format(filt, filt_df.reset_index().mission[ii]), where="mid")
    
        ax1.set_title(this_label)
        if LOGX: ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"Wavelength [$\rm \mu m$]")
        ax1.set_ylabel(r"Flux [erg/s/cm$^2$/$\rm \AA$]")
        ax1.legend(bbox_to_anchor=(1.27,1), fontsize=11)

        if save_output:
            savename = os.path.join("output" , "spectra_{}.pdf".format(objectid) ) 
            plt.savefig(savename, bbox_inches="tight")
        
        if cc < show_nbr_figures:
            plt.show()
        else:
            plt.close()

    return(True)
```

```python
## Get Source list:

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
labels.append("None")



sample_table = clean_sample(coords, labels , verbose=1)

print("Number of sources in sample table: {}".format(len(sample_table)))
```

```python
## Initialize multi-index object
df_spec = MultiIndexDFObject()
```

```python
## Get Keck Spectra (COSMOS only)
df_spec_DEIMOS = KeckDEIMOS_get_spec(sample_table = sample_table, search_radius_arcsec=1)
df_spec.append(df_spec_DEIMOS)
```

```python
%%time
## Get Spectra for HST
df_spec_HST = HST_get_spec(sample_table , search_radius_arcsec = 0.5, datadir = "./data/")
df_spec.append(df_spec_HST)
```

```python
%%time
## Get SDSS Spectra
df_spec_SDSS = SDSS_get_spec(sample_table , search_radius_arcsec=5)
df_spec.append(df_spec_SDSS)
```

```python
%%time
## Get Spitzer IRS Spectra
df_spec_IRS = SpitzerIRS_get_spec(sample_table, search_radius_arcsec=1 , COMBINESPEC=False)
df_spec.append(df_spec_IRS)

```

```python
%%time
## Get DESI and BOSS spectra with SPARCL
df_spec_DESIBOSS = DESIBOSS_get_spec(sample_table, search_radius_arcsec=5)
df_spec.append(df_spec_DESIBOSS)
```

```python
df_spec.data
```

```python
### Plotting ####
create_figures(df_spec = df_spec,
             bin_factor=10,
             show_nbr_figures = 10,
             save_output = False,
             )
```

```python
END
```

```python
### FOR TESTING ONLY ####

bin_factor = 10

for cc, (objectid, singleobj_df) in enumerate(df_spec.data.groupby('objectid')):

    if cc == 0:
        
        fig = plt.figure(figsize=(9,6))
        ax1 = fig.add_subplot(1,1,1)
    
        this_label = list(singleobj_df.groupby('label'))[0][0]
    
        ## Logarithmic or linear x-axis?
        all_instruments = [ list(singleobj_df.groupby('instrument'))[ii][0] for ii in range(len(list(singleobj_df.groupby('instrument')))) ]
        if "IRS" in all_instruments:
            LOGX = True
        else:
            LOGX = False
        
        for ff, (filt,filt_df) in enumerate(singleobj_df.groupby('filter')):
    
            print("{} entries for a object {} and filter {}".format(len(filt_df.flux), objectid , filt))
            for ii in range(len(filt_df.flux)):
    
                wave = filt_df.reset_index().wave[ii]
                flux = filt_df.reset_index().flux[ii]
                err = filt_df.reset_index().err[ii]
                mask = np.where(np.isfinite(err))[0]
                wave = wave[mask]
                flux = flux[mask]
                err = err[mask]
                
                #ax1.plot(wave/1e4 , flux , "-" , label="{} ({})".format(filt, filt_df.reset_index().mission[0]))
                wave_bin , flux_bin, _ = bin_spectra(wave, flux, bin_factor=bin_factor)
                ax1.step(wave_bin/1e4 , flux_bin , "-" , label="{} ({})".format(filt, filt_df.reset_index().mission[ii]), where="mid")
    
        ax1.set_title(this_label)
        if LOGX: ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"Wavelength [$\rm \mu m$]")
        ax1.set_ylabel(r"Flux [erg/s/cm$^2$/$\rm \AA$]")
        ax1.legend(bbox_to_anchor=(1.27,1), fontsize=11)
```
