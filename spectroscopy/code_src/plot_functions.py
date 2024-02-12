import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib as mpl

from data_structures_spec import MultiIndexDFObject

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