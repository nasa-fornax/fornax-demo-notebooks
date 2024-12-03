import os

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip

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

    ## This is the faster way, however, it outputs warnings because there are empty slices.
    #lam_bin = np.arange(np.nanmin(wave.value)+dlam/2, np.nanmax(wave.value)+dlam/2 + dlam , dlam)
    #flux_bin = np.asarray( [ np.nanmedian(flux[(wave.value >= (ll-dlam/2)) & (wave.value < (ll+dlam/2) )].value) for ll in lam_bin ] )

    ## This way is a bit slower but we can avoid empty slices.
    lam_bin = np.arange(np.nanmin(wave.value)+dlam/2, np.nanmax(wave.value)+dlam/2 + dlam , dlam)
    lam_bins = []
    flux_bins = []
    for ll in lam_bin:
        sel_tmp = np.where( (wave.value >= (ll-dlam/2)) & (wave.value < (ll+dlam/2) ) )[0]
        if len(sel_tmp) > 0:
            flux_bins.append( np.nanmedian(flux[sel_tmp].value) )
            lam_bins.append(ll)

    ## Add the units back!
    lam_bins = np.asarray(lam_bins) * wave[0].unit
    flux_bins = np.asarray(flux_bins) * flux[0].unit
    dlam = dlam * wave[0].unit

    return(lam_bins , flux_bins, dlam)


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
        if ("IRS" in all_instruments) | ("PACS" in all_instruments) | ("SPIRE" in all_instruments):
            LOGX = True
        else:
            LOGX = False
        
        for ff, (filt,filt_df) in enumerate(singleobj_df.groupby('filter')):
    
            #print("{} entries for a object {} and filter {}".format(len(filt_df.flux), objectid , filt))
            for ii in range(len(filt_df.flux)):

                # get data
                wave = filt_df.reset_index().wave[ii].to(u.micrometer)
                flux = filt_df.reset_index().flux[ii]
                err = filt_df.reset_index().err[ii]

                # do masking to remove value that are not finite
                mask = np.where((np.isfinite(err)) & (flux > 0))[0]
                wave = wave[mask]
                flux = flux[mask]
                err = err[mask]

                #ax1.plot(wave , flux , "-"  , label="{} ({})".format(filt, filt_df.reset_index().mission[ii]) )
                wave_bin , flux_bin, _ = bin_spectra(wave, flux, bin_factor=bin_factor)

                # do some more clearning (mainly to remove some very low values)
                selnotnan = np.where(~np.isnan(flux_bin))[0]
                flux_bin = flux_bin[selnotnan]
                wave_bin = wave_bin[selnotnan]
                clip_mask = sigma_clip(flux_bin , sigma_lower = 3, cenfunc=np.nanmedian , sigma_upper = 5, stdfunc = np.nanstd).mask
                wave_bin = wave_bin[~clip_mask]
                flux_bin = flux_bin[~clip_mask]
                
                ax1.step(wave_bin.to(u.micrometer) , flux_bin.to(u.erg / u.second / (u.centimeter**2) / u.angstrom) , "-" , label="{} ({})".format(filt, filt_df.reset_index().mission[ii]), where="mid") # all in um and..
    
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