import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy import stats
from tqdm import tqdm


def setup_text_plots():
    """
    This function adjusts matplotlib settings so that all figures have a uniform format and look.
    """
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelpad'] = 7
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

def create_figures(sample_table , df_lc, show_nbr_figures, save_output):
    '''
    Creates figures of the lightcurves for each source in the sample_table.
    
    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates, objectid's and journal reference labels of the sources
        
    df_lc : lightcurve object
        Lightcurve objects from which to create the lightcurve figures.
        
    show_nbr_figures : int
        Number of figures to show inline. For example, `show_nbr_figures = 5' would
        show the first 5 figures inline.
        
    save_output: bool
        Whether to save the lightcurve figures. If saved, they will be in the "output" directory.
        
    
    Returns
    -------
    Saves the figures in the output directory
    
    
    Notes
    -----
    By default, if save_output is True, figures are
    made and saved for *all* sources. If that is too much, the user can create a selection
    from the sample of which sources they like to plot (and save).
    
    '''
    
    if (show_nbr_figures == 0) & (not save_output):
        print("No figures are shown or saved.")
        return(False)
    
    cc = 0
    for row in sample_table:
        objectid = row['objectid']
        cc += 1 # counter (1-indexed)
        
        ## Set up =================
        # choose whether to plot data from the serial or parallel calls
        singleobj = df_lc.data.loc[objectid]
        # singleobj = parallel_df_lc.data.loc[objectid]

        # Set up for plotting. We use the "mosaic" method so we can plot
        # the ZTF data in a subplot for better visibility.
        fig, axes = plt.subplot_mosaic(mosaic=[["A"],["A"],["B"]] , figsize=(10,8))
        plt.subplots_adjust(hspace=0.3 , wspace=0.3)

        ## Plot all the bands in the *main plot* (A) ====================
        leg_handles_A = []
        max_list = [] # store maximum flux for each band
        ztf_minmax_tab = Table(names=["tmin","tmax","fluxmin","fluxmax"]) # store the min and max of the ZTF band fluxes and time
        has_ztf = False # flag to set to True if ZTF data is available.
        has_icecube = False # flag to set to True if IceCube data is available.

        for band in singleobj.index.unique('band'):

            # get data
            band_lc = singleobj.loc[:, band, :]
            band_lc.reset_index(inplace = True)

            # first clean dataframe to remove erroneous rows
            band_lc_clean = band_lc[band_lc['time'] < 65000]
            
            # Do some sigma-clipping, but only if more than 10 data points.
            if len(band_lc_clean) >= 10:
                band_lc_clean = band_lc_clean[np.abs(stats.zscore(band_lc_clean.flux.values.astype(float))) < 3.0]

            # before plotting need to scale the Kepler, K2, and TESS fluxes to the other available fluxes
            if band in ['Kepler', 'K2', 'TESS']: # Note: these are not included anymore...

                #remove outliers in the dataset
                #bandlc_clip = band_lc_clean[(np.abs(stats.zscore(band_lc_clean['flux'])) < 3.0)]

                #find the maximum value of 'other bands'
                if len(band_lc_clean) > 0:  
                    max_electrons = max(band_lc_clean.flux)
                    factor = np.mean(max_list)/ max_electrons
                    #lh = axes["A"].errorbar(bandlc_clip.time, bandlc_clip.flux * factor, bandlc_clip.err* factor,
                    #                        capsize = 3.0,label = band)
                    lh = axes["A"].errorbar(band_lc_clean.time, band_lc_clean.flux * factor, band_lc_clean.err* factor,
                                            capsize = 3.0,label = band)

            # ZTF is special because we are plotting the data also in "B" zoom-in
            elif band in ['zg','zr','zi']: # for ZTF
                has_ztf = True
                max_list.append(max(band_lc_clean.flux)) 
                lh = axes["A"].errorbar(band_lc_clean.time, band_lc_clean.flux, band_lc_clean.err,
                                        capsize = 1.0, elinewidth=0.5,marker='o',markersize=2,linestyle='', label = "ZTF {}".format(band))
                ztf_minmax_tab.add_row( [np.min(band_lc_clean.time) , np.max(band_lc_clean.time) , np.min(band_lc_clean.flux) , np.max(band_lc_clean.flux) ] )


                # plot ZTF in zoomin
                p1 = axes["B"].errorbar(band_lc_clean.time, band_lc_clean.flux, band_lc_clean.err,
                                        capsize = 1.0, elinewidth=0.5, marker='o',linestyle='',markersize=0.5, alpha=0.5,
                                        label = "ZTF {}".format(band) , color=lh.lines[0].get_color())



                # overplot running mean fo ZTF in zoomin 
                xx = band_lc_clean.time.values # Note: need to use .values here to remove indexing.
                yy = band_lc_clean.flux.values # Note: need to use .values here to remove indexing.
                ee = band_lc_clean.err.values # Note: need to use .values here to remove indexing.
                x_bin = 30 # in MJD
                x_grid = np.arange(np.nanmin(xx) , np.nanmax(xx)+x_bin/4 , x_bin/4)
                tmp = Table(names=["xbin","ybin","yerr"])

                for xxx in x_grid:
                    s = np.where( np.abs(xx - xxx) < x_bin/2 )[0]
                    if len(s) > 1:
                        mn = np.nansum(yy[s]*ee[s]) / np.nansum(ee[s]) # weighted mean
                        tmp.add_row([xxx , mn , np.nanstd(yy[s])])
                    else:
                        tmp.add_row([xxx , np.nan , np.nan])

                axes["B"].plot(tmp["xbin"] , tmp["ybin"] , "-", linewidth=1.5 , color=p1.lines[0].get_color())



            # IceCube is special because it's only events (= limits on plot.)
            elif band in ["IceCube"]:
                has_icecube = True
                # We deal with this later. Need to wait for all the things to plot
                # so we know the y limits.

            # Now plot everything else
            else:
                max_list.append(max(band_lc_clean.flux)) 
                lh = axes["A"].errorbar(band_lc_clean.time, band_lc_clean.flux, band_lc_clean.err,
                                        capsize = 3.0, label = band)

            if band not in ["IceCube"]:
                leg_handles_A.append(lh) # add legend handles

        ## Now plot IceCube.
        # we had to wait for all the data to be plotted so we know
        # the y-limits of the resulting final plot. So, we do IceCube
        # at the end.
        if has_icecube:
            band_lc = singleobj.loc[:, "IceCube", :]
            band_lc.reset_index(inplace = True)
            band_lc_clean = band_lc[band_lc['time'] < 65000]

            y = axes["A"].get_ylim()[0] + np.diff(axes["A"].get_ylim())*0.7
            dy = np.diff(axes["A"].get_ylim())/20
            lh = axes["A"].errorbar(band_lc_clean.time , np.repeat(y , len(band_lc_clean.time)) , yerr=dy, uplims=True ,
                                    fmt="o"  , label="IceCube" , color="black")

            leg_handles_A.append(lh) # add legend handles (for IceCube)


        ## Do Axes ===============
        axes["A"].set_ylabel('Flux(mJy)')

        # Plot the ZTF bands in a separate plot to show their variability
        # more clearly. Can still also plot the rest, just change the x and
        # y axis limits. Only do this if ZTF is available for source.
        if has_ztf:
            axes["B"].set_ylabel('Flux(mJy)')
            axes["B"].set_xlabel('Time(MJD)')
            axes["B"].set_xlim( np.min(ztf_minmax_tab["tmin"])-100 , np.max(ztf_minmax_tab["tmax"])+100 )
        else:
            axes["A"].set_xlabel('Time(MJD)')

        ## Make nice axis
        axes["A"].grid(linestyle=":",color="lightgray", linewidth=1)
        axes["A"].minorticks_on()
        axes["A"].tick_params(axis='x', which='minor', bottom=True)
        axes["A"].tick_params(axis="both", which="major",direction='in', length=6, width=1)
        axes["A"].tick_params(axis="both", which="minor",direction='in', length=3, width=1)
        axes["B"].grid(linestyle=":",color="lightgray", linewidth=1)
        axes["B"].minorticks_on()
        axes["B"].tick_params(axis='x', which='minor', bottom=True)
        axes["B"].tick_params(axis="both", which="major",direction='in', length=6, width=1)
        axes["B"].tick_params(axis="both", which="minor",direction='in', length=3, width=1)

        plt.legend(handles=leg_handles_A , bbox_to_anchor=(1.2,3.5))
        plt.tight_layout()

        if save_output:
            savename = os.path.join("output" , "lightcurve_{}.pdf".format(objectid) ) 
            plt.savefig(savename, bbox_inches="tight")

        if cc <= show_nbr_figures:
            plt.show()
        else:
            plt.close()
            
            
        # If figures are not saved, we only have to loop over the 
        # number of figures that are shown to the user.
        if (not save_output) & (cc == show_nbr_figures):
            print("Done")
            return(True)
        else:
            pass
            
    print("Done")
    return(True)
