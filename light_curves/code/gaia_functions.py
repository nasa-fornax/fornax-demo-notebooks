import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table, hstack, vstack
from astropy.time import Time
from astroquery.gaia import Gaia
from tqdm import tqdm

from data_structures import MultiIndexDFObject
from sample_selection import make_coordsTable


def Gaia_get_lightcurve(coords_list, labels_list , verbose):
    '''
    Creates a lightcurve Pandas MultiIndex object from Gaia data for a list of coordinates.
    This is the MAIN function.
    
    Parameters
    ----------
    coords_list : list of Astropy SkyCoord objects
        List of (id,coordinates) tuples of the sources
    
    labels_list : list of str
        List of labels for each soruce
        
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
    
    
    Returns
    --------
    MultiIndexDFObject  including all the sources with matches
    in Gaia. This includes epoch photometry.
    
    '''

    ## Retrieve Gaia table with Source IDs ==============
    gaia_table = Gaia_retrieve_catalog(coords_list , labels_list,
                                         search_radius = 1/3600.,
                                         verbose = verbose
                                         )
    
    print(gaia_table.columns)   
    
    ## Extract Light curves ===============
    # For each of the objects, request the EPOCH_PHOTOMETRY from the Gaia DataLink Service

    ## Run search
    prod_tab = Gaia_retrieve_EPOCH_PHOTOMETRY(ids=list(gaia_table["source_id"]) , verbose=verbose)

    ## Create light curves =================
    # Note that epoch photometry is used unless there is none, in which case
    # the median photometry is used with a median time of all Gaia observations.
    gaia_epoch_phot = Gaia_mk_lightcurves(prod_tab , verbose=verbose)

    ## Create Gaia Pandas MultiIndex object and append to existing data frame.
    df_lc_gaia = Gaia_mk_MultiIndex(coords_list=coords_list,
                                    labels_list=labels_list, gaia_phot = gaia_table,
                                    gaia_epoch_phot=gaia_epoch_phot, verbose=verbose)

    df_lc = MultiIndexDFObject()
    df_lc.append(df_lc_gaia)
    
    
    return(df_lc)

def Gaia_retrieve_catalog(coords_list , labels_list , search_radius, verbose):
    '''
    Retrieves the photometry table for a list of sources.
    
    Parameter
    ----------    
    coords_list : list of Astropy SkyCoord objects
        List of (id,coordinates) tuples of the sources
    
    labels_list : list of str
        List of labels for each soruce
                
    search_radius : float 
        Search radius in degrees, e.g., 1/3600.
        suggested search radius is 1 arcsecond or 1/3600.
        
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
        
    Returns
    --------
    Astropy table with the Gaia photometry for each source.
    
    '''
    t1 = time.time()
    
    #first make an astropy table from our master list of coordinates
    # as input to the pyvo TAP query 
    coordstab = make_coordsTable(coords_list, labels_list)

    #this query is too slow without gaia.random_index.  
    #Gaia helpdesk is aware of this bug somewhere on their end
    querystr = f"""
        SELECT gaia.ra, gaia.dec, gaia.random_index, gaia.source_id, mt.ra, mt.dec, mt.objectid
        FROM tap_upload.table_test AS mt
        JOIN gaiadr3.gaia_source_lite AS gaia
        ON 1=CONTAINS(POINT('ICRS',mt.ra,mt.dec),CIRCLE('ICRS',gaia.ra,gaia.dec,{search_radius}))
        """
    #use an asynchronous query of the Gaia database
    #cross match with our uploaded table
    j = Gaia.launch_job_async(query=querystr, upload_resource=coordstab, upload_table_name="table_test")
    
    results = j.get_results()
    
    if verbose : print("\nSearch completed in {:.2f} seconds".format((time.time()-t1) ) )
    if verbose : print("Number of objects matched: {} out of {}.".format(len(results),len(coords_list) ) )

    return results






## Define function to retrieve epoch photometry
def Gaia_retrieve_EPOCH_PHOTOMETRY(ids, verbose):
    """
    Function to retrieve EPOCH_PHOTOMETRY (or actually any) catalog product for Gaia
    entries using the DataLink. Note that the IDs need to be DR3 source_id and needs to be a list.
    
    Code fragments taken from: https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products
    
    Parameters
    ----------
    ids : list of int
        List of Gaia DR3 source IDs (source_id). 
        
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
    
    Returns
    --------
    Returns a dictionary (key = source_id) with a table of photometry as a function of time.
        
    """
    
    ## Log in (apparently not necessary for small queries)
    #Gaia.login(user=None , password=None)
    
    ## Some Definitions
    retrieval_type = 'EPOCH_PHOTOMETRY'# Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
    data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
    data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'

    ## Get the files
    datalink = Gaia.load_data(ids=ids,
                              data_release = data_release,
                              retrieval_type=retrieval_type,
                              data_structure = data_structure, verbose = False, output_file = None , overwrite_output_file=True)
    dl_keys  = list(datalink.keys())
    
    if verbose > 2:
        print(f'The following Datalink products have been downloaded:')
        for dl_key in dl_keys:
            print(f' * {dl_key}')
    
    ## Extract the info
    prod_tab = dict() # Dictionary to save the light curves. The key is the source_id
    for dd in ids:
        if verbose > 2: print("{}: ".format(dd) , end=" ")
        this_dl_key = 'EPOCH_PHOTOMETRY-Gaia DR3 {}.xml'.format(dd)
        if this_dl_key in datalink.keys():
            prod_tab[str(dd)] = datalink[this_dl_key][0].to_table()
            if verbose > 2: print("found")
        else:
            pass
            if verbose > 2: print("not found")
    
    return(prod_tab)

## Define function to extract light curve from product table.
def Gaia_mk_lightcurves(prod_tab , verbose):
    """
    This function creates light curves from the table downloaded with DataLink from the Gaia server.
    
    Parameters
    ----------
    prod_tab : astropy table
        Product table downloaded via datalink, produced by `Gaia_retrieve_EPOCH_PHOTOMETRY()`.
        
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
        
    Returns
    --------
    Returns a dictionary (key = source_id) including a dictionary of light curves for bands "G", "BP", "RP". Each
        of them includes a time stamp (`time_jd` and `time_isot`) a magnitude (`mag`) and magnitude error (`magerr`).
    """
    
    bands = ["G","BP","RP"]
    output = dict()
    for ii,key in enumerate(list(prod_tab.keys()) ):
        if verbose > 2: print(key)
    
        output[str(key)] = dict()
        for band in bands:
            sel_band = np.where( (prod_tab[key]["band"] == band) & (prod_tab[key]["rejected_by_photometry"] == False) )[0]
            if verbose > 1: print("Number of entries for band {}: {}".format(band , len(sel_band)))
            
            time_jd = prod_tab[key][sel_band]["time"] + 2455197.5 # What unit???
            time_isot = Time(time_jd , format="jd").isot
            mag = prod_tab[key][sel_band]["mag"]
            magerr = 2.5/np.log(10) * prod_tab[key][sel_band]["flux_error"]/prod_tab[key][sel_band]["flux"]
            
            output[str(key)][band] = Table([time_jd , time_isot , mag , magerr] , names=["time_jd","time_isot","mag","magerr"] ,
                                           dtype = [float , str , float , float], units=[u.d , ""  , u.mag , u.mag])
            
    return(output)



def Gaia_mk_MultiIndex(coords_list, labels_list, gaia_phot, gaia_epoch_phot , verbose):
    '''
    Creates a MultiIndex Pandas Dataframe for the Gaia observations. Specifically, it 
    returns the epoch photometry as a function of time. For sources without Gaia epoch
    photometry, it just returns the mean photometry a epoch 2015-09-24T19:40:33.468, which
    is the average epoch of the observations of sources with multi-epoch photometry.
    
    Parameters
    ----------
    coords_list : list of Astropy SkyCoord objects
        List of (id,coordinates) tuples of the sources
    
    gaia_phot : dict
        The Gaia mean photometry (will be linked by object ID in 'data' catalog)
    
    gaia_epoch_phot : dict
        The Gaia epoch photometry (is a dictionary created by 'Gaia_mk_lightcurves()' function)
    
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
    
    
    Returns
    --------
    Returns a Pandas data frame with indices ("objectid","band","time"). The data frame contains flux and flux error,
    both in mJy. The output can be appended to another lightcurve Pandas data
    frame via df_lc_object.append(df_lc)
    '''

    for objectid, _ in coords_list:
        #print("{} matched to: ".format( data["Object Name"][ii])  , end=" ")

        ## Check if this object has a Gaia light curve:

        # get Gaia source_id
        #sel = np.where(data["Object Name"][ii] == gaia_phot["input_object_name"])[0]
        sel = np.where(objectid == gaia_phot["objectid"])[0]
        lab = labels_list[objectid]
        
        if len(sel) > 0:
            source_id = gaia_phot["source_id"][sel[0]]
            if verbose > 1: print(source_id , end=" ")

            if str(source_id) in gaia_epoch_phot.keys(): # Match to Gaia multi-epoch catalog
                if verbose > 1: print("Has Gaia epoch photometry")

                for band in ["G","BP","RP"]:

                    # get data
                    d = gaia_epoch_phot[str(source_id)][band]["time_isot"]
                    t = Time(d , format="isot") # convert to time object
                    y = gaia_epoch_phot[str(source_id)][band]["mag"]
                    dy = gaia_epoch_phot[str(source_id)][band]["magerr"]

                    # compute flux and flux error in mJy
                    y2 = 10**(-0.4*(y - 23.9))/1e3 # in mJy
                    dy2 = dy / 2.5 * np.log(10) * y2 # in mJy

                    # create single instance
                    dfsingle = pd.DataFrame(
                                dict(flux=np.asarray(y2), # in mJy
                                 err=np.asarray(dy2), # in mJy
                                 time=t.mjd, # in MJD
                                 #objectid=gaia_phot["input_object_name"][sel],
                                 objectid=np.repeat(objectid, len(y)),label=lab,
                                 band="Gaia {}".format(band.lower())
                                                )
                                           ).set_index(["objectid","label", "band", "time"])

                    # add to table
                    try:
                        this_df_lc
                    except NameError:
                        #this_df_lc doesn't exist (yet)
                        this_df_lc = dfsingle.copy()
                    else:
                        #this_df_lc_gaia exists
                        this_df_lc = pd.concat([this_df_lc, dfsingle])

            else: # No match to Gaia multi-epoch catalog: can safely ignore this object
                
                if verbose > 1: print("No Gaia epoch photometry ")

 
        else: # no match to Gaia
            if verbose > 1: print("none")
            
    return(this_df_lc)


def Gaia_plot_lightcurves(df_lc , nbr_objects):
    '''
    Plots the Gaia light curves for a select number of sources.
    
    Parameter
    ---------
    df_lc : MultiIndex Lightcurve object
        MultiIndex Lightcurve object containing the Gaia light curves produced
        by `Gaia_get_lightcurve()`
    
    nbr_objects : int
        Number of sources to plot
        
        
    Returns
    -------
    None, just makes a figure.
    
    
    '''
    ## First get the ids/names of sources that have Gaia multi-epoch observations.
    object_ids = list(df_lc.data.index.levels[0]) # get list of objectids in multiIndex table

    fig = plt.figure(figsize=(12,4))
    plt.subplots_adjust(wspace=0.3)
    axs = [ fig.add_subplot(1,3,ii+1) for ii in range(3) ]
    cmap = plt.get_cmap("Spectral")

    for dd in object_ids[0:int(nbr_objects)]:
        try:

            for bb,band in enumerate(["G","BP","RP"]):

                this_tab = df_lc.data.loc[dd,:,"Gaia {}".format(band.lower()),:].reset_index(inplace=False)
                t = Time(this_tab["time"] , format="mjd") # convert to time object
                #axs[bb].plot(t.mjd , this_tab["flux"] , "-" , linewidth=1 , markersize=0.1)
                axs[bb].errorbar(t.mjd , this_tab["flux"] , yerr=this_tab["err"] , fmt="-o",linewidth=0.5 , markersize=3 , label="{}".format(dd))
        except:
            pass

    for ii in range(3):
        axs[ii].set_title(np.asarray(["G","BP","RP"])[ii])
        axs[ii].legend(fontsize=6 , ncol=3)
        axs[ii].set_xlabel("MJD (Days)" , fontsize=10)
        axs[ii].set_ylabel(r"Flux ($\mu$Jy)", fontsize=10)

    plt.show()
    
    return(True)
def Gaia_extract_median_photometry(gaia_table):
    '''
    Extract the median photometry from a Gaia table produced by `Gaia_retrieve_median_photometry`.
    
    Parameter
    ---------
    gaia_table : astropy table
        Gaia table including photometry produced by `Gaia_retrieve_median_photometry`.
        
        
    Returns
    --------
    Astropy table containing median photometry.
    
    '''
    
    # Once we matched the objects, we have to extract the photometry for them. Here we extract
    # the mean photometry (later we will do the time series).
    # Note that the fluxes are in e/s, not very useful. However, there are magnitudes (what unit??) but without errors.
    # We can get the errors from the flux errors?
    # Also note that we should include the source_id in order to search for epoch photometry

    ## Define keys (columns) that will be used later. Also add wavelength in angstroms for each filter
    other_keys = ["source_id","phot_g_n_obs","phot_rp_n_obs","phot_bp_n_obs","input_object_name"] # some other useful info
    mag_keys = ["phot_bp_mean_mag" , "phot_g_mean_mag" , "phot_rp_mean_mag"]
    magerr_keys = ["phot_bp_mean_mag_error" , "phot_g_mean_mag_error" , "phot_rp_mean_mag_error"]
    flux_keys = ["phot_bp_mean_flux" , "phot_g_mean_flux" , "phot_rp_mean_flux"]
    fluxerr_keys = ["phot_bp_mean_flux_error" , "phot_g_mean_flux_error" , "phot_rp_mean_flux_error"]
    mag_lambda = ["5319.90" , "6735.42" , "7992.90"]

    ## Get photometry. Note that this includes only objects that are 
    # matched to the catalog. We have to add the missing ones later.
    _phot = gaia_table[mag_keys]
    _err = hstack( [ 2.5/np.log(10) * gaia_table[e]/gaia_table[f] for e,f in zip(fluxerr_keys,flux_keys) ] )
    gaia_phot2 = hstack( [_phot , _err] )

    ## Clean up (change units and column names)
    _ = [gaia_phot2.rename_column(f,m) for m,f in zip(magerr_keys,fluxerr_keys)]
    for key in magerr_keys:
        gaia_phot2[key].unit = "mag"
    gaia_phot2["input_object_name"] = gaia_table["input_object_name"].copy()

    ## Add Some other useful information
    for key in other_keys:
        gaia_phot2[key] = gaia_table[key]

    gaia_phot = gaia_phot2.copy()

      
    return(gaia_phot)
