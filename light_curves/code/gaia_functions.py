from astroquery.gaia import Gaia
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.table import Table

# We first define some function:
# 1. Gaia_retrieve_EPOCH_PHOTOMETRY: to retrieve the epoch photometry for a given Gaia (DR3) ID
# 2. Gaia_mk_lightcurves: to create light curves from the table downloaded with DataLink
#                         from the Gaia server using the function Gaia_retrieve_EPOCH_PHOTOMETRY()
# 3. Gaia_mk_MultibandTimeSeries: creates a MultibandTimeSeries object (not really used)
# 4. Gaia_mk_MultiIndex: creates a Pandas MultiIndex data frame, which can be added to the existing
#                        data frame containing the other observations.


## Define function to retrieve epoch photometry
def Gaia_retrieve_EPOCH_PHOTOMETRY(ids, verbose):
    """
    Function to retrieve EPOCH_PHOTOMETRY (or actually any) catalog product for Gaia
    entries using the DataLink. Note that the IDs need to be DR3 source_id and needs to be a list.
    
    Code fragments taken from: https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products
    
    Parameters
    ----------
    ids (int): List of Gaia DR3 source IDs (source_id). 
    
    Returns a dictionary (key = source_id) with a table of photometry as a function of time.
        
    """
    
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
    
    if verbose:
        print(f'The following Datalink products have been downloaded:')
        for dl_key in dl_keys:
            print(f' * {dl_key}')
    
    ## Extract the info
    prod_tab = dict() # Dictionary to save the light curves. The key is the source_id
    for dd in ids:
        if verbose: print("{}: ".format(dd) , end=" ")
        this_dl_key = 'EPOCH_PHOTOMETRY-Gaia DR3 {}.xml'.format(dd)
        if this_dl_key in datalink.keys():
            prod_tab[str(dd)] = datalink[this_dl_key][0].to_table()
            if verbose: print("found")
        else:
            pass
            if verbose: print("not found")
    
    return(prod_tab)

## Define function to extract light curve from product table.
def Gaia_mk_lightcurves(prod_tab):
    """
    This function creates light curves from the table downloaded with DataLink from the Gaia server.
    
    Parameters
    ----------
    prod_tab (astropy table): product table downloaded via datalink, produced by `Gaia_retrieve_EPOCH_PHOTOMETRY()`.
        
    
    Returns a dictionary (key = source_id) including a dictionary of light curves for bands "G", "BP", "RP". Each
        of them includes a time stamp (`time_jd` and `time_isot`) a magnitude (`mag`) and magnitude error (`magerr`).
    """
    
    bands = ["G","BP","RP"]
    output = dict()
    for ii,key in enumerate(list(prod_tab.keys()) ):
        print(key)
    
        output[str(key)] = dict()
        for band in bands:
            sel_band = np.where( (prod_tab[key]["band"] == band) & (prod_tab[key]["rejected_by_photometry"] == False) )[0]
            print("Number of entries for band {}: {}".format(band , len(sel_band)))
            
            time_jd = prod_tab[key][sel_band]["time"] + 2455197.5 # What unit???
            time_isot = Time(time_jd , format="jd").isot
            mag = prod_tab[key][sel_band]["mag"]
            magerr = 2.5/np.log(10) * prod_tab[key][sel_band]["flux_error"]/prod_tab[key][sel_band]["flux"]
            
            output[str(key)][band] = Table([time_jd , time_isot , mag , magerr] , names=["time_jd","time_isot","mag","magerr"] ,
                                           dtype = [float , str , float , float], units=[u.d , ""  , u.mag , u.mag])
            
    return(output)

## Function to add light curves into Multi-Band Time Series object
def Gaia_mk_MultibandTimeSeries(epoch_phot):
    
    """
    Creates MultibandTimeSeries object from epoch photometry lightcurves.
    NOT USED ANYMORE BECAUSE WE SWITCHED TO PANDAS MULTIINDEX ARRAY.
    
    Parameters
    ----------
    epoch_phot (dictionary): Epoch photometry light curve (see `Gaia_mk_lightcurves`)
    
    
    Returns a dictionary of MultibandTimeSeries light curves (for each source_id)
    
    """
    
    
    
    ## For each source, create a MultibandTimeSeries object and add the bands.
    bands = ["G","BP","RP"]
    out = dict()
    for key in epoch_phot.keys():
        
        # Initialize
        ts = MultibandTimeSeries()
        
        # Add bands
        for band in bands:
            ts.add_band(time = Time(epoch_phot[str(key)][band]["time_jd"] , format="jd") ,
                 data = epoch_phot[str(key)][band]["mag"],
                 band_name=band
                )
        out[key] = ts
        
    return(out)


def Gaia_mk_MultiIndex(data , gaia_phot , gaia_epoch_phot , verbose):
    '''
    Creates a MultiIndex Pandas Dataframe for the Gaia observations. Specifically, it 
    returns the epoch photometry as a function of time. For sources without Gaia epoch
    photometry, it just returns the mean photometry a epoch 2015-09-24T19:40:33.468, which
    is the average epoch of the observations of sources with multi-epoch photometry.
    
    Parameters
    ----------
    data (astropy table): the catalog with the source IDs and names (here: CLAGN)
    gaia_phot (dictionary): The Gaia mean photometry (will be linked by object ID in 'data' catalog)
    gaia_epoch_phot (dictionary): The Gaia epoch photometry (is a dictionary created by 'Gaia_mk_lightcurves()' function)
    verbose (int): verbosity level (0=silent)
    
    Returns a Pandas data frame with indices ("objectid","band","time"). The data frame contains flux and flux error,
    both in mJy. The output can be appended to another lightcurve Pandas data
    frame via df_lc_object.append(df_lc)
    '''

    for ii in range(len(data)):
        print("{} matched to: ".format( data["Object Name"][ii])  , end=" ")

        ## Check if this object has a Gaia light curve:

        # get Gaia source_id
        sel = np.where(data["Object Name"][ii] == gaia_phot["input_object_name"])[0]
        if len(sel) > 0:
            source_id = gaia_phot["source_id"][sel[0]]
            print(source_id , end=" ")

            if str(source_id) in gaia_epoch_phot.keys(): # Match to Gaia multi-epoch catalog
                print("Has Gaia epoch photometry")

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
                                 objectid=np.repeat(ii+1, len(y)),
                                 band="Gaia {}".format(band.lower())
                                                )
                                           ).set_index(["objectid", "band", "time"])

                    # add to table
                    try:
                        df_lc
                    except NameError:
                        #df_lc doesn't exist (yet)
                        df_lc = dfsingle.copy()
                    else:
                        #df_lc_gaia exists
                        df_lc = pd.concat([df_lc, dfsingle])

            else: # No match to Gaia multi-epoch catalog: use single epoch photometry
                print("No Gaia epoch photometry, append single epoch photometry ")

                for band in ["G","BP","RP"]:

                    # get data
                    t = Time("2015-09-24T19:40:33.468" , format="isot") # just random date: FIXME: NEED TO GET ACTUAL OBSERVATION TIME!
                    y = gaia_phot["phot_{}_mean_mag".format(band.lower())][sel]
                    dy = gaia_phot["phot_{}_mean_mag_error".format(band.lower())][sel]

                    # compute flux and flux error in mJy
                    y2 = 10**(-0.4*(y - 23.9))/1e3 # in mJy
                    dy2 = dy / 2.5 * np.log(10) * y2 # in mJy

                    # create single instance
                    dfsingle = pd.DataFrame(
                                dict(flux=np.asarray(y2), # in mJy
                                 err=np.asarray(dy2), # in mJy
                                 time=t.mjd, # in MJD
                                 #objectid=gaia_phot["input_object_name"][sel],
                                 objectid=np.repeat(ii+1, len(y)),
                                 band="Gaia {}".format(band.lower())
                                    )
                    ).set_index(["objectid", "band", "time"])

                    # add to table
                    try:
                        df_lc
                    except NameError:
                        #df_lc doesn't exist (yet)
                        df_lc = dfsingle.copy()
                    else:
                        #df_lc_gaia exists
                        df_lc = pd.concat([df_lc, dfsingle])

        else: # no match to Gaia
            print("none")
            
    return(df_lc)