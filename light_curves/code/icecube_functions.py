## Functions related to IceCube matching
import os
import zipfile

import astropy.units as u
import numpy as np
import pandas as pd
import wget
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, vstack
from tqdm import tqdm

from data_structures import MultiIndexDFObject


def icecube_get_lightcurve(coords_list, labels_list, object_names , icecube_select_topN , path , verbose):
    '''
    Extracts IceCube Neutrino events for a given source position and saves it into a lightcurve
    Pandas MultiIndex object.
    This is the MAIN function.
    
    Parameters
    ----------
    coords_list : list of Astropy SkyCoord objects
        List of coordinates of the sources
    
    labels_list : list of str
        List of labels for each soruce
        
    object_names : list of str
        List of unique names for each source
        
    icecube_select_topN : int
        Number of top events to 
        
    path : str
        Path to temporary directory where to save the downloaded IceCube data.
        
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
    
    
    Returns
    --------
    IceCube Neutrino events in a Light curve Pandas MultiIndex object for all the input sources.
    
    '''
    
    ## DOWNLOAD ##
    # Downloads the IceCube data (currently from my personal Caltech Box directory) and
    # unzipps it. Only does it if the files have not yet been downloaded. The total file
    # size is about 30MB (zipped) and 110MB unzipped.
    icecube_download_data(path = path , verbose = verbose)
    
    ## LOAD ICECUBE CATALOG ###
    # This loads the IceCube catalog, which is dispersed in different files.
    # Each file has a list of events with their energy, time, and approximate direction.
    icecube_events , _ = icecube_get_catalog(path = path , verbose = verbose)

    # sort by Neutrino energy that way it is easier to get the top N events.
    icecube_events.sort(keys="energy_logGeV" , reverse=True)
    
    
    ### MATCH OBJECTS ###
    # Here we match the objects to the IceCube catalog to extract the N highest energy events close
    # to the objects' coordinates. We also want to include the errors in position of the IceCube
    # events.

    ## Top N (in energy) events to selected
    #icecube_select_topN = 3

    ## create SkyCoord objects from event coordinates
    c2 = SkyCoord(icecube_events["ra"], icecube_events["dec"], unit="deg", frame='icrs')

    ## Match
    icecube_matches = []
    icecube_matched = []
    ii = 0
    df_lc = MultiIndexDFObject()
    for cc,coord in enumerate(tqdm(coords_list)):

        # get all distances
        dist_angle =  coord.separation(c2)

        # make selection: here we have to also include errors on the
        # angles somehow.
        sel = np.where( (dist_angle.to(u.degree).value - icecube_events["AngErr"]) <= 0.0)[0]

        # select the top N events in energy. Note that we already sorted the table
        # by energy_logGeV. Hence we only have to pick the top N here.
        if len(sel) < icecube_select_topN:
            this_topN = len(sel)
        else:
            this_topN = icecube_select_topN * 1

        if len(sel) > 0:
            icecube_matches.append(icecube_events[sel[0:this_topN]])
            icecube_matches[ii]["Ang_match"] = dist_angle.to(u.degree).value[sel[0:this_topN]]
            icecube_matches[ii]["Ang_match"].unit = u.degree
            icecube_matched.append(cc)

            ii += 1


        else:
            pass # no match found
            if verbose > 0: print("No match found.")


    ## ADD TO LIGHTCURVE OBJECT ####
    ii = 0
    for cc,coord in enumerate(tqdm(coords_list)):
        lab = labels_list[cc]
        if cc in icecube_matched:
            ## Create single instance
            dfsingle = pd.DataFrame(
                                    dict(flux=np.asarray(icecube_matches[ii]["energy_logGeV"]), # in log GeV
                                     err=np.repeat(0,len(icecube_matches[ii])), # in mJy
                                     time=np.asarray(icecube_matches[ii]["mjd"]), # in MJD
                                     objectid=np.repeat(cc+1, len(icecube_matches[ii])),label=lab,
                                     band="IceCube"
                                        )
                        ).set_index(["objectid", "label", "band", "time"])

            ## Append
            df_lc.append(dfsingle)

            ii += 1
            
    if verbose > 0: print("IceCube Matched and added to lightcurve object.")
    
    return(df_lc)
    
    
def icecube_get_catalog(path , verbose):
    '''
    Creates the combined IceCube catalog based on the yearly catalog.
    
    Parameters
    -----------
    path : str
        Path to the directory where the icecube catalogs are saved.
                
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
    
    Returns
    --------
    (EVENTS , event_names) : (table , list of str) 
        Returns a catalog with events with columns ["mjd","energy_logGeV","AngErr","ra","dec","az","zen"].
        Returns also a list of the file names of the IceCube event file names (for convenience)
    '''
    
    event_names = ["IC40_exp.csv",
                    "IC59_exp.csv",
                    "IC79_exp.csv",
                    "IC86_III_exp.csv",
                    "IC86_II_exp.csv",
                    "IC86_IV_exp.csv",
                    "IC86_I_exp.csv",
                    "IC86_VII_exp.csv",
                    "IC86_VI_exp.csv",
                    "IC86_V_exp.csv"
                  ]
    
    EVENTS = Table(names=["mjd","energy_logGeV","AngErr","ra","dec","az","zen"] ,
                   units=[u.d , u.electronvolt*1e9 , u.degree , u.degree , u.degree , u.degree , u.degree ])
    for event_name in event_names:
        if verbose > 0: print("Loading: ", event_name)
        tmp = ascii.read(os.path.join(path , "icecube_10year_ps" , "events" , event_name))
        tmp.rename_columns(names=tmp.keys() , new_names=EVENTS.keys() )

        EVENTS = vstack([EVENTS , tmp])
    if verbose > 0: print("done")
    return(EVENTS , event_names)



def icecube_download_data(path , verbose):
    '''
    Download and unzipps the IceCube data (approx. 40MB zipped, 120MB unzipped). Directly
    downloaded from the IceCube webpage:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    
    Parameters
    ----------
    path : str
        Path to temporary directory where to save the downloaded IceCube data.
        
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full
        
    Returns
    -------
    Unzipped IceCube event tables in the `path` directory.
    
    
    '''

    ## Download
    if not os.path.exists(os.path.join(path , "icecube_events.zip")):

        if verbose > 0: print("Downloading IceCube data to {} | ".format(path) , end=" ")
        file_url = "http://icecube.wisc.edu/data-releases/20210126_PS-IC40-IC86_VII.zip"
        wget.download(url = file_url , out = path)

        ## Unzip
        if verbose > 0: print("Unzipping IceCube data | ", end=" ")
        with zipfile.ZipFile(os.path.join(path , "20210126_PS-IC40-IC86_VII.zip"), 'r') as zip_ref:
            zip_ref.extractall(path)

        if verbose > 0: print("Done.")

    else:
        if verbose > 0: print("Data already downloaded.")
    
    return(True)
        
    