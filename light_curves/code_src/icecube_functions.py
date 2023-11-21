# Functions related to IceCube matching
import os
import zipfile

from urllib.parse import urlparse
from urllib.request import urlretrieve

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, vstack
from tqdm import tqdm

from data_structures import MultiIndexDFObject


def Icecube_get_lightcurve(sample_table, icecube_select_topN=3):
    '''
    Extracts IceCube Neutrino events for a given source position and saves it into a lightcurve
    Pandas MultiIndex object.
    This is the MAIN function.

    Parameters
    ----------
    sample_table : Astropy Table
        main source catalog with coordinates, labels, and objectids

    icecube_select_topN : int
        Number of top events to

 
    Returns
    --------
    MultiIndexDFObject: IceCube Neutrino events for all the input sources.

    '''

    # Downloads the IceCube data and unzipps it. Only does it if the files have not yet been downloaded. The
    # total file size is about 30MB (zipped) and 110MB unzipped.
    icecube_download_data(verbose=False)

    # This loads the IceCube catalog, which is dispersed in different files.
    # Each file has a list of events with their energy, time, and approximate direction.
    icecube_events, _ = icecube_get_catalog(verbose=False)

    # create SkyCoord objects from icecube event coordinates
    icecube_skycoords = SkyCoord(icecube_events["ra"], icecube_events["dec"], unit="deg", frame='icrs')

    #here are the skycoords from mysample defined above
    mysample_skycoords = sample_table['coord']

    #Match
    idx_mysample, idx_icecube, d2d, d3d = icecube_skycoords.search_around_sky(mysample_skycoords, 1*u.deg)

    #need to filter reponse based on position error circles
    #really want d2d to be less than the error circle of icecube = icecube_events["AngErr"] in degrees
    filter_arr = d2d < icecube_events["AngErr"][idx_icecube]
    filter_idx_mysample = idx_mysample[filter_arr]
    filter_idx_icecube = idx_icecube[filter_arr]
    filter_d2d = d2d[filter_arr]

    #keep these matches together with objectid and lebal as new entries in the df.
    obid_match = sample_table['objectid'][filter_idx_mysample]
    label_match = sample_table['label'][filter_idx_mysample]
    time_icecube= icecube_events['mjd'][filter_idx_icecube]
    flux_icecube = icecube_events['energy_logGeV'][filter_idx_icecube]

    #save the icecube info in correct format for the rest of the data
    icecube_df = pd.DataFrame({'flux': flux_icecube, 
                               'err': np.zeros(len(obid_match)), 
                               'time': time_icecube, 
                               'objectid': obid_match, 
                               'label': label_match, 
                               'band': "IceCube"})

    # sort by Neutrino energy that way it is easier to get the top N events.
    icecube_df = icecube_df.sort_values(['objectid', 'flux'], ascending=[True, False])
    
    #now can use a groupby to only keep the top N (by GeV flux) icecube matches for each object
    filter_icecube_df = icecube_df.groupby('objectid').head(icecube_select_topN).reset_index(drop=True)

    #put the index in to match with df_lc
    filter_icecube_df.set_index(["objectid", "label", "band", "time"], inplace = True)
    
    return (filter_icecube_df)


def icecube_get_catalog(path="data", verbose=False):
    '''
    Creates the combined IceCube catalog based on the yearly catalog.

    Parameters
    -----------
    path : str
        Download directory path
    verbose : bool
        Default False. Display extra info and warnings if true.

    Returns
    --------
    (EVENTS , event_names) : (table , list of str)
        Returns a catalog of events with columns: mjd, energy_logGeV, AngErr, ra, dec, az, zen
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

    EVENTS = Table(names=["mjd", "energy_logGeV", "AngErr", "ra", "dec", "az", "zen"],
                   units=[u.d, u.GeV, u.degree, u.degree, u.degree, u.degree, u.degree])

    for event_name in event_names:
        if verbose:
            print("Loading: ", event_name)
        tmp = ascii.read(os.path.join(path, "icecube_10year_ps", "events", event_name))
        tmp.rename_columns(names=tmp.keys(), new_names=EVENTS.keys())

        EVENTS = vstack([EVENTS, tmp])
    if verbose:
        print("done")
    return (EVENTS, event_names)


def icecube_download_data(url="http://icecube.wisc.edu/data-releases/20210126_PS-IC40-IC86_VII.zip",
                          path="data", verbose=False):
    '''
    Download and unzipps the IceCube data (approx. 40MB zipped, 120MB unzipped). Directly
    downloaded from the IceCube webpage:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/

    Parameters
    ----------
    url : str
        Icecube data URL
    path : str
        Path to download data to.
    verbose : bool
        Default False. Display extra info and warnings if true.

    Returns
    -------
    Unzipped IceCube event tables in the data directory.

    '''

    file_path = os.path.join(path, os.path.basename(urlparse(url).path))

    if not os.path.exists(file_path):

        # Download
        if verbose:
            print(f"Downloading IceCube data to {file_path}.")

        _ = urlretrieve(url, file_path)

        # Unzip
        if verbose:
            print("Unzipping IceCube data.")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        if verbose:
            print("Done.")

    else:
        if verbose:
            print(f"Data is already downloaded, see {path}")
