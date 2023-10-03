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


def icecube_get_lightcurve(coords_list, labels_list, icecube_select_topN=3, verbose=False):
    '''
    Extracts IceCube Neutrino events for a given source position and saves it into a lightcurve
    Pandas MultiIndex object.
    This is the MAIN function.

    Parameters
    ----------
    coords_list : list of Astropy SkyCoord objects
        List of (id,coordinates) tuples of the sources

    labels_list : list of str
        List of labels for each soruce

    icecube_select_topN : int
        Number of top events to

    verbose : bool
        Default False. Display extra info and warnings if true.

    Returns
    --------
    MultiIndexDFObject: IceCube Neutrino events for all the input sources.

    '''

    # Downloads the IceCube data and unzipps it. Only does it if the files have not yet been downloaded. The
    # total file size is about 30MB (zipped) and 110MB unzipped.
    icecube_download_data(verbose=verbose)

    # This loads the IceCube catalog, which is dispersed in different files.
    # Each file has a list of events with their energy, time, and approximate direction.
    icecube_events, _ = icecube_get_catalog(verbose=verbose)

    # sort by Neutrino energy that way it is easier to get the top N events.
    icecube_events.sort(keys="energy_logGeV", reverse=True)

    # create SkyCoord objects from event coordinates
    c2 = SkyCoord(icecube_events["ra"], icecube_events["dec"], unit="deg", frame='icrs')

    # Match
    icecube_matches = []
    icecube_matched = []
    df_lc = MultiIndexDFObject()

    for index, (objectid, coord) in enumerate(tqdm(coords_list)):

        # get all distances
        dist_angle = coord.separation(c2)

        # make selection: here we have to also include errors on the
        # angles somehow.
        sel = np.where((dist_angle.to(u.degree).value - icecube_events["AngErr"]) <= 0.0)[0]

        # select the top N events in energy. Note that we already sorted the table
        # by energy_logGeV. Hence we only have to pick the top N here.
        if len(sel) < icecube_select_topN:
            this_topN = len(sel)
        else:
            this_topN = icecube_select_topN * 1

        if len(sel) > 0:
            icecube_matches.append(icecube_events[sel[0:this_topN]])
            icecube_matches[index]["Ang_match"] = dist_angle.to(u.degree).value[sel[0:this_topN]]
            icecube_matches[index]["Ang_match"].unit = u.degree
            icecube_matched.append(objectid)

        else:
            if verbose:
                print("No match found.")

    # Add to lightcurve object
    for index, (objectid, coord) in enumerate(tqdm(coords_list)):
        label = labels_list[index]
        if objectid in icecube_matched:
            # Create single instance. flux in log GeV, error is N/A & set to 0, time is MJD
            dfsingle = pd.DataFrame({'flux': np.asarray(icecube_matches[index]["energy_logGeV"]),
                                     'err': np.repeat(0, len(icecube_matches[index])),
                                     'time': np.asarray(icecube_matches[index]["mjd"]),
                                     'objectid': np.repeat(objectid, len(icecube_matches[index])),
                                     'label': label,
                                     'band': "IceCube"}).set_index(["objectid", "label", "band", "time"])

            df_lc.append(dfsingle)

    if verbose:
        print("IceCube Matched and added to lightcurve object.")

    return (df_lc)


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
