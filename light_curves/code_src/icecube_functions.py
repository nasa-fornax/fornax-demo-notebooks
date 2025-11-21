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

from data_structures import MultiIndexDFObject

DATA_PATH = os.path.dirname(os.path.dirname(__file__)) + \
    "/data/"  # absolute path to light_curves/data/


def icecube_get_lightcurves(sample_table, *, icecube_select_topN=3, max_search_radius=2.0):
    '''
    Extracts IceCube Neutrino events for a given source position.
    This is the MAIN function.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.
    icecube_select_topN : int
        Maximum number of events to return for a single object in sample_table. The brightest events
        within the match radius will be returned.

    max_search_radius : float
        Maximum radius (degrees) to look for matches in IceCube. Actual match radius will not exceed the
        IceCube error of an individual event. Beware that setting this to a high number can cause
        the code to look through a large number of potential matches for each object, which may
        impact performance.

    Returns
    --------
     df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time]. The resulting internal pandas DataFrame
        contains the following columns:
        
            flux : float
                Neutrino event energy expressed as log10(GeV).
            err : float
                Placeholder uncertainty column (always 0.0 for IceCube events).
            time : float
                Event time in modified Julian date (MJD).
            objectid : int
                Input sample object identifier.
            band : str
                Always the string "IceCube".
            label : str
                Literature label associated with each source.
    '''
    max_search_radius = max_search_radius * u.deg

    # Downloads the IceCube data and unzipps it. Only does it if the files have not yet been downloaded. The
    # total file size is about 30MB (zipped) and 110MB unzipped.
    icecube_download_data(verbose=False)

    # This loads the IceCube catalog, which is dispersed in different files.
    # Each file has a list of events with their energy, time, and approximate direction.
    icecube_events, _ = icecube_get_catalog(verbose=False)

    # create SkyCoord objects from icecube event coordinates
    icecube_skycoords = SkyCoord(
        icecube_events["ra"], icecube_events["dec"], unit="deg", frame='icrs')

    # here are the skycoords from mysample defined above
    mysample_skycoords = sample_table['coord']

    # Match
    idx_mysample, idx_icecube, d2d, d3d = icecube_skycoords.search_around_sky(
        mysample_skycoords, max_search_radius)

    # need to filter reponse based on position error circles
    # really want d2d to be less than the error circle of icecube = icecube_events["AngErr"] in degrees
    filter_arr = d2d < icecube_events["AngErr"][idx_icecube]
    filter_idx_mysample = idx_mysample[filter_arr]
    filter_idx_icecube = idx_icecube[filter_arr]
    filter_d2d = d2d[filter_arr]

    # keep these matches together with objectid and lebal as new entries in the df.
    obid_match = sample_table['objectid'][filter_idx_mysample]
    label_match = sample_table['label'][filter_idx_mysample]
    time_icecube = icecube_events['mjd'][filter_idx_icecube]
    flux_icecube = icecube_events['energy_logGeV'][filter_idx_icecube]

    # save the icecube info in correct format for the rest of the data
    icecube_df = pd.DataFrame({'flux': flux_icecube,
                               'err': np.zeros(len(obid_match)),
                               'time': time_icecube,
                               'objectid': obid_match,
                               'label': label_match,
                               'band': "IceCube"})

    # sort by Neutrino energy that way it is easier to get the top N events.
    icecube_df = icecube_df.sort_values(['objectid', 'flux'], ascending=[True, False])

    # now can use a groupby to only keep the top N (by GeV flux) icecube matches for each object
    filter_icecube_df = icecube_df.groupby('objectid').head(
        icecube_select_topN).reset_index(drop=True)

    # put the index in to match with df_lc
    filter_icecube_df.set_index(["objectid", "label", "band", "time"], inplace=True)

    return (MultiIndexDFObject(data=filter_icecube_df))


def icecube_get_catalog(path=DATA_PATH, verbose=False):
    '''
    Creates the combined IceCube catalog based on the yearly catalogs.

    Parameters
    -----------
    path : str
        Download directory path
    verbose : bool
        Default False. Display extra info and warnings if true.

    Returns
    --------
    EVENTS : astropy.table.Table
        Combined event table with columns:
            mjd : float
                Event detection time (Modified Julian Date).
            energy_logGeV : float
                log10(Energy / GeV).
            AngErr : float
                Angular positional uncertainty in degrees.
            ra : float
                Right ascension of event direction (deg).
            dec : float
                Declination of event direction (deg).
            az : float
                Azimuth (deg).
            zen : float
                Zenith angle (deg).

    event_names : list of str
        Filenames of the event catalog files that were read and combined.
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
                          path=DATA_PATH, verbose=False):
    '''
    Download and unzipps the IceCube data (approx. 40MB zipped, 120MB unzipped). Directly
    downloaded from the IceCube webpage:
    https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/

    Parameters
    ----------
    url : str
        Icecube data URL
    path : str
        Destination directory for downloaded and unzipped data.
    verbose : bool
        Default False. Display extra info and warnings if true.

    Returns
    -------
    None
        Extracted data files are written to disk in `path`.

    '''

    file_path = os.path.join(path, os.path.basename(urlparse(url).path))

    # if the data has not already been downloaded
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
