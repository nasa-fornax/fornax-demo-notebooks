# Functions related to IceCube matching
import numpy as np
import pandas as pd
from astroquery.heasarc import Heasarc

from data_structures import MultiIndexDFObject


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
    icecube_tbl = icecube_query_catalog(sample_table, max_search_radius=max_search_radius)

    # save the icecube info in correct format for the rest of the data
    icecube_df = pd.DataFrame({'flux': icecube_tbl['cat_event_energy'],
                               'err': np.zeros(len(icecube_tbl)),
                               'time': icecube_tbl['cat_time'],
                               'objectid': icecube_tbl['samp_objectid'],
                               'label': icecube_tbl['samp_label'],
                               'band': "IceCube"})

    # sort by Neutrino energy that way it is easier to get the top N events.
    icecube_df = icecube_df.sort_values(['objectid', 'flux'], ascending=[True, False])

    # now can use a groupby to only keep the top N (by GeV flux) icecube matches for each object
    filter_icecube_df = icecube_df.groupby('objectid').head(
        icecube_select_topN).reset_index(drop=True)

    # put the index in to match with df_lc
    filter_icecube_df.set_index(["objectid", "label", "band", "time"], inplace=True)

    return (MultiIndexDFObject(data=filter_icecube_df))


def icecube_query_catalog(sample_table, *, max_search_radius):
    '''
    Cross match the input sample against the IceCube All-Sky Point-Source Neutrino Events Catalog (2008-2018)
    using a TAP query with a table upload via `astroquery.heasarc.Heasarc`.

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
    max_search_radius : float
        Maximum radius (degrees) to look for matches in IceCube. Actual match radius will not exceed the
        IceCube error of an individual event. Beware that setting this to a high number can cause
        the code to look through a large number of potential matches for each object, which may
        impact performance.

    Returns
    -------
    astropy.table.Table
        One row per matched IceCube event, with columns `samp_objectid`, `samp_label`, `cat_time` [MJD],
        and `cat_event_energy` [log10(GeV)].
    '''
    # Put sample_table into the format TAP expects.
    upload_table = sample_table['objectid', 'label']
    upload_table['ra'] = sample_table['coord'].ra.deg
    upload_table['dec'] = sample_table['coord'].dec.deg

    # ADQL to find IceCube events within max_search_radius of our sample targets.
    # AND DISTANCE removes events with an error radius less than the distance between the event and our sample target.
    icecube_catalog = "icecubepsc"
    query = f"""
        SELECT samp.objectid, samp.label, cat.time, cat.event_energy
        FROM {icecube_catalog} cat, tap_upload.sample_table samp
        WHERE 1=CONTAINS(POINT('ICRS', samp.ra, samp.dec), CIRCLE('ICRS', cat.ra, cat.dec, {max_search_radius}))
        AND DISTANCE(POINT('ICRS', samp.ra, samp.dec), POINT('ICRS', cat.ra, cat.dec)) < cat.error_radius
    """

    # Query HEASARC for IceCube events within our sample target areas.
    # This will prepend the catalog name ("samp" or "cat") to the column names, eg, "time" -> "cat_time".
    icecube_tbl = Heasarc.query_tap(query, uploads={"sample_table": upload_table}).to_table()

    # We want log10 of the event energy.
    icecube_tbl['cat_event_energy'] = np.log10(icecube_tbl['cat_event_energy'])

    return icecube_tbl
