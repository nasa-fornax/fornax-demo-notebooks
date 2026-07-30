import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo
from astropy.table import vstack
from astroquery.heasarc import Heasarc
from tqdm.auto import tqdm

from data_structures import MultiIndexDFObject


def make_hist_error_radii(missioncat):
    """
    Plots a histogram of error radii from a HEASARC catalog

    example calling sequences:
    resulttable = make_hist_error_radii('FERMIGTRIG')


    Parameters
    ----------
    missioncat : str
        single catalog within HEASARC to grab error radii values  Must be one of the catalogs listed here:
            https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html#getting-list-of-available-missions
    Returns
    -------
    heasarcresulttable : astropy.table.Table
        Table of the first 5000 rows containing:

            name : str
                Mission source identifier.
            ra : float (deg)
                Right Ascension (ICRS).
            dec : float (deg)
                Declination (ICRS).
            error_radius : float (deg)
                Positional localization radius provided by the mission.

    Notes
    -----
    This helper function is intended for exploratory analysis to understand
    the typical localization precision of different HEASARC catalogs, which
    can vary from arcminutes to many degrees depending on the mission.

    """
    # need to know the distribution of error radii for the catalogs of interest
    # this will inform the ligh curve query, as we are not interested in
    # error radii which are 'too large' so we need a way of defining what that is.
    # leaving this code here in case user wants to change the cutoff error radii
    # based on their science goals.  It is not currently used anywhere in the code

    # get the pyvo HEASARC service.
    heasarc_tap = pyvo.regsearch(servicetype='tap', keywords=['heasarc'])[0]

    # simple query to select sources from that catalog
    heasarcquery = f"""
        SELECT TOP 5000 cat.name, cat.ra, cat.dec, cat.error_radius
        FROM {missioncat} as cat
         """
    heasarcresult = heasarc_tap.service.run_sync(heasarcquery)

    #  Convert the result to an Astropy Table
    heasarcresulttable = heasarcresult.to_table()

    # make a histogram
    # zoom in on the range of interest
    # error radii are in units of degrees
    plt.hist(heasarcresulttable["error_radius"], bins=30, range=[0, 10])

    # in case anyone wants to look further at the data
    return heasarcresulttable


def heasarc_get_lightcurves(sample_table, *, catalog_constraints={"FERMIGTRIG": 1.0, "SAXGRBMGRB": 3.0}):
    """
    Search selected HEASARC catalogs for events that spatially coincide with entries in the input sample_table.

    The catalogs currently supported are: "FERMIGTRIG", "SAXGRBMGRB", "icecubepsc".
    Results are treated as **single events**, not light curves.
    Returned flux values for "FERMIGTRIG" and "SAXGRBMGRB" should not be physically interpreted;
    only the timestamps and catalog labels carry scientific meaning.

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

    catalog_constraints : dict
        Dictionary specifying which HEASARC catalog(s) to search and the source matching constraints for each.

        Key : str
            "FERMIGTRIG", "SAXGRBMGRB", or "icecubepsc".

        Value : float or dict

            For "FERMIGTRIG" or "SAXGRBMGRB": (float) the maximum allowed error radius in degrees.
                For example: `{"FERMIGTRIG": 1.0}`.
                Recommended values are about 1.0 to 3.0. Very large values have the potential to
                match many target objects.

            For "icecubepsc": a dict with keys `select_topN` and `max_search_radius`.
                For example: `{"icecubepsc": {"select_topN": 3, "max_search_radius": 2.0}}`.
                "select_topN" : int
                    Maximum number of events to return for a single object in sample_table.
                    The brightest events (highest energies) within the match radius will be returned.
                "max_search_radius" : float
                    Maximum radius (degrees) to look for matches in IceCube. Actual match radius
                    will not exceed the IceCube error of an individual event. Beware that setting
                    this to a high number can cause the code to look through a large number of
                    potential matches for each object, which may impact performance.

    Returns
    -------
     df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time]. The resulting internal
        pandas DataFrame contains the following columns:

            flux : float
                For "FERMIGTRIG" and "SAXGRBMGRB": A placeholder value that will be used to
                    mark a vertical line on the plot.
                For "icecubepsc": Neutrino event energy, expressed as log10(GeV).
            err : float
                For "FERMIGTRIG" and "SAXGRBMGRB": A placeholder value.
                For "icecubepsc": Always 0.0.
            time : float
                Event time in modified Julian date (MJD).
            objectid : int
                Input sample object identifier.
            band : str
                'FERMIGTRIG', 'SAXGRBMGRB', or (for "icecubepsc") 'IceCube'.
            label : str
                Literature label associated with each source.
    """
    _validate_catalog_constraints(catalog_constraints)

    # Prepping sample_table with float R.A. and DEC column instead of SkyCoord mixin for TAP upload

    # set the maximum number of rows in sample_table that can be uploaded in one go.
    nchunk = 50000

    upload_table = sample_table['objectid', 'label']
    upload_table['ra'] = sample_table['coord'].ra.deg
    upload_table['dec'] = sample_table['coord'].dec.deg

    # setup to store the data
    df_lc = MultiIndexDFObject()

    for heasarc_cat, constraint in tqdm(catalog_constraints.items()):
        print('working on mission', heasarc_cat)

        if heasarc_cat == "icecubepsc":
            query = _handle_icecube(heasarc_cat, max_search_radius=constraint["max_search_radius"])
        else:
            query = _handle_fermi_sax(heasarc_cat, max_error_radius=constraint)

        # instead of uploading upload_table in one go, split it
        # into several tables with a maximum size of nchunk.
        # upload_tables: is a list of tables to be uploaded one at a time.
        # We use groupby in pandas to do the split
        ids = [g.index.values for k, g in upload_table.to_pandas().groupby(
            np.arange(len(upload_table)) // nchunk)]
        upload_tables = [upload_table[idd] for idd in ids]

        # hresult: is a list of query results corresponding to upload_tables.
        # hresulttable: is the stacked table of all the results from individual calls.
        hresult = [Heasarc.query_tap(query, uploads={'mytable': upload_table})
                   for upload_table in upload_tables]
        hresulttable = vstack([hr.to_table() for hr in hresult])

        if heasarc_cat == "icecubepsc":
            df_heasarc = _handle_icecube(heasarc_cat, hresulttable=hresulttable,
                                         select_topN=constraint["select_topN"])
        else:
            df_heasarc = _handle_fermi_sax(heasarc_cat, hresulttable=hresulttable)

        # Append to existing MultiIndex light curve object
        df_lc.append(df_heasarc)

    return df_lc


def _handle_fermi_sax(heasarc_cat, *, max_error_radius=None, hresulttable=None):
    """
    Build the TAP query for `heasarc_cat` if `max_error_radius` is given, otherwise process the query result `hresulttable`.

    Parameters
    ----------
    heasarc_cat : str
        The catalog name, "FERMIGTRIG" or "SAXGRBMGRB". Used in the query, and as the `band` value
        when building the DataFrame.
    max_error_radius : float, optional
        Maximum allowed error radius, in degrees. If given, the query string is returned
        immediately and `hresulttable` is ignored.
    hresulttable : astropy.table.Table, optional
        Query result table to process into a DataFrame. Only used when `max_error_radius` is not
        given.

    Returns
    -------
    str or pandas.DataFrame
        The ADQL query string, if `max_error_radius` is given, otherwise the light curve DataFrame
        processed from `hresulttable`.
    """
    if max_error_radius is not None:
        # SELECT cat.name, cat.ra, cat.dec, cat.error_radius, cat.time AS time,
        query = f"""
            SELECT cat.time AS time, mt.objectid AS objectid, mt.label AS label
            FROM {heasarc_cat} cat, tap_upload.mytable mt
            WHERE cat.error_radius < {max_error_radius}
            AND CONTAINS(POINT('ICRS',mt.ra,mt.dec), CIRCLE('ICRS',cat.ra,cat.dec,cat.error_radius))=1
        """
        return query

    # really just need to mark this spot with a vertical line in the plot, it's not actually a light curve
    # so making up a flux and an error, but the time stamp and mission are the real variables we want to keep
    df_heasarc = pd.DataFrame(dict(flux=np.full(len(hresulttable), 0.1), err=np.full(len(hresulttable), 0.1),
                                   time=hresulttable['time'], objectid=hresulttable['objectid'],
                                   band=np.full(len(hresulttable), heasarc_cat),
                                   label=hresulttable['label']))

    return df_heasarc.set_index(["objectid", "label", "band", "time"])


def _handle_icecube(heasarc_cat, *, max_search_radius=None, hresulttable=None, select_topN=None):
    """
    Build the TAP query for IceCube if `max_search_radius` is given, otherwise process the query result `hresulttable`.

    Parameters
    ----------
    heasarc_cat : str
        The catalog name, "icecubepsc".
    max_search_radius : float, optional
        Maximum radius (degrees) to look for matches in IceCube. Actual match radius will not exceed the
        IceCube error of an individual event. Beware that setting this to a high number can cause
        the code to look through a large number of potential matches for each object, which may
        impact performance.
        If given, the query string is returned immediately and `hresulttable` is ignored.
    hresulttable : astropy.table.Table, optional
        Query result table to process into a DataFrame. Only used when `max_search_radius` is not
        given.
    select_topN : int, optional
        Maximum number of events to return for a single object in sample_table. The brightest events
        within the match radius will be returned. Only used when `hresulttable` is given.

    Returns
    -------
    str or pandas.DataFrame
        The ADQL query string, if `max_search_radius` is given, otherwise the light curve DataFrame
        processed from `hresulttable` and `select_topN`.
    """
    if max_search_radius is not None:
        # ADQL to find IceCube events within max_search_radius of our sample targets.
        # AND DISTANCE removes events with an error radius less than the distance between the event and our sample target.
            # SELECT mt.objectid, mt.label, cat.time, cat.event_energy
        query = f"""
            SELECT cat.event_energy AS event_energy, cat.time AS time, mt.objectid AS objectid, mt.label AS label
            FROM {heasarc_cat} cat, tap_upload.mytable mt
            WHERE 1=CONTAINS(POINT('ICRS', mt.ra, mt.dec), CIRCLE('ICRS', cat.ra, cat.dec, {max_search_radius}))
            AND DISTANCE(POINT('ICRS', mt.ra, mt.dec), POINT('ICRS', cat.ra, cat.dec)) < cat.error_radius
            """
        return query

    # We want log10 of the event energy.
    hresulttable['event_energy'] = np.log10(hresulttable['event_energy'])

    # Save the icecube info in correct format for the rest of the data.
    icecube_df = pd.DataFrame({'flux': hresulttable['event_energy'],
                               'err': np.zeros(len(hresulttable)),
                               'time': hresulttable['time'],
                               'objectid': hresulttable['objectid'],
                               'label': hresulttable['label'],
                               'band': "IceCube"})

    # Sort by neutrino energy and keep only the top N IceCube matches for each object.
    icecube_df = icecube_df.sort_values(['objectid', 'flux'], ascending=[True, False])
    df_heasarc = icecube_df.groupby('objectid').head(select_topN).reset_index(drop=True)

    return df_heasarc.set_index(["objectid", "label", "band", "time"])


def _validate_catalog_constraints(catalog_constraints):
    """
    Validate that catalog_constraints conforms to the requirements documented in
    heasarc_get_lightcurves(). Raises ValueError if it does not.

    Parameters
    ----------
    catalog_constraints : dict
        The catalog_constraints argument to validate.
    """
    icecube_cat = "icecubepsc"
    error_radius_cats = {"FERMIGTRIG", "SAXGRBMGRB"}
    valid_keys = error_radius_cats | {icecube_cat}

    if not isinstance(catalog_constraints, dict):
        raise ValueError("catalog_constraints must be a dict.")

    for key, value in catalog_constraints.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unsupported catalog_constraints key '{key}'. Must be one of {sorted(valid_keys)}.")

        # FERMIGTRIG or SAXGRBMGRB
        if key in error_radius_cats:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"catalog_constraints['{key}'] must be a float (maximum error radius in degrees), "
                    f"got {value}.")
            continue

        # icecubepsc
        if not isinstance(value, dict) or set(value) != {"select_topN", "max_search_radius"}:
            raise ValueError(
                f"catalog_constraints['{icecube_cat}'] must be a dict with exactly the keys "
                "'select_topN' and 'max_search_radius', "
                f"got {value}.")
        if not isinstance(value["select_topN"], int):
            raise ValueError(
                f"catalog_constraints['{icecube_cat}']['select_topN'] must be an int, "
                f"got {value['select_topN']}.")
        if not isinstance(value["max_search_radius"], (int, float)):
            raise ValueError(
                f"catalog_constraints['{icecube_cat}']['max_search_radius'] must be a float, "
                f"got {value['max_search_radius']}.")
