import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo
from astropy.table import vstack
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


def heasarc_get_lightcurves(sample_table, *, catalog_error_radii={"FERMIGTRIG": 1.0, "SAXGRBMGRB": 3.0}):
    """
    Search selected HEASARC gamma-ray mission catalogs for transient events
    that spatially coincide with entries in the input sample_table.

    This function performs a TAP upload of the source coordinates, queries
    each HEASARC catalog for events with sufficiently small localization
    error circles, and records an event “marker” in the returned light-curve
    structure. HEASARC does not provide multi-epoch flux light curves, so
    each event is represented as a single time-stamped point with a fixed
    placeholder flux.

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
    catalog_error_radii : dict
        Dictionary specifying which HEASARC mission catalogs to search and the
        **maximum allowed error radius (degrees)** for each.

        Key : str
            A valid HEASARC table name, for example: "FERMIGTRIG" or "SAXGRBMGRB".
        Value : float (degrees)
            Maximum localization error radius to accept from that catalog.
            Recommended values are about 1.0 to 3.0.
            Very large values have the potential to match many target objects.

    Returns
    -------
     df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time]. The resulting internal
        pandas DataFrame contains the following columns:

            flux : float
                Flux values in counts per second (ct/s) as provided by the HEASARC
                mission catalogs (e.g., FERMIGTRIG, SAXGRBMGRB).
            err : float
                Flux uncertainties in counts per second (ct/s), when available.
                Placeholder flux uncertainty (0.1 ct/s).
            time : float
                Time of the event or binned observation in Mission Elapsed Time
                converted into MJD.
            objectid : int
                Input sample object identifier.
            band : str
                Mission-specific band or catalog name
                (e.g., 'FERMIGTRIG', 'SAXGRBMGRB').
            label : str
                Literature label associated with each source.    

    Notes
    -----
    * Each HEASARC catalog entry is treated as a **single event**, not a light curve.
    * Events are matched using cone searches based on the mission-provided
      error radius for each source.
    * Large sample tables are internally split into chunks (max 50,000 rows)
      for TAP upload compatibility.
    * Returned flux values should not be physically interpreted; only the
      timestamps and catalog labels carry scientific meaning.
    """

    # Prepping sample_table with float R.A. and DEC column instead of SkyCoord mixin for TAP upload

    # set the maximum number of rows in sample_table that can be uploaded in one go.
    nchunk = 50000

    upload_table = sample_table['objectid', 'label']
    upload_table['ra'] = sample_table['coord'].ra.deg
    upload_table['dec'] = sample_table['coord'].dec.deg

    # setup to store the data
    df_lc = MultiIndexDFObject()

    # get the pyvo HEASARC service.
    heasarc_tap = pyvo.regsearch(servicetype='tap', keywords=['heasarc'])[0]

    # Note that the astropy table is uploaded when we run the query with run_sync
    for heasarc_cat, max_error_radius in tqdm(catalog_error_radii.items()):
        print('working on mission', heasarc_cat)

        hquery = f"""
            SELECT cat.name, cat.ra, cat.dec, cat.error_radius, cat.time AS time, 
            mt.objectid AS objectid, mt.label AS label
            FROM {heasarc_cat} cat, tap_upload.mytable mt
            WHERE
            cat.error_radius < {max_error_radius} AND
            CONTAINS(POINT('ICRS',mt.ra,mt.dec),CIRCLE('ICRS',cat.ra,cat.dec,cat.error_radius))=1
             """

        # instead of uploading upload_table in one go, split it
        # into several tables with a maximum size of nchunk.
        # upload_tables: is a list of tables to be uploaded one at a time.
        # We use groupby in pandas to do the split
        ids = [g.index.values for k, g in upload_table.to_pandas().groupby(
            np.arange(len(upload_table)) // nchunk)]
        upload_tables = [upload_table[idd] for idd in ids]

        # hresult: is a list of query results corresponding to upload_tables.
        # hresulttable: is the stacked table of all the results from individual calls.
        hresult = [heasarc_tap.service.run_sync(hquery, uploads={'mytable': upload_table})
                   for upload_table in upload_tables]
        hresulttable = vstack([hr.to_table() for hr in hresult])

        # add results to multiindex_df
        # really just need to mark this spot with a vertical line in the plot, it's not actually a light curve
        # so making up a flux and an error, but the time stamp and mission are the real variables we want to keep
        df_heasarc = pd.DataFrame(dict(flux=np.full(len(hresulttable), 0.1), err=np.full(len(hresulttable), 0.1),
                                       time=hresulttable['time'], objectid=hresulttable['objectid'],
                                       band=np.full(len(hresulttable), heasarc_cat),
                                       label=hresulttable['label'])).set_index(["objectid", "label", "band", "time"])

        # Append to existing MultiIndex light curve object
        df_lc.append(df_heasarc)

    return df_lc
