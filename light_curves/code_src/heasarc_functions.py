import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import vstack
import pyvo
from tqdm.auto import tqdm

from data_structures import MultiIndexDFObject


def make_hist_error_radii(missioncat):
    """plots a histogram of error radii from a HEASARC catalog

    example calling sequences:
    resulttable = make_hist_error_radii('FERMIGTRIG')
    resulttable = make_hist_error_radii('SAXGRBMGRB')


    Parameters
    ----------
    missioncat : str
        single catalog within HEASARC to grab error radii values  Must be one of the catalogs listed here:
            https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html#getting-list-of-available-missions
    Returns
    -------
    heasarcresulttable : astropy table
        results of the heasarc search including name, ra, dec, error_radius

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
    """Searches HEASARC archive for light curves from a specific list of mission catalogs

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    catalog_error_radii : dict
        Catalogs to query and their corresponding max error radii. Dictionary key must be one of the tables listed
        here: https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html#getting-list-of-available-missions.
        Value must be the maximum error radius to include in the returned catalog of objects (ie., we are not
        interested in GRBs with a 90degree error radius because they will fit all of our objects).

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
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
            SELECT cat.name, cat.ra, cat.dec, cat.error_radius, cat.time, mt.objectid, mt.label
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
