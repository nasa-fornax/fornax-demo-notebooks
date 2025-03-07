import time
import numpy as np
import pandas as pd

from astropy.table import Table
from astroquery.gaia import Gaia
from data_structures import MultiIndexDFObject


def gaia_get_lightcurves(sample_table, *, search_radius=1/3600, verbose=0):
    '''
    Creates a lightcurve Pandas MultiIndex object from Gaia data for a list of coordinates.
    This is the MAIN function.

    Parameters
    ----------
    sample_table : Astropy Table
        main source catalog with coordinates, labels, and objectids
    search_radius: float(degrees)
        How far from a sources is ok for a match
     verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full

    Returns
    --------
    MultiIndexDFObject with Gaia light curve photometry

    '''
    # This code is broken into two steps.  The first step, `Gaia_retrieve_catalog` retrieves the
    # Gaia source ids for the positions of our sample. These come from the "Gaia DR3 source lite catalog".
    # However, that catalog only has a single photometry point per object.  To get the light curve
    # information, we use the function `gaia_retrieve_epoch_photometry` to use the source ids to
    # access the "EPOCH_PHOTOMETRY" catalog.

    # Retrieve Gaia table with Source IDs ==============
    gaia_table = Gaia_retrieve_catalog(sample_table,
                                       search_radius=search_radius,
                                       verbose=verbose
                                       )
    # if none of the objects were found, there's nothing to load and the Gaia_retrieve_EPOCH_PHOTOMETRY fnc
    # will raise an HTTPError. just return an empty dataframe instead of proceeding
    if len(gaia_table) == 0:
        return MultiIndexDFObject()

    # Extract Light curves ===============
    # request the EPOCH_PHOTOMETRY from the Gaia DataLink Service

    gaia_df = Gaia_retrieve_epoch_photometry(gaia_table)

    # if the epochal photometry is empty, return an empty dataframe
    if len(gaia_df) == 0:
        return MultiIndexDFObject()

    # Create light curves =================
    df_lc = Gaia_clean_dataframe(gaia_df)

    return df_lc


def Gaia_retrieve_catalog(sample_table, search_radius, verbose):
    '''
    Retrieves the photometry table for a list of sources.

    Parameter
    ----------
    sample_table : Astropy Table
        main source catalog with coordinates, labels, and objectids

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

    # first make an astropy table from our master list of coordinates
    # as input to the pyvo TAP query
    upload_table = sample_table['objectid', 'label']
    upload_table['ra'] = sample_table['coord'].ra.deg
    upload_table['dec'] = sample_table['coord'].dec.deg

    # this query is too slow without gaia.random_index.
    # Gaia helpdesk is aware of this bug somewhere on their end
    querystr = f"""
        SELECT gaia.ra, gaia.dec, gaia.random_index, gaia.source_id, mt.ra, mt.dec, mt.objectid, mt.label
        FROM tap_upload.table_test AS mt
        JOIN gaiadr3.gaia_source_lite AS gaia
        ON 1=CONTAINS(POINT('ICRS',mt.ra,mt.dec),CIRCLE('ICRS',gaia.ra,gaia.dec,{search_radius}))
        """
    # use an asynchronous query of the Gaia database
    # cross match with our uploaded table
    j = Gaia.launch_job_async(query=querystr, upload_resource=upload_table, upload_table_name="table_test")

    results = j.get_results()

    if verbose:
        print(f"\nSearch completed in {time.time() - t1:.2f} seconds \n"
              f"Number of objects matched: {len(results)} out of {len(sample_table)}.")

    return results


def Gaia_chunks(lst, n):
    """
    "Split an input list into multiple chunks of size =< n"

    Parameters
    ----------
    lst: list of gaia Ids
    n: int = maximum size of the desired chunk

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def Gaia_retrieve_epoch_photometry(gaia_table):
    """
    Function to retrieve EPOCH_PHOTOMETRY catalog product for Gaia
    entries using the DataLink. Note that the IDs need to be DR3 source_id and needs to be a list.

    Code fragments taken from:
    https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products#datalink_jntb_get_above_lim

    Parameters
    ----------
    gaia_table: Astropy Table
        catalog of gaia source ids as well as the coords, objectid, and labels of our targets

    Returns
    --------
    Returns a dictionary (key = source_id) with a table of photometry as a function of time.

    """

    # gaia datalink server has a threshold of max 5000 requests,
    # so we break the input datasets into chunks of size <=5000 sources
    # and then send each chunk into the datalink server
    ids = list(gaia_table["source_id"])
    dl_threshold = 5000  # Datalink server threshold
    ids_chunks = list(Gaia_chunks(ids, dl_threshold))
    datalink_all = []

    # setup to request the epochal photometry
    # See astroquery.gaia.Gaia.load_data docs for options for parameter options as they may change
    retrieval_type = "EPOCH_PHOTOMETRY"
    data_structure = "RAW"
    data_release = "Gaia DR3"

    for chunk in ids_chunks:
        datalink = Gaia.load_data(ids=chunk,
                                  data_release=data_release,
                                  retrieval_type=retrieval_type,
                                  data_structure=data_structure,
                                  verbose=False,
                                  valid_data=True,
                                  overwrite_output_file=True,
                                  format="votable")

        # datalink contains a single VOTable, but it's wrapped in a list which is itself wrapped in a dict
        # it's safest to act as if both list and dict may contain an arbitrary number of items
        # we want to extract the VO table, turn it into a pandas dataframe, and add it to the datalink_all list
        for list_of_tables in datalink.values():
            for votable in list_of_tables:
                votable = votable.to_table()
                # Filter out masked cells from the multidim rows, so we can convert them later to a
                # MultiIndexDFObject avoiding the ``TypeError: unhashable type: 'MaskedConstant'``.
                # We do it here before heading to pandas land as then there are
                # way more complications with dealing with np.ma.MaskedArrays and pandas indexing and
                # issues about view vs index and size of arrays.
                # This is knowingly a terrible hack, btw

                flux = []
                flux_err = []
                mag = []
                obs_time = []
                for r_flux, r_flux_err, r_mag, r_obs_time in votable.iterrows(
                        'g_transit_flux', 'g_transit_flux_error', 'g_transit_mag', 'g_transit_time'):
                    obs_time.append(r_obs_time[~r_flux.mask].data)
                    mag.append(r_mag[~r_flux.mask].data)
                    flux_err.append(r_flux_err[~r_flux.mask].data)
                    flux.append(r_flux[~r_flux.mask].data)

                votable.update(Table({'g_transit_time': obs_time, 'g_transit_mag': mag,
                                      'g_transit_flux_error': flux_err, 'g_transit_flux': flux}))
                datalink_all.append(votable.to_pandas())

    # if there is no epochal photometry return an empty dataframe
    if len(datalink_all) == 0:
        return pd.DataFrame()

    datalink_all = pd.concat(datalink_all)

    # join with gaia_table to attach the objectid and label
    idcols = ["source_id", "objectid", "label"]
    gaia_source_df = gaia_table[idcols].to_pandas().set_index("source_id")
    gaia_df = datalink_all.set_index("source_id").join(gaia_source_df, how="left")

    return gaia_df.reset_index()


# clean and transform the data
def Gaia_clean_dataframe(gaia_df):
    """
    Clean and transform the EPOCH_PHOTOMETRY dataframe in preparation to add to other light curves

    Parameters
    ----------
    gaia_df: Pandas dataframe with light curve info


    Returns
    --------
    MultiIndexDFObject with all Gaia light curves

    """

    # df.flux is in electron/s
    # already have the conversion from mag to mJy so go with that.  Need to convert either way

    # generate magerr from fluxerr and flux
    gaia_df["mag"] = gaia_df.g_transit_mag
    gaia_df["magerr"] = 2.5 / np.log(10) * gaia_df.g_transit_flux_error / gaia_df.g_transit_flux

    # compute flux and flux error in mJy
    gaia_df["flux_mJy"] = 10 ** (-0.4 * (gaia_df.mag - 23.9)) / 1e3  # in mJy
    gaia_df["fluxerr_mJy"] = gaia_df.magerr / 2.5 * np.log(10) * gaia_df.flux_mJy  # in mJy

    # get time in mjd
    gaia_df["time_mjd"] = gaia_df.g_transit_time + 55197.5

    gaia_df["band"] = 'gaia_G'

    # need to rename some columns for the MultiIndexDFObject
    colmap = dict(flux_mJy="flux", fluxerr_mJy="err", time_mjd="time",
                  objectid="objectid", label="label", band="band")

    # and only keep those columns that we need for the MultiIndexDFObject
    gaia_df = gaia_df[colmap.keys()].rename(columns=colmap)

    gaia_df = gaia_df.explode(['flux', 'err', 'time'])
    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    df_lc = MultiIndexDFObject(data=gaia_df.set_index(indexes)[columns])

    return df_lc
