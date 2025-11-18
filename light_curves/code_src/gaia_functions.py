import time

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia

from data_structures import MultiIndexDFObject


def gaia_get_lightcurves(sample_table, *, search_radius=1 / 3600, verbose=0):
    '''
    Creates a lightcurve Pandas MultiIndex object from Gaia data for a list of coordinates.
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
    search_radius: float(degrees)
        Cone-search radius in degrees for matching Gaia DR3 sources.
        Default is 1/3600 (1 arcsec).
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full

    Returns
    --------
      df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time]. The resulting internal pandas DataFrame
        contains the following columns:
        
            flux : float
                Gaia G-band transit flux in electrons per second (e-/s).
            err : float
                Uncertainty on the G-band transit flux (e-/s).
            time : float
                Time of observation in MJD.
            objectid : int
                Input sample object identifier.
            band : str
                Gaia photometric band. For epoch photometry this is always "G".
            label : str
                Literature label associated with each source.

    '''
    # This code is broken into two steps.  The first step, `gaia_retrieve_catalog` retrieves the
    # Gaia source ids for the positions of our sample. These come from the "Gaia DR3 source lite catalog".
    # However, that catalog only has a single photometry point per object.  To get the light curve
    # information, we use the function `gaia_retrieve_epoch_photometry` to use the source ids to
    # access the "EPOCH_PHOTOMETRY" catalog.

    # Retrieve Gaia table with Source IDs ==============
    gaia_table = gaia_retrieve_catalog(sample_table,
                                       search_radius=search_radius,
                                       verbose=verbose
                                       )
    # if none of the objects were found, there's nothing to load and the gaia_retrieve_epoch_photometry fnc
    # will raise an HTTPError. just return an empty dataframe instead of proceeding
    if len(gaia_table) == 0:
        return MultiIndexDFObject()

    # Extract Light curves ===============
    # request the EPOCH_PHOTOMETRY from the Gaia DataLink Service

    gaia_df = gaia_retrieve_epoch_photometry(gaia_table)

    # if the epochal photometry is empty, return an empty dataframe
    if len(gaia_df) == 0:
        return MultiIndexDFObject()

    # Create light curves =================
    df_lc = gaia_clean_dataframe(gaia_df)

    return df_lc


def gaia_retrieve_catalog(sample_table, search_radius, verbose):
    '''
    Retrieves the photometry table for a list of sources.

    Parameter
    ----------
    sample_table : astropy.table.Table
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.
    search_radius : float
        Search radius in degrees, e.g., 1/3600.
        suggested search radius is 1 arcsecond or 1/3600.
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full

    Returns
    --------
    gaia_table : astropy.table.Table
        Table containing Gaia DR3 source matches for the input coordinates.
        The table includes the following columns:

            ra : float
                Right Ascension of the matched Gaia source (degrees).
            dec : float
                Declination of the matched Gaia source (degrees).
            random_index : int
                Gaia random index used for efficient data server partitioning.
            source_id : int
                Unique Gaia DR3 source identifier.
            objectid : int
                Input sample object identifier.
            label : str
                Literature label associated with each source.
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
    j = Gaia.launch_job_async(query=querystr, upload_resource=upload_table,
                              upload_table_name="table_test")

    results = j.get_results()

    if verbose:
        print(f"\nSearch completed in {time.time() - t1:.2f} seconds \n"
              f"Number of objects matched: {len(results)} out of {len(sample_table)}.")

    return results


def gaia_chunks(lst, n):
    """
    "Split an input list into multiple chunks of size =< n"

    Parameters
    ----------
    lst: list of gaia Ids
    n: int = maximum size of the desired chunk

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def gaia_retrieve_epoch_photometry(gaia_table):
    """
    Function to retrieve EPOCH_PHOTOMETRY catalog product for Gaia
    entries using the DataLink. Note that the IDs need to be DR3 source_id and needs to be a list.

    Code fragments taken from:
    https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products#datalink_jntb_get_above_lim

    Parameters
    ----------
    gaia_table: Astropy Table
        Table returned by gaia_retrieve_catalog. Must include:
        source_id : int
            Gaia DR3 source identifier for the photometry request.

        objectid : int
            Used to label the MultiIndex rows later.

        label : str
            Text label used for grouping and plotting.

    Returns
    --------
    gaia_df : pandas.DataFrame
        Concatenated Gaia epoch photometry for all matched sources.
        The resulting DataFrame contains the following columns:

            source_id : int
                Gaia DR3 source identifier for this transit measurement.
            g_transit_flux : float
                Gaia G-band transit flux (electrons per second).
            g_transit_flux_error : float
                Flux uncertainty (electrons per second).
            g_transit_mag : float
                Gaia G-band magnitude for the transit.
            g_transit_time : float
                Time of observation in Gaia mission time (days).
            objectid : int
                Input sample object identifier.
            label : str
                Literature label associated with each source.
    """

    # gaia datalink server has a threshold of max 5000 requests,
    # so we break the input datasets into chunks of size <=5000 sources
    # and then send each chunk into the datalink server
    ids = list(gaia_table["source_id"])
    dl_threshold = 5000  # Datalink server threshold
    ids_chunks = list(gaia_chunks(ids, dl_threshold))
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
                # We need to filter out masked cells from the multidim rows, so we can convert them later to a
                # MultiIndexDFObject avoiding the ``TypeError: unhashable type: 'MaskedConstant'``.

                import numpy.ma
                arr_cols = ['g_transit_flux', 'g_transit_flux_error',
                            'g_transit_mag', 'g_transit_time']
                keep_cols = arr_cols + ['source_id']

                datalink_df = votable.to_table()[keep_cols].to_pandas().explode(arr_cols)
                mask = np.array(
                    [val is numpy.ma.masked for val in datalink_df.g_transit_flux.to_numpy()])
                datalink_df = datalink_df.loc[~mask].astype({col: float for col in arr_cols})
                datalink_all.append(datalink_df)

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
def gaia_clean_dataframe(gaia_df):
    """
    Clean and transform the EPOCH_PHOTOMETRY dataframe in preparation to add to other light curves

    Parameters
    ----------
   gaia_df : pandas.DataFrame
        Raw Gaia epoch photometry returned by `gaia_retrieve_epoch_photometry()`.
        Must contain the following columns:

            g_transit_flux : float
                G-band transit flux (electrons per second).

            g_transit_flux_error : float
                Uncertainty on the transit flux (electrons per second).

            g_transit_time : float
                Gaia mission time, which will be converted to MJD.

            objectid : int
                Identifier mapped from the input sample.

            label : str
                Literature or provenance label for the target.

    Returns
    --------
    df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time].  
        The resulting internal pandas DataFrame contains:

            flux : float
                Gaia G-band flux in electrons per second (e-/s).

            err : float
                Uncertainty on the Gaia flux (e-/s).

            time : float
                Observation time converted from Gaia mission time to MJD.

            objectid : int
                Input sample object identifier.

            band : str
                Always "G" for Gaia epoch photometry.

            label : str
                Literature label associated with each source.
    """

    # df.flux is in electron/s
    # already have the conversion from mag to mJy so go with that.  Need to convert either way

    # generate magerr from fluxerr and flux
    gaia_df["mag"] = gaia_df.g_transit_mag
    gaia_df["magerr"] = 2.5 / np.log(10) * gaia_df.g_transit_flux_error / gaia_df.g_transit_flux

    # compute flux and flux error in mJy
    gaia_df["flux"] = 10 ** (-0.4 * (gaia_df.mag - 23.9)) / 1e3  # in mJy
    gaia_df["err"] = gaia_df.magerr / 2.5 * np.log(10) * gaia_df.flux  # in mJy

    # get time in mjd
    gaia_df["time"] = gaia_df.g_transit_time + 55197.5

    gaia_df["band"] = 'G'

    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    df_lc = MultiIndexDFObject(data=gaia_df.set_index(indexes)[columns])

    return df_lc
