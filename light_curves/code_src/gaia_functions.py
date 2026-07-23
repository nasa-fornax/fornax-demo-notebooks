import time

import lsdb
import numpy as np
import pandas as pd
from dask.distributed import Client

from data_structures import MultiIndexDFObject

# Gaia DR3 epoch photometry, hosted as a HATS catalog by LINCC.
# This "object" catalog holds exactly the DR3 sources that have epoch photometry (one row per
# source), with the light curves stored in a nested `epoch_photometry` column. Because it is
# crossmatchable by position and already restricted to sources with light curves, a single
# LSDB cross-match returns both the source match and its epoch photometry.
GAIA_EPOCH_PHOT_HATS = "https://data.lsdb.io/hats/gaia_dr3_epoch_phot/"


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
    # We cross-match our sample against the Gaia DR3 epoch-photometry HATS catalog with LSDB
    # (the same cloud-native approach used for the ZTF and Pan-STARRS light curves). That
    # catalog already contains only the sources that have epoch photometry, stored in a nested
    # column, so a single cross-match returns the light curves directly.
    gaia_df = gaia_retrieve_epoch_photometry(sample_table,
                                             search_radius=search_radius,
                                             verbose=verbose
                                             )

    # if the epochal photometry is empty, return an empty dataframe
    if len(gaia_df) == 0:
        return MultiIndexDFObject()

    # Create light curves =================
    df_lc = gaia_clean_dataframe(gaia_df)

    return df_lc


def gaia_retrieve_epoch_photometry(sample_table, search_radius, verbose):
    '''
    Retrieves Gaia DR3 epoch photometry for a list of sources by cross-matching against the
    Gaia DR3 epoch-photometry HATS catalog hosted by LINCC, using LSDB.

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
    search_radius : float
        Search radius in degrees, e.g., 1/3600.
        suggested search radius is 1 arcsecond or 1/3600.
    verbose : int
        How much to talk. 0 = None, 1 = a little bit , 2 = more, 3 = full

    Returns
    --------
    gaia_df : pandas.DataFrame
        Flattened Gaia epoch photometry, one row per transit measurement, for the matched
        sources. The resulting DataFrame contains the following columns:

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
    '''
    t1 = time.time()

    # Open the Gaia DR3 epoch-photometry HATS catalog. We only need the light curves (nested in
    # `epoch_photometry`); ra/dec are used implicitly for the positional cross-match.
    epoch_catalog = lsdb.open_catalog(
        GAIA_EPOCH_PHOT_HATS,
        columns=["ra", "dec", "source_id", "epoch_photometry"],
    )

    # convert our sample's coordinates into an LSDB catalog for cross-matching
    sample_df = pd.DataFrame({
        'objectid': sample_table['objectid'],
        'ra_deg': sample_table['coord'].ra.deg,
        'dec_deg': sample_table['coord'].dec.deg,
        'label': sample_table['label'],
    })
    sample_lsdb = lsdb.from_dataframe(
        sample_df,
        ra_column="ra_deg",
        dec_column="dec_deg",
        margin_threshold=10,
        drop_empty_siblings=True,
    )

    # cross-match our sample (left) against Gaia (right), keeping only the nearest source.
    # search_radius is in degrees (for backwards compatibility); LSDB wants arcseconds.
    matched = sample_lsdb.crossmatch(
        epoch_catalog,
        radius_arcsec=search_radius * 3600,
        n_neighbors=1,
        suffixes=("", ""),
        suffix_method="all_columns",
    )

    # the cross-match is lazy; run it on a local Dask cluster.
    # Use multiple workers with a single thread per worker for better performance on Fornax
    with Client(threads_per_worker=1, memory_limit=None):
        matched_df = matched.compute()

    if verbose:
        print(f"\nSearch completed in {time.time() - t1:.2f} seconds \n"
              f"Number of objects matched: {len(matched_df)} out of {len(sample_table)}.")

    if len(matched_df) == 0:
        return pd.DataFrame()

    # push objectid/label into the nested epoch_photometry frame so they attach to every
    # flattened transit row, then flatten to one row per transit measurement.
    matched_df["epoch_photometry.objectid"] = matched_df["objectid"]
    matched_df["epoch_photometry.label"] = matched_df["label"]
    gaia_df = matched_df["epoch_photometry"].nest.to_flat()

    # the nested arrays can arrive as object dtype; coerce the columns we use to float and drop
    # transits with no valid flux (masked/NaN measurements).
    arr_cols = ['g_transit_flux', 'g_transit_flux_error', 'g_transit_mag', 'g_transit_time']
    for col in arr_cols:
        gaia_df[col] = pd.to_numeric(gaia_df[col], errors="coerce").astype(np.float64)
    gaia_df = gaia_df.dropna(subset=['g_transit_flux'])

    return gaia_df.reset_index(drop=True)


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

    gaia_df["band"] = 'Gaia_G'

    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    df_lc = MultiIndexDFObject(data=gaia_df.set_index(indexes)[columns])

    return df_lc
