import numpy as np
from astropy.table import Table
import lsdb
from dask.distributed import Client
from data_structures import MultiIndexDFObject
from upath import UPath
import astropy.units as u
import pandas as pd
from data_structures import MultiIndexDFObject

def ztf_get_lightcurves(sample_table, *, radius=1):
    """Searches ZTF hats files for light curves from a list of input coordinates.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves

    note that other light curves in this notebook will have "time" as MJD instead of HMJD.
    if your science depends on precise times, this will need to be corrected.

    """

    # read in the ZTF light curves to lsdb
    ztf_lc = lsdb.read_hats(
        UPath('s3://irsa-fornax-testdata/ZTF/dr23/lc/hats'),
        #margin_cache=UPath('s3://stpubdata/panstarrs/ps1/public/hats/detection_10arcs', anon=True),
        columns=["objectid","objra", "objdec", "hmjd", "filterid", "mag", "magerr", "catflags"]
    )
    # convert astropy table to pandas dataframe
    # special care for the SkyCoords in the table
    sample_df = pd.DataFrame({'objectid': sample_table['objectid'],
                              'ra_deg': sample_table['coord'].ra.deg,
                              'dec_deg': sample_table['coord'].dec.deg,
                              'label': sample_table['label']})
    
    #start a dask client for lsdb to use
    client = Client()  
    
    # convert dataframe to hipscat
    sample_lsdb = lsdb.from_dataframe(
        sample_df,
        ra_column="ra_deg",
        dec_column="dec_deg",
        margin_threshold=10,
        # Optimize partition size
        drop_empty_siblings=True
    )

    # plan to cross match ZTF object with my sample
    # only keep the best match
    matched_lc = ztf_lc.crossmatch(
        sample_lsdb,
        radius_arcsec=radius,
        n_neighbors=1,
    )

    # here is where the actual work gets done
    # compute the cross match with object table
    df = matched_lc.compute()

    #done with dask
    client.close() 
    
    # explode each array column into individual rows
    df = df.explode(["hmjd_ztf_lc_dr23", 
                     "mag_ztf_lc_dr23", 
                     "magerr_ztf_lc_dr23",
                    "catflags_ztf_lc_dr23"], ignore_index=True)
    df = df.astype({
        "hmjd_ztf_lc_dr23":   "float",
        "mag_ztf_lc_dr23":    "float",
        "magerr_ztf_lc_dr23": "float",
        "catflags_ztf_lc_dr23": "int"
    })
    
    # drop any epochs flagged as bad
    df = df.loc[df["catflags_ztf_lc_dr23"] < 32768]

    # map ZTF filter IDs → simple band names
    filter_map = {1: 'ztf_g', 2: 'ztf_r', 3: 'ztf_i'}
    df['band'] = df['filterid_ztf_lc_dr23'].map(filter_map).astype(str)

    # convert mag/magerr → flux/err (mJy)
    mag    = df["mag_ztf_lc_dr23"].to_numpy()
    magerr = df["magerr_ztf_lc_dr23"].to_numpy()
    flux_up  = ((mag - magerr) * u.ABmag).to_value('mJy')
    flux_low = ((mag + magerr) * u.ABmag).to_value('mJy')
    df["flux"] = (mag * u.ABmag).to_value('mJy')
    df["err"]  = (flux_up - flux_low) / 2

    # make the dataframe of light curves
    df_lc = pd.DataFrame({
        'flux': df["flux"],
        'err': df["err"],
        'time': df["hmjd_ztf_lc_dr23"],
        'objectid': df['objectid_from_lsdb_dataframe'],
        'band': df['band'],
        'label': df['label_from_lsdb_dataframe']
    }).set_index(["objectid", "label", "band", "time"])

    return MultiIndexDFObject(data=df_lc)

