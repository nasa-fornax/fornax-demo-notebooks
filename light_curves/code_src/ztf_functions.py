import multiprocessing as mp
import re
from upath import UPath

import astropy.units as u
import pandas as pd
import pyarrow.fs
import pyarrow.parquet
import pyvo
import tqdm

import helpers.scale_up
from data_structures import MultiIndexDFObject
from hats_functions import hats_get_lightcurves


def ztf_hats_get_lightcurves(sample_table, *, radius):
    """
    Fetch ZTF DR23 light curves from HATs catalogs, filter and convert mags→flux.

    This wrapper cross-matches your sample to the ZTF HATs object catalog,
    pulls down the corresponding light-curve arrays, filters out bad epochs,
    and converts AB magnitudes to fluxes in mJy.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Input table of targets. Must have columns:
        
        - ``objectid`` : int or str  
          Unique identifier for each source.  
        - ``coord`` : astropy.coordinates.SkyCoord  
          SkyCoord of each source for cross-matching.  
        - ``label`` : str  
          Label to propagate into the output light-curve index.  
    radius : float, optional
        Cross-match search radius in arcseconds (default is 1.0).

    Returns
    -------
    data_structures.MultiIndexDFObject
        A container whose `.data` is a pandas DataFrame with a MultiIndex
        (``objectid``, ``label``, ``band``, ``time``) and columns:
        
        - ``flux`` : float  
          Flux in mJy, converted from AB magnitudes.  
        - ``err`` : float  
          1σ flux uncertainty in mJy, derived from magerr.

    """
    # 1) fetch the raw arrays 
    raw = hats_get_lightcurves(
        sample_table,
        object_catalog_path=UPath('s3://irsa-fornax-testdata/ZTF/dr23/objects/hats'),
        object_margin_cache=None,
        lightcurve_catalog_path=UPath('s3://irsa-fornax-testdata/ZTF/dr23/lc/hats'),
        lightcurve_margin_cache=None,
        object_columns=["oid", "ra", "dec"],
        light_curve_columns=["objectid", "hmjd", "filterid", "mag", "magerr", "catflags"],
        id_col='objectid',
        time_col='hmjd',
        flux_col='mag',    # this becomes raw.data['flux']
        err_col='magerr',  # this becomes raw.data['err']
        radius=radius,
        filter_id_to_name=None,
    )
    # 2) turn the MultiIndex into columns
    df = raw.data.reset_index()

    # 3) rename columns
    df = df.rename(columns={"hmjd": "time"})
    #these are called flux and err in the generic hats_get_lightcurves, but are really magnitudes for ztf
    df = df.rename(columns={'flux': 'mag', 'err': 'magerr'})

    # 4) explode each array column into individual rows
    df = df.explode(["time", "mag", "magerr"], ignore_index=True)
    df = df.astype({
        "time":   "float",
        "mag":    "float",
        "magerr": "float",
    })

    # 5) drop any epochs flagged as bad
    # df = df.loc[df["catflags"] < 32768]

    # 6) convert mag/magerr → flux/err (mJy)
    mag    = df["mag"].to_numpy()
    magerr = df["magerr"].to_numpy()
    flux_up  = ((mag - magerr) * u.ABmag).to_value('mJy')
    flux_low = ((mag + magerr) * u.ABmag).to_value('mJy')
    df["flux"] = (mag * u.ABmag).to_value('mJy')
    df["err"]  = (flux_up - flux_low) / 2

    # 7) clean up intermediate columns (optional)
    df = df.drop(columns=["mag", "magerr"])

    # 8) re-index and return
    df_lc = df.set_index(['objectid', 'label', 'band', 'time'])
    
    return MultiIndexDFObject(data=df_lc)

