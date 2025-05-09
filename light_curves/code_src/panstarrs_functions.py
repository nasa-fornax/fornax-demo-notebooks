import numpy as np
import pandas as pd
from astropy.table import Table
import lsdb
from dask.distributed import Client
from data_structures import MultiIndexDFObject
from upath import UPath
from hats_functions import hats_get_lightcurves


def panstarrs_hats_get_lightcurves(sample_table: Table, *, radius: float = 1.0):
    """
    Searches panstarrs hats files for light curves from a list of input coordinates.  
    
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
          PSF flux in mJy (``psfFlux * 1e3``).  
        - ``err`` : float  
          PSF flux uncertainty in mJy (``psfFluxErr * 1e3``).


    """
    df_lc =  hats_get_lightcurves(
        sample_table,
        object_catalog_path='s3://stpubdata/panstarrs/ps1/public/hats/otmo',
        object_margin_cache='s3://stpubdata/panstarrs/ps1/public/hats/otmo_10arcs',
        lightcurve_catalog_path='s3://stpubdata/panstarrs/ps1/public/hats/detection',
        lightcurve_margin_cache='s3://stpubdata/panstarrs/ps1/public/hats/detection_10arcs',
        object_columns=["objID", "raMean", "decMean", "nStackDetections"],
        light_curve_columns=["objID", "detectID", "obsTime", "filterID", "psfFlux", "psfFluxErr"],
        id_col='objID',
        time_col='obsTime',
        flux_col='psfFlux',
        err_col='psfFluxErr',
        radius=radius,
        filter_id_to_name={
            1: 'Pan-STARRS g',
            2: 'Pan-STARRS r',
            3: 'Pan-STARRS i',
            4: 'Pan-STARRS z',
            5: 'Pan-STARRS y',
        }
    )
    return df_lc


    return MultiIndexDFObject(data=df_lc)
