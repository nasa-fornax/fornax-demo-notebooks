import numpy as np
import pandas as pd
from astropy.table import Table
import lsdb
from dask.distributed import Client
from data_structures import MultiIndexDFObject
from upath import UPath

def hats_get_lightcurves(
    sample_table: Table,
    object_catalog_path: str,
    lightcurve_catalog_path: str,
    object_columns: list[str],
    light_curve_columns: list[str],
    filter_id_to_name: dict[int, str],
    id_col: str,
    time_col: str,
    flux_col: str,
    err_col: str,
    *,
    radius: float = 1.0,
    object_margin_cache=None,
    lightcurve_margin_cache=None,
) -> MultiIndexDFObject:
    """
    Generic LSDB/HATs light-curve fetcher by cross-matching an object catalog
    to an input coord list, then joining to a light-curve catalog.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with columns ['objectid', 'coord', 'label'] (SkyCoord in 'coord').
    object_catalog_path : str
        S3/HATs path to the **object** catalog (used for cross-matching).
    lightcurve_catalog_path : str
        S3/HATs path to the **light curve** catalog.
    object_columns : list of str
        Columns to read from the object catalog (must include `id_col`, ra/dec).
    detect_columns : list of str
        Columns to read from the light-curve catalog (must include `id_col`,
        `time_col`, `flux_col`, `err_col`, plus the filter-ID column).
    filter_id_to_name : dict
        Mapping from integer filter IDs to human-readable band names.
    id_col : str
        Name of the object-ID column in both catalogs.
    time_col : str
        Name of the time/MJD column in the light-curve catalog.
    flux_col : str
        Name of the flux column in the light-curve catalog.
    err_col : str
        Name of the flux-uncertainty column in the light-curve catalog.
    radius : float, optional
        Cross-match radius in arcseconds (default 1.0).

    Returns
    -------
    MultiIndexDFObject
        A multi-indexed (objectid, label, band, time) container of the light curves.
    """
    # 1) read the two catalogs lazily
    
    # this table will be used for cross matching with our sample's ra and decs
    # but does not have light curve information
    obj_cat = lsdb.read_hats(object_catalog_path,
                             columns=object_columns,
                             margin_cache=object_margin_cache)
    # this table houses the light curves
    lc_cat  = lsdb.read_hats(lightcurve_catalog_path,
                             columns=light_curve_columns,
                            margin_cache=lightcurve_margin_cache)
    
    # 2) convert astropy sample_table into an LSDB catalog
    sample_df = pd.DataFrame({
        'objectid': sample_table['objectid'],
        'ra_deg':    sample_table['coord'].ra.deg,
        'dec_deg':   sample_table['coord'].dec.deg,
        'label':     sample_table['label']
    })
    # convert dataframe to hats catalog
    sample_lsdb = lsdb.from_dataframe(
        sample_df,
        ra_column='ra_deg',
        dec_column='dec_deg',
        margin_threshold=10,
        drop_empty_siblings=True
    )

    # 3) cross-match to find nearest object
    # only keep the best match
    matched_obj = obj_cat.crossmatch(
        sample_lsdb,
        radius_arcsec=radius,
        n_neighbors=1,
        suffixes=("", "")     # ← prevent “objID” → “objID_1”
    )

    # 4) join to the light-curve catalog
    matched_lc = matched_obj.join(
        lc_cat,
        left_on=id_col,
        right_on=id_col,
        output_catalog_name=f"{id_col}_lc",
        suffixes=["", ""]
    )

    # 5) compute on Dask
    # here is where the actual work gets done
    with Client():
        matched_df = matched_lc.compute()

    # 6) handle no matches
    if matched_df.empty:
        return MultiIndexDFObject(data=pd.DataFrame())

    # 7) map filter IDs to names
    filter_col = next(c for c in light_curve_columns if 'filter' in c.lower())
    band_names = np.vectorize(filter_id_to_name.get)(matched_df[filter_col])

    # 8) build and index the DataFrame
    df_lc = pd.DataFrame({
        'flux':     pd.to_numeric(matched_df[flux_col], errors='coerce'),
        'err':      pd.to_numeric(matched_df[err_col],  errors='coerce'),
        'time':     pd.to_numeric(matched_df[time_col], errors='coerce'),
        'objectid': matched_df['objectid'].astype(np.int64),
        'band':     band_names,
        'label':    matched_df['label'].astype(str)
    })
    
    df = pd.DataFrame(df_lc).set_index(['objectid', 'label', 'band', 'time'])
    
    return MultiIndexDFObject(data=df)


