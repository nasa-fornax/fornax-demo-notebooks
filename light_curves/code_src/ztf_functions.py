from dask.distributed import Client
import pandas as pd
import lsdb
from astropy import units as u
from upath import UPath
from data_structures import MultiIndexDFObject

def ztf_get_lightcurves(sample_table, *, radius=1.0):
    """
    Search ZTF HATS files for light curves around a list of target coordinates
    
    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with columns:
        - objectid : identifier for each target
        - coord : SkyCoord object for each target position
        - label : user-defined label for each target
    radius : float, optional
        Search radius in arcseconds. Default is 1.0.

    Returns
    -------
    MultiIndexDFObject
     """
    # 1) Start Dask client & read full ZTF light-curve catalog
    client = Client(threads_per_worker=2, memory_limit=None)
    suffix = 'ztf_lc_dr23'
    ztf_lc = lsdb.read_hats(
        UPath('s3://irsa-fornax-testdata/ZTF/dr23/lc/'),
        columns=[
            "objectid", "objra", "objdec",
            "hmjd", "filterid", "mag", "magerr", "catflags"
        ]
    )

    # 2) Convert Astropy table → pandas → LSDB catalog
    sample_df = pd.DataFrame({
        'objectid': sample_table['objectid'],
        'ra_deg':   sample_table['coord'].ra.deg,
        'dec_deg':  sample_table['coord'].dec.deg,
        'label':    sample_table['label'],
    })
    sample_lsdb = lsdb.from_dataframe(
        sample_df,
        ra_column="ra_deg",
        dec_column="dec_deg",
        margin_threshold=10,
        drop_empty_siblings=True
    )
    
    # 3) Cross-match per band
    band_map = {1: "ztf_g", 2: "ztf_r", 3: "ztf_i"}
    per_band_dfs = []

    for fid, band_name in band_map.items():
        # 3a) filter to one band and select relevant sky tiles
        ztf_band = ztf_lc.query(f"filterid == {fid}")

        # 3b) crossmatch: sample (left) → filtered band (right)
        matched = sample_lsdb.crossmatch(
            ztf_band,
            radius_arcsec=radius,
            n_neighbors=1
        )
        df = matched.compute()

        # 3c) explode any length-1 arrays
        array_cols = [c for c in df.columns if isinstance(df.iloc[0][c], (list, tuple))]
        if array_cols:
            df = df.explode(array_cols, ignore_index=True)

        # 3d) cast, drop bad flags, assign band
        df = df.astype({
            f"hmjd_{suffix}":   float,
            f"mag_{suffix}":    float,
            f"magerr_{suffix}": float,
            f"catflags_{suffix}": int
        })
        df = df[df[f"catflags_{suffix}"] < 32768]
        df["band"] = band_name

        per_band_dfs.append(df)

    client.close()

    # 4) Concatenate, convert mags → fluxes, and build final MultiIndex
    df_all = pd.concat(per_band_dfs, ignore_index=True)
    mag    = df_all[f"mag_{suffix}"].to_numpy()
    magerr = df_all[f"magerr_{suffix}"].to_numpy()
    flux_up  = ((mag - magerr) * u.ABmag).to_value('mJy')
    flux_low = ((mag + magerr) * u.ABmag).to_value('mJy')
    df_all["flux"] = (mag * u.ABmag).to_value('mJy')
    df_all["err"]  = (flux_up - flux_low) / 2

    df_lc = pd.DataFrame({
        'flux':     df_all["flux"],
        'err':      df_all["err"],
        'time':     df_all[f"hmjd_{suffix}"],
        'objectid': df_all['objectid_from_lsdb_dataframe'],
        'label':    df_all['label_from_lsdb_dataframe'],
        'band':     df_all['band']
    }).set_index(["objectid", "label", "band", "time"])

    return MultiIndexDFObject(data=df_lc)

