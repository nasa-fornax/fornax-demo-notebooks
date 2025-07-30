import lsdb
import pandas as pd
from astropy import units as u
from dask.distributed import Client

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
    ztf_lc = lsdb.read_hats(
        's3://ipac-irsa-ztf/contributed/dr23/lc/hats/',
        columns=[
            "objectid", "objra", "objdec", "filterid",
            "lightcurve.hmjd", "lightcurve.mag", "lightcurve.magerr", "lightcurve.catflags"
        ]
    )

    # 2) Convert Astropy table → pandas → LSDB catalog
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
        drop_empty_siblings=True
    )
    del sample_df

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
            n_neighbors=1,
            require_right_margin=True,
            suffixes=("_sample", ""),
        )
        df = matched.compute()

        # 3c) explode to one row per data point, add band name, and drop points with bad flags
        df["lightcurve.objectid"] = df["objectid_sample"]
        df["lightcurve.label"] = df["label_sample"]
        df["lightcurve.band"] = band_name
        df = df["lightcurve"].nest.to_flat()
        df = df.query("catflags < 32768")
        del df["catflags"]

        per_band_dfs.append(df)
        del df

    client.close()

    # 4) Concatenate, convert mags → fluxes, and build final MultiIndex
    df_all = pd.concat(per_band_dfs, ignore_index=True).rename(columns={"hmjd": "time"})
    del per_band_dfs
    mag = df_all["mag"].to_numpy()
    magerr = df_all["magerr"].to_numpy()
    flux_up = ((mag - magerr) * u.ABmag).to_value('mJy')
    flux_low = ((mag + magerr) * u.ABmag).to_value('mJy')
    df_all["flux"] = (mag * u.ABmag).to_value('mJy')
    df_all["err"] = (flux_up - flux_low) / 2

    index_cols = ["objectid", "label", "band", "time"]
    return MultiIndexDFObject(data=df_all.set_index(index_cols)[["flux", "err"]])
