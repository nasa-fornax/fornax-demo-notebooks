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
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.    
    radius : float, optional
        Matching radius in arcseconds (default: 1.0).
        This is the separation used by LSDB `crossmatch()` to associate
        ZTF detections with each sample coordinate.

    Returns
    -------
    df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time]. The resulting internal pandas DataFrame
        contains the following columns:

            flux : float
                Flux values in millijansky (mJy), converted from ZTF AB magnitudes.
            err : float
                Flux uncertainty in millijansky (mJy), derived from the AB-magnitude
                upper/lower bounds.
            time : float
                Modified Julian Date (MJD) derived from ZTF HMJD values.
            objectid : int
                Input sample object identifier.
            band : str
                ZTF band label ('ztf_g', 'ztf_r', or 'ztf_i').
            label : str
                Literature label associated with each source.     

    Notes
    -----
    - ZTF data are retrieved from the DR23 LSDB HATS light-curve archive.
    - Only detections with `catflags < 32768` are retained (recommended quality filter).
    - Fluxes are converted from AB magnitudes using Astropy, then converted to mJy.
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
