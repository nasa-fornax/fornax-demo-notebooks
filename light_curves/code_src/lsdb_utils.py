import lsdb
import pandas as pd


def sample_table_to_lsdb(sample_table):
    """Convert a sample astropy Table to an LSDB catalog.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with columns: coord (SkyCoord), objectid (int), label (str).

    Returns
    -------
    lsdb.Catalog
        Spatially partitioned catalog ready for crossmatch or join.
    """
    # SkyCoord cannot be stored directly in a DataFrame; extract ra/dec explicitly
    sample_df = pd.DataFrame({
        'objectid': sample_table['objectid'],
        'ra_deg': sample_table['coord'].ra.deg,
        'dec_deg': sample_table['coord'].dec.deg,
        'label': sample_table['label'],
    })
    return lsdb.from_dataframe(
        sample_df,
        ra_column="ra_deg",
        dec_column="dec_deg",
        margin_threshold=10,
        drop_empty_siblings=True,
    )
