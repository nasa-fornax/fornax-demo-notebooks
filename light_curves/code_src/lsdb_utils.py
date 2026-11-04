from contextlib import contextmanager

import lsdb
import pandas as pd
from distributed import Client, get_worker, worker_client


@contextmanager
def dask_client():
    """Context manager yielding a Dask distributed client to compute a lazy LSDB result.

    If this function is running inside a task on an existing Dask cluster (e.g., because
    the caller was submitted to a shared Dask Gateway cluster via `client.submit()`),
    reuse that cluster with `worker_client()`, which safely gives back this task's worker
    slot for the duration of the nested computation so it can't deadlock against the
    sub-tasks it launches.

    Otherwise (e.g., the caller was invoked directly, with no Dask cluster running),
    start a small local, single-threaded-per-worker cluster and tear it down afterward.
    """
    try:
        get_worker()
    except ValueError:
        with Client(threads_per_worker=1, memory_limit=None) as client:
            yield client
    else:
        with worker_client() as client:
            yield client


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
