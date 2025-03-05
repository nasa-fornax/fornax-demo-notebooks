---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: notebook
  language: python
  name: python3
---

# Cross-Matching ZTF and Pan-STARRS using LSDB

+++

[LSDB](https://lsdb.io) is a useful package for performing large cross-matches between source catalogs. It's built to run across multiple nodes with Dask parallelization, but even without parallelization it is high-performance. Here we will benchmark the performance of LSDB on the NASA Fornax platform with and without Dask.

We will start small, trying to cross-match 10,000 sources from ZTF with Pan-STARRS. We will then scale up by factors of roughly 10 until either (a) the platform can no longer handle the load, or (b) we do the full cross-match.

For each level, we want to know the performance with (1) no Dask, (2) minimal Dask - like 2 workers, (3) bigger Dask - as many workers as we can use, and (4) auto-scaling Dask.

+++

## Install LSDB

Fornax has LSDB installed, but this notebook was written for lsdb v0.3.0, and it breaks with other versions. I haven't had time to figure out the specifics, so for now let's stick with v0.3.0.

```{code-cell} ipython3
%pip install git+https://github.com/astronomy-commons/lsdb.git@v0.3.0
```

## Preconfiguring the Run
First choose the number of rows we want to cross-match and our `dask` environment. Note that you can also configure `dask` using the `daskhub` options on Fornax. If you go this route, leave `dask_workers = 0` below.

```{code-cell} ipython3
# The left table will have about this many rows. The cross-matched product will have fewer.
Nrows = 10_000

# dask_workers can be 0 (no dask), 1-Ncores, or "scale" for auto-scaling
dask_workers = 0
```

## Imports
We require the use of `astropy` for coordinates and units, and `lsdb` to read the catalogs and perform the cross-match. Optionally, we will set up `dask` parallelization.

```{code-cell} ipython3
from astropy.coordinates import SkyCoord
from astropy import units as u
from lsdb.core.search import ConeSearch

import lsdb

# Set up dask cluster
if dask_workers != 0:
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()

    if dask_workers == "scale":
        import os
        cluster.adapt(minimum_cores=1, maximum_cores=os.cpu_count())
    else:
        cluster.scale(dask_workers)
        
    client = Client(cluster)
    client

# Select the search radius to give us the right number of rows
radius = { # Nrows: radius_arcseconds
           10_000:     331,
          100_000:    1047,
        1_000_000:    3318,
       10_000_000:  11_180,
      100_000_000:  33_743,
    1_000_000_000: 102_000,
}
```

## Read in catalogs and downselect ZTF to Nrows

```{code-cell} ipython3
# Define sky area. Here we're using the Kepler field.
c = SkyCoord('19:22:40  +44:30:00', unit=(u.hourangle, u.deg))
cone_ra, cone_dec = c.ra.value, c.dec.value
radius_arcsec = radius[Nrows]
cone_filter = ConeSearch(cone_ra, cone_dec, radius_arcsec)

# Read ZTF DR20
ztf_path = ("s3://irsa-mast-tike-spitzer-data/data/ZTF/dr20/objects/hipscat/ztf-dr20-objects-hipscat")
ztf_piece = lsdb.read_hipscat(ztf_path, columns=["oid", "ra", "dec"], search_filter=cone_filter)

# Read Pan-STARRS DR2
ps1_path = "s3://stpubdata/panstarrs/ps1/public/hipscat/otmo"
ps1 = lsdb.read_hipscat(ps1_path, storage_options={'anon': True},
    columns=["objName","objID","raMean","decMean"])
```

## Initialize the crossmatch and compute, measuring the time elapsed.

```{code-cell} ipython3
# Setting up the cross-match actually takes very little time
ztf_x_ps1 = ztf_piece.crossmatch(ps1, radius_arcsec=1, n_neighbors=1, suffixes=("_ztf", "_ps1"))
```

```{code-cell} ipython3
%%time
# Executing the cross-match does take time
xmatch = ztf_x_ps1.compute()
```

```{code-cell} ipython3
# Check the length of the resulting table
print(f"Number of rows in:  {len(ztf_piece.compute()):,d}")
print(f"Number of rows out: {len(xmatch):,d}")
```

## Record benchmarks

+++

Benchmarks on Fornax XLarge instance using 
- no dask (t0)
- one dask worker (t1)
- two dask workers (t2)
- four (t4)
- eight (t8)
- sixteen (t16)
- autoscaling dask with 1-128 cores (tX)
  
| Nrows |  Nout | t0 (s) | t1 (s) | t2 (s) | t4 (s) | t8 (s) | t16 (s) | tX (s) |
| ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------- | ------ |
| 1e4   |  8593 |  1.46  |  6.37  |  5.80  |  5.88  |  5.53  |   5.59  |  5.43  |
| 1e5   | 85237 |  2.12  |  6.50  |  5.99  |  6.18  |  5.62  |   5.71  |  5.79  |
| 1e6   | 8.4e5 |  3.46  |  8.84  |  6.88  |  7.07  |  6.35  |   6.39  |  6.78  |
| 1e7   | 8.4e6 |  27.8  |  33.0  |  21.0  |  15.0  |  11.8  |  11.4   |  11.4  |
| 1e8   | 8.6e7 |  192   |    -   |    -   |    -   |    -   |    -    |    -   |
| 1e9   | 8.7e8 |  1535  |    -   |    -   |    -   |    -   |    -    |    -   |

("-" indicates out-of-memory behavior.)

+++

## Summary

Fornax is capable of hosting cross-matches between large catalogs. There is no performance enhancement with `dask` until cross-matching ~10 million sources, at which point you get roughly a factor of two improvement at best. Larger than that hits memory issues with `dask`, and takes hours without `dask` (although I haven't actually finished the 1e8 match). There are ways to configure the maximum memory used by a `dask` worker, which I haven't yet explored. That might help.

+++

## Addendum

The `dask.distributed.LocalCluster` has an argument `memory_limit` that seems to help on the larger runs. According to the documentation, system memory is divided equally between all workers by default. However, the larger runs end up exceeding the memory budget per worker. When I specify `memory_limit=1/dask_workers` (i.e., manually splitting the memory between workers), it allows the 1e8 run to finish, with up to 8 workers. It is not clear to me why this works, while the automatic memory division does not. The 1e9 run still hits memory errors using any `dask`, and trying a full catalog cross-match did not finish within 24 hours (sorry NASA folks).


| Nrows |  Nout | t0 (s) | t1 (s) | t2 (s) | t4 (s) | t8 (s) | t16 (s) | tX (s) |
| ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------- | ------ |
| 1e8   | 8.6e7 |  192   |  191   |  119   |   79   |   60   |    -    |    -   |
| 1e9   | 8.7e8 |  1535  |    -   |    -   |    -   |    -   |    -    |    -   |
