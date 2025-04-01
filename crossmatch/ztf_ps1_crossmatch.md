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

# Learning Goals

By the end of this tutorial, you will:
- understand how to cross-match cloud-based catalogs using `lsdb`.
- understand how to parallelize `lsdb` cross-matches using `dask`.
- have a feeling for when `dask` parallelization can be helpful.

+++

# Introduction

[LSDB](https://lsdb.io) is a useful package for performing large cross-matches between source catalogs. It's built to run across multiple nodes with Dask parallelization, but even without parallelization it is high-performance. Here we will benchmark the performance of LSDB on the NASA Fornax platform with and without Dask.

We will start small, trying to cross-match 10,000 sources from ZTF with Pan-STARRS. We will then scale up by factors of roughly 10 until either (a) the platform can no longer handle the load, or (b) we do the full cross-match.

For each level, we want to know the performance with (1) no Dask, (2) minimal Dask - like 2 workers, (3) bigger Dask - as many workers as we can use, and (4) auto-scaling Dask.

+++

# Imports
We require the following packages:
- `os` solely for the `cpu_count` function,
- `upath` to generate Path objects using cloud URIs,
- `astropy` for coordinates and units,
- `lsdb` to read the catalogs and perform the cross-match, and
- `dask` for parallelization.

```{code-cell} ipython3
# %pip install -r requirements_ztf_ps1_crossmatch.txt
```

```{code-cell} ipython3
from os import cpu_count
from upath import UPath
from astropy.coordinates import SkyCoord
from astropy import units as u

import lsdb
from lsdb.core.search import ConeSearch
from dask.distributed import Client, LocalCluster
```

## 1. Preconfiguring the Run
First choose the number of rows we want to cross-match and our `dask` environment. Note that you can also configure `dask` using the `daskhub` options on Fornax. If you go this route, leave `dask_workers = 0` below.

```{code-cell} ipython3
# The left table will have about this many rows. The cross-matched product will have fewer.
Nrows = 10_000

# dask_workers can be 0 (no dask), 1-Ncores, or "scale" for auto-scaling
dask_workers = 0


# Set up dask cluster
if dask_workers != 0:
    cluster = LocalCluster()

    if dask_workers == "scale":
        cluster.adapt(minimum_cores=1, maximum_cores=cpu_count())
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

## 2. Read in catalogs and downselect ZTF to Nrows

```{code-cell} ipython3
# Define sky area. Here we're using the Kepler field.
c = SkyCoord('19:22:40  +44:30:00', unit=(u.hourangle, u.deg))
cone_ra, cone_dec = c.ra.value, c.dec.value
radius_arcsec = radius[Nrows]
cone_filter = ConeSearch(cone_ra, cone_dec, radius_arcsec)

# Read ZTF DR23
ztf_path = UPath("s3://irsa-fornax-testdata/ZTF/dr23/objects/hats/")
ztf_piece = lsdb.read_hats(ztf_path, columns=["oid", "ra", "dec"], search_filter=cone_filter)

# Read Pan-STARRS DR2
ps1_path = UPath("s3://stpubdata/panstarrs/ps1/public/hats/otmo", anon=True)
ps1_margin = UPath("s3://stpubdata/panstarrs/ps1/public/hats/otmo_10arcs", anon=True)

ps1 = lsdb.read_hats(ps1_path, margin_cache=ps1_margin,
    columns=["objName","objID","raMean","decMean"],
    search_filter=cone_filter)
```

## 3. Initialize the crossmatch and compute, measuring the time elapsed.

```{code-cell} ipython3
# Setting up the cross-match actually takes very little time
ztf_x_ps1 = ztf_piece.crossmatch(ps1, radius_arcsec=1, n_neighbors=1, suffixes=("_ztf", "_ps1"))
ztf_x_ps1
```

```{code-cell} ipython3
%%time
# Executing the cross-match does take time
xmatch = ztf_x_ps1.compute()
xmatch
```

```{code-cell} ipython3
# Check the length of the resulting table
print(f"Number of rows in:  {len(ztf_piece.compute()):,d}")
print(f"Number of rows out: {len(xmatch):,d}")
```

## 4. Record benchmarks

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
| 1e4   |  8590 |  6.54  |  6.77  |  6.48  |  5.49  |  7.07  |   6.69  |  6.00  |
| 1e5   | 85242 |  5.41  |  6.87  |  5.56  |  5.89  |  5.70  |   5.86  |  11.2  |
| 1e6   | 8.4e5 |  13.7  |  9.09  |  9.77  |  8.72  |  7.37  |   7.38  |  8.12  |
| 1e7   | 8.4e6 |  74    |  56.7  |  28.2  |  17.3  |  14.1  |   12.2  |  16.9  |
| 1e8   | 8.6e7 |  397   |    -   |    -   |    -   |    -   |    -    |    -   |
| 1e9   | 8.7e8 |  3099  |    -   |    -   |    -   |    -   |    -    |    -   |

("-" indicates out-of-memory behavior.)

+++

## 5. Summary

Fornax is capable of hosting cross-matches between large catalogs. There is no performance enhancement with `dask` until cross-matching ~1 million sources, and more significant at 10 million, at which point you get up to a factor of 5 improvement. Larger than 10M, `dask` encounters memory issues, and takes a long time without `dask`. There are ways to configure the maximum memory used by a `dask` worker, which I begin to explore in the next section.

+++

## 6. Addendum

The `dask.distributed.LocalCluster` has an argument `memory_limit` that seems to help on the larger runs. According to the documentation, system memory is divided equally between all workers by default. However, the larger runs end up exceeding the memory budget per worker. When I specify `memory_limit=1/dask_workers` (i.e., manually splitting the memory between workers), it allows the 1e8 run to finish, with up to 8 workers. It is not clear to me why this works, while the automatic memory division does not. The 1e9 run still hits memory errors using any `dask`, and trying a full catalog cross-match did not finish within 24 hours (sorry NASA folks).


| Nrows |  Nout | t0 (s) | t1 (s) | t2 (s) | t4 (s) | t8 (s) | t16 (s) | tX (s) |
| ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------- | ------ |
| 1e8   | 8.6e7 |  397   |  376   |  204   |  124   |   81   |    -    |    -   |
| 1e9   | 8.7e8 |  3099  |    -   |    -   |    -   |    -   |    -    |    -   |

+++

# About this Notebook

This notebook was authored by [Zach Claytor](mailto:zclaytor@stsci.edu), Astronomical Data Scientist at Space Telescope Science Institute.

+++ {"jp-MarkdownHeadingCollapsed": true}

# Citations

If you use `astropy` for published research, please cite the authors. 
Follow this link for more information about citing `astropy`:

* [Citing `astropy`](https://www.astropy.org/acknowledging.html)
