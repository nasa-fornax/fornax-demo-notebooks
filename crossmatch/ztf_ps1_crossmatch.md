---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: python3
  language: python
---

# Cross-Matching ZTF and Pan-STARRS using LSDB

+++

# Learning Goals

By the end of this tutorial, you will:
- understand how to cross-match cloud-based catalogs using `lsdb`.
- understand how to parallelize `lsdb` cross-matches using `dask`.
- have a feeling for when `dask` parallelization can be helpful.
- understand the limitations of each of the available environments on Fornax.

+++

# Introduction

[LSDB](https://lsdb.io) is a useful package for performing large cross-matches between source catalogs. It can leverage the [Dask](https://www.dask.org/) library to work with larger-than-memory data sets and distribute computation tasks across multiple cores. Run on a cloud computing platform, users can read catalogs from the cloud and perform large cross-matches without ever having to download a file. Here we will benchmark the performance of LSDB cloud-based cross-matching on the NASA Fornax platform with and without Dask.

We will start small, trying to cross-match 10,000 sources from ZTF with Pan-STARRS. We will then scale up by factors of roughly 10 until either (a) the platform can no longer handle the load, or (b) we do the full cross-match.

For each level, we want to know the performance with (1) default Dask configuration, (2) minimal Dask - 1 worker, (3) bigger Dask - as many workers as we can use, and (4) auto-scaling Dask.

+++

# Runtime

As of August 8, 2025, as written (10,000 rows with the "default" `dask` settings), this notebook takes about 45 seconds to run on the "small" Fornax environment. Users can modify the configuration for larger cross-matches, which will take more time. E.g., cross-matching 10 million rows on the "large" environment can take ~5 minutes.

+++

# Imports
We require the following packages:
- `os` solely for the `cpu_count` function,
- `datetime` for measuring the crossmatch time,
- `pandas` to write and read the CSV of benchmarks
- `matplotlib` to plot the benchmarks,
- `astropy` for coordinates and units,
- `lsdb` to read the catalogs and perform the cross-match, and
- `dask` for parallelization.

```{code-cell} ipython3
# %pip install -r requirements_ztf_ps1_crossmatch.txt
```

```{code-cell} ipython3
from os import cpu_count
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

import lsdb
from lsdb.core.search import ConeSearch
from dask.distributed import Client, LocalCluster
```

## 1. Preconfiguring the Run
First choose the number of rows we want to cross-match and our `dask` environment. Note that you can also configure `dask` using the `daskhub` options on Fornax. If you go this route, leave `dask_workers = None` below.

For tips on using `dask` with `lsdb`, see [`lsdb`'s Dask Cluster Tips](https://docs.lsdb.io/en/stable/tutorials/dask-cluster-tips.html).

```{code-cell} ipython3
# At the time of writing, Fornax has four sizes of environment available:
# small, medium, large, or xlarge
fornax_env = "small"

# The left table will have about this many rows. The cross-matched product will have fewer.
Nrows = 10_000

# dask_workers can be:
# - "default" (uses the default scheduler which runs all tasks in the main process)
# - an integer, which uses a fixed number of workers
# - "scale" to use an adaptive, auto-scaling cluster.
dask_workers = "default"


# Set up dask cluster
if dask_workers == "scale":
    cluster = LocalCluster()
    cluster.adapt(minimum_cores=1, maximum_cores=cpu_count())
    client = Client(cluster)
elif isinstance(dask_workers, int):        
    client = Client(
        # Number of Dask workers - Python processes to run
        n_workers=dask_workers,
        # Limits number of Python threads per worker
        threads_per_worker=1,
        # Memory limit per worker
        memory_limit=None,
    )
elif dask_workers == "default":
    # Either using default configuration or configuring Dask
    # using the built-in DaskHub
    pass
else:
    raise ValueError("`dask_workers` must be one of 'default', 'scale' or an int.")

# Select the search radius to give us the right number of rows.
radius = { # Nrows: radius_arcseconds
           10_000:     331,
          100_000:    1047,
        1_000_000:    3318,
       10_000_000:  11_180,
      100_000_000:  33_743,
    1_000_000_000: 102_000,
}
# The values in the above dictionary were determined experimentally, by 
# incrementing/decrementing the radius until the desired number of 
# catalog rows was returned.
```

## 2. Read in catalogs and downselect ZTF to Nrows

```{code-cell} ipython3
# Define sky area. Here we're using the Kepler field.
c = SkyCoord('19:22:40  +44:30:00', unit=(u.hourangle, u.deg))
cone_ra, cone_dec = c.ra.value, c.dec.value
radius_arcsec = radius[Nrows]
cone_filter = ConeSearch(cone_ra, cone_dec, radius_arcsec)

# Read ZTF DR23
ztf_path = "s3://ipac-irsa-ztf/contributed/dr23/objects/hats"
ztf_piece = lsdb.open_catalog(
    ztf_path, 
    columns=["oid", "ra", "dec"], 
    search_filter=cone_filter
)

# Read Pan-STARRS DR2
ps1_path = "s3://stpubdata/panstarrs/ps1/public/hats/otmo"
ps1_margin = "s3://stpubdata/panstarrs/ps1/public/hats/otmo_10arcs"
ps1 = lsdb.open_catalog(
    ps1_path, 
    margin_cache=ps1_margin,
    columns=["objName","objID","raMean","decMean"],
)
```

## 3. Initialize the crossmatch and compute, measuring the time elapsed.

```{code-cell} ipython3
# Setting up the cross-match actually takes very little time
ztf_x_ps1 = ztf_piece.crossmatch(ps1, radius_arcsec=1, n_neighbors=1, suffixes=("_ztf", "_ps1"))
ztf_x_ps1
```

```{code-cell} ipython3
# Executing the cross-match does take time
t0 = datetime.now()
xmatch = ztf_x_ps1.compute()
t1 = datetime.now() - t0

print("Time Elapsed:", t1)
xmatch
```

```{code-cell} ipython3
# Check the length of the resulting table
rows_out = len(xmatch)
print(f"Number of rows out: {rows_out:,d}")
```

```{code-cell} ipython3
# It's good practice to explicitly close the Dask cluster/client when finished
if dask_workers != "default":
    if dask_workers == "scale":
        cluster.close()
    client.close()
```

## 4. Record and plot benchmarks

Write the recorded benchmark to an output file, then make plots to analyze the benchmarks.

```{code-cell} ipython3
filename = f"xmatch_benchmarks.csv"

try:
    # read in file if it exists
    benchmarks = pd.read_csv(filename, index_col=["Env", "Nrows", "Nworkers"])
except FileNotFoundError:
    # otherwise create an empty DataFrame
    multi_index = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=["Env", "Nrows", "Nworkers"])
    benchmarks = pd.DataFrame(index=multi_index, columns=["time", "Nrows_out", "updated"])

# assign values
benchmarks.loc[
    (fornax_env, Nrows, dask_workers),
    ["time", "Nrows_out", "updated"]
    ] = t1.total_seconds(), int(rows_out), datetime.now().strftime("%Y-%m-%d %H:%M:%S")

benchmarks = benchmarks.sort_index()
#benchmarks.to_csv(filename) # Uncomment this to write the new benchmarks to file
benchmarks
```

```{code-cell} ipython3
benchmarks = pd.read_csv("xmatch_benchmarks.csv", index_col=["Env", "Nrows", "Nworkers"])

Ncpus = {
    "small": 2,
    "medium": 4,
    "large": 16,
    "xlarge": 128,
}

nworkers = [1, 2, 4, 8, 16, 32, "default", "scale"]

def plot_by_nworkers(fornax_env, ax):
    # Plot execution time vs. number of dask workers for each scale job
    b = benchmarks.loc[fornax_env]
    ncpu = Ncpus[fornax_env]
    
    for n in b.index.levels[0]:
        try:
            # get the relevant benchmarks
            b_N = b.xs(n, level="Nrows").reset_index()

            # convert Nworkers to categorical with desired order
            b_N["Nworkers"] = pd.Categorical(b_N["Nworkers"], 
                                             categories=[str(x) for x in nworkers], 
                                             ordered=True)
            b_N = b_N.sort_values("Nworkers")

            # only plot if there is more than 1 data point
            if len(b_N) > 1:
                ax.plot("Nworkers", "time", marker="s", linestyle="-", data=b_N, label=f"{n} rows")
        except:
            pass

    ax.set(yscale="log", xlabel="Nworkers", ylabel="Execution Time (s)",
           title=f"Fornax '{fornax_env}' environment ({ncpu} CPUs)")
    ax.legend()

def plot_by_nrows(fornax_env, ax):
    # Plot execution time vs. number of cross-matched rows for each dask worker scenario
    b = benchmarks.loc[fornax_env]
    ncpu = Ncpus[fornax_env]

    for n in nworkers:
        try:
            # get the relevant benchmarks
            b_N = b.xs(str(n), level="Nworkers").reset_index()
            # only plot if there is more than 1 data point
            if len(b_N) > 1:
                ax.plot("Nrows", "time", marker="s", linestyle="-", data=b_N, label=f"{n} workers")
        except:
            pass
            
    ax.set(xlabel="Nrows", ylabel="Execution Time (s)", xscale="log", yscale="log")
    ax.legend()


fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for i, env in enumerate(["small", "medium", "large"]):
    plot_by_nworkers(env, axs[0, i])
    plot_by_nrows(env, axs[1, i])
fig.tight_layout()
```

With all of our benchmarks recorded, we can see how the `lsdb` cross-match performs for larger and larger catalogs, and with more and more `dask` workers available. Across all Fornax environment sizes and cross-match sizes, increasing the number of `dask` workers improves the execution time. Interestingly, the agnostic approach---not manually specifying any `dask` parameters, but instead letting `lsdb` use the default `dask` behavior---usually results in the best performance. This indicates that the `lsdb` cross-match functions are well-optimized and well-configured for use with `dask` without much user oversight.

+++

## 5. Summary

Fornax is capable of hosting cross-matches between large catalogs using the `lsdb` package. `lsdb` leverages `dask` to efficiently plan cross-match jobs using multiple workers, which results in large performance gains, especially as jobs scale up to millions or tens of millions of catalog rows.

**Recommendations**
- For catalogs with 100,000 rows or more, consider using `lsdb` to convert the catalogs to HATS format and cross-matching.
- Unless you know exactly what you are doing with `dask`, it is acceptable (and usually even optimal!) to use `lsdb`'s default `dask` settings to parallelize jobs. In other words, when using `lsdb`, you don't need to configure or even import `dask` at all to leverage its parallelization power.

In this tutorial we have cross-matched sections of catalogs in a particular region of sky, where most of the stars likely fall in the same or neighboring sky cells. It would be useful in the future to try this using generic samples of stars from across the entire sky. How different is performance when the catalog rows come from many sky cells instead of just a few? We have also generated the cross-match using `compute`, which loads the full result into memory, but `lsdb` supports larger-than-memory cross-matches by writing them directly to disk using `to_hats`. It would also be illustrative to run these benchmarks, and benchmarking a full cross-match between ZTF and Pan-STARRS, by writing the result to disk.

+++

# About this Notebook

This notebook was authored by [Zach Claytor](mailto:zclaytor@stsci.edu), Astronomical Data Scientist at Space Telescope Science Institute.

+++

# References

* This work uses [`astropy`](https://www.astropy.org/acknowledging.html).
