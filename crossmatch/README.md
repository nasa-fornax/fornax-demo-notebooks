# Cross-matching ZTF and Pan-STARRS 1 using LSDB


[LSDB](https://lsdb.io) is a useful package for performing large cross-matches between source catalogs.
It's built to run across multiple nodes with Dask parallelization, but even without parallelization it is high-performance.
In this use case demonstration, we will benchmark the performance of LSDB on the NASA Fornax platform with and without Dask.

```{toctree}
---
maxdepth: 1
---
ztf_ps1_crossmatch

```