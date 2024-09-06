---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: root *
  language: python
  name: conda-root-py
---

# Will figure out the name later

By the IPAC Science Platform Team, last edit: Sep 5, 2024

***


## Learning Goals

```
By the end of this tutorial, you will:

- Work with LSDB to read in large datasets lazily
- Do non-spatial selection of targets
- Visualize and compare different samples of stars?
```


## Introduction

TBD
- starting with https://lsdb.readthedocs.io/en/stable/tutorials/getting_data.html 
- for filtering with pandas or pyarrow: https://irsa.ipac.caltech.edu/docs/notebooks/wise-allwise-catalog-demo.html
- usecase?

## Imports
Here are the libraries used in this network. They are also mostly mentioned in the requirements in case you don't have them installed.
- *sys* and *os* to handle file names, paths, and directories
- *numpy*  and *pandas* to handle array functions
- *matplotlib* *pyplot* and *cm* for plotting data
- *astropy.io fits* for accessing FITS files
- *astropy.table Table* for creating tidy tables of the data


This cell will install them if needed:

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
#!pip install -r requirements_stars.txt
```

```{code-cell} ipython3
import lsdb
```

```{code-cell} ipython3
gaia_dr3 = lsdb.read_hipscat("https://data.lsdb.io/unstable/gaia_dr3/gaia/")
gaia_dr3
```

```{code-cell} ipython3
print(gaia_dr3.columns.tolist())
```

```{code-cell} ipython3
gaia_dr3 = lsdb.read_hipscat(
    "https://data.lsdb.io/unstable/gaia_dr3/gaia/",
    margin_cache="https://data.lsdb.io/unstable/gaia_dr3/gaia_10arcs/",
    columns=[
        "source_id",
        "ra",
        "dec",
        "phot_g_mean_mag",
        "phot_rp_mean_mag",
        "phot_bp_mean_mag",
        "phot_proc_mode",
        "azero_gspphot",
        "teff_gspphot",
        "classprob_dsc_combmod_star",
    ],
)
gaia_dr3
```

```{code-cell} ipython3
gaia_dr3.plot_pixels("Gaia DR3 Pixel Map")
```

```{code-cell} ipython3
gaia_dr3.dtypes
```

```{code-cell} ipython3
from dask.distributed import Client

client = Client(n_workers=4, memory_limit="auto")
client
```

```{code-cell} ipython3

```
