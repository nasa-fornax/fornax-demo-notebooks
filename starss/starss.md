---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
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

```
