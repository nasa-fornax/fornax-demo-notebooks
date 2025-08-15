---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: py-light_curve_generator
  language: python
  name: py-light_curve_generator
---

# Title: How to Make a Tutorial Notebook in the fornax-demo-notebooks repo


## Learning Goals
By the end of this tutorial, you will be able to (list 2 - 5 high level goals):
  * write a python tutorial
  * meet all of the checklisted requirements to submit your code for code review


## Introduction:
Alter this file according to your use case but retain the basic structure and try to use the same syntax for section headings, bullet points, etc.

The Introduction should provide context and motivation. Why should someone use this notebook?  
Give background on the science or technical problem.  Point out the parts that are particularly challenging and what solutions we chose for what reasons.

## Input:
 * This tutorial template
 * List the data, catalogs, or files needed, and where they come from.

## Output:
 * List the products the notebook generates (plots, tables, derived data, etc.)

## Runtime:

Please report actual numbers and machine details for your notebook if it is expected to run longer or requires specific machines, e.g., on Fornax. Also, if querying archives, please include a statement like, "This runtime is heavily dependent on archive servers which means runtime will vary for users".)

Here is a template runtime statement
- As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Astrophysics Default Image’ and the ‘{name: size}’ server with NGB RAM/ NCPU.

## Authors:
Only Author names, or even teams is required here

## Acknowledgements:
Did anyone help you?  Probably these teams did, so inlcude them: MAST, HEASARC, & IRSA Fornax teams

Did you use AI for any part of this tutorial, if so please include a statement such as:
"AI: This notebook was created with assistance from OpenAI’s ChatGPT 5 model."

## Imports:
This should be a list of the modules that are required to run this code.  Importantly, even those that are already installed in Fornax should be listed here so users wanting to run this locally on their own machines have the information they need to do this.

Examples:
 * `acstools` to work with HST magnitude to flux conversion
 * `astropy` to work with coordinates/units and data structures
 * `astroquery` to interface with archives APIs
 * `hpgeom` to locate coordinates in HEALPix space
 * `scipy` to do statistics
 * `tqdm` to track progress on long running jobs

This cell will install them if needed:

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_light_curve_generator.txt

#make sure that you have built a requirements_notebook_name.txt file with these modules to be imported.
```

```{code-cell} ipython3
#example import statements

import sys
import time

import astropy.units as u
import pandas as pd
from astropy.table import Table

# local code imports
sys.path.append('code_src/')
#from data_structures import MultiIndexDFObject
```

If you have written functions, please take those out of this notebook and put them into a code_src directory in a .py file with some useful name, eg., data_structures.py or heasarc_functions.py

+++

## 1. Data Access

```{code-cell} ipython3
# Example: load a table from a local file
#data = Table.read("example_data.fits")
#data[:5]
```

```{code-cell} ipython3
sample_table
```

## 2. Data Exploration
Describe what the data look like. Add summary statistics, initial plots, sanity checks.

```{code-cell} ipython3
# Example: histogram of one column
#plt.hist(data["redshift"], bins=30)
#plt.xlabel("Redshift")
#plt.ylabel("Number of galaxies")
#
```

## 3. Analysis
The working part of the notebook. Lay out the step-by-step analysis workflow.  Each subsection should describe what is being done and why. These can be sections or subsections.

+++

### 3.1 Design Principles
* Make no assumptions: define terms, common acronyms, link to things you reference
* Keep in mind who your audience is
* Design for portability - will this notebook work on both Fornax and someone's individual laptop
* Cells capture logical units of work
* Use markdown before or after cells to describe what is happening in the notebook

+++

## 4. PR Review

Notebooks go through a two step process: first step is getting into the repo, and the second step gets it into the [published tutorials](https://nasa-fornax.github.io/fornax-demo-notebooks/). 

To complete the review for the first step, both a science and technical reviewer will be looking at this checklist to see if the new tutorial notebook meets all of the requirements, or has a reasonable excuse not to. Please consider these checklist requirements as you are writing your code.

+++

## References

This work made use of:

* STScI style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
* Fornax tech and science review guidelines:

+++

## About this notebook:

Last Updated: 2025-08-14

Contact: [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov) with questions or problems.

```{code-cell} ipython3

```
