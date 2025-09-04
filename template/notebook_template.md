---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Title: How to Make a Tutorial Notebook in the fornax-demo-notebooks repo

## Learning Goals

By the end of this tutorial, you will be able to (list 2 - 5 high level goals):

-   Write a python tutorial using [MyST markdown](https://mystmd.org) format.
-   Meet all of the checklist requirements to submit your code for code review.

## Introduction

Alter this file according to your use case but retain the basic structure and try to use the same syntax for things like section headings, numbering schemes, and bullet points.
Specifically the headings in this Intro section should not be edited to maintain consistency between notebooks.

All contributed notebooks should be in [MyST markdown](https://mystmd.org) format.  See the [Fornax documentation](https://docs.fornax.sciencecloud.nasa.gov/markdown-and-code-dev) for more info about this.

The Introduction should provide context and motivation.
Why should someone use this notebook?
Give background on the science or technical problem.
Point out the parts that are particularly challenging and what solutions we chose for what reasons.

### Input

-   List the data, catalogs, or files needed, and where they come from.
    If there are data that get downloaded to Fornax as part of this notebook, place those in a `data` directory.
    Please do not change the name of this directory for consistency with other notebooks.
    Do not add the contents of `data` to the repo, just the empty directory.

### Output

-   List the products the notebook generates (plots, tables, derived data, etc.)
-   If there are intermediate products produced by your notebook, generate an `output` directory for those data.
    Please do not change the name of this directory for consistency with other notebooks.
    Do not add the contents of `output` to the repo, just the empty directory.

### Runtime

Please report actual numbers and machine details for your notebook if it is expected to run longer or requires specific machines, e.g., on Fornax.
Also, if querying archives, please include a statement like:
"This runtime is heavily dependent on archive servers which means runtime will vary for users".

Here is a template runtime statement:
As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Astrophysics Default Image’ and the ‘{name: size}’ server with NGB RAM/ NCPU.

## Imports

This should be a list of the modules that are required to run this code.
Importantly, even those that are already installed in Fornax should be listed here so users wanting to run this locally on their own machines have the information they need to do this.

Make sure that you have built a requirements_notebook_name.txt file with the modules to be imported.  
The name of the notebook should be present in the name of the requirements file, as in our example "requirements_template.txt"

```{code-cell} ipython3
# This cell should not be edited below this line except for the name of the requirements_notebook_name.txt

# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_template.txt
```

```{code-cell} ipython3
import numpy  # Create example data and make a histogram
```

If you have written functions, please take those out of this notebook and put them into a code_src directory in a .py file with some useful name, eg., data_structures.py or heasarc_functions.py

+++

## 1. Data Access
The name of this, and all future sections can change.
In general, it probably is a good idea to start with something like "Data Access".
Pleae note, and stick to, the existing numbering scheme.

```{code-cell} ipython3
# Create some example data.
data = np.random.randint(0, 100, size=100)
```

## 2. Data Exploration

Describe what the data look like. Add summary statistics, initial plots, sanity checks.

For cuts or other data filtering and cleaning steps, explain the scientific reasoning behind them. 
This helps people understand both the notebook and the data so that they're more equipped to use the data appropriately in other contexts.

+++

:::{tip}
Please include narrative along with *all* your code cells to help the reader figure out what you are doing and why you chose that path.

Using [MyST admonitions](https://mystmd.org/guide/admonitions) such as this `tip` are encouraged 
:::

```{code-cell} ipython3
hist, bin_edges = np.histogram(data, bins=10)
hist
```

For any Figures, please add a few sentences about what the users should be noticing.

+++

## 3. Analysis

The working part of the notebook.
Lay out the step-by-step analysis workflow.
Each subsection should describe what is being done and why.
These can be sections or subsections.

+++

### 3.1 Design Principles

-   Make no assumptions: define terms, common acronyms, link to things you reference.
-   Keep in mind who your audience is.
-   Design for portability - will this notebook work on both Fornax and someone's individual laptop.
-   Cells capture logical units of work.
-   Use markdown before or after cells to describe what is happening in the notebook.

+++

### 3.2 Style Principles

-   Follow suggestions of The Turing Way community [markdown style](https://book.the-turing-way.org/community-handbook/style)
-   Write each sentence in a new line (line breaks) to make changes easier to read in PRs.
-   Avoid latin abbreviation to avoid failing CI.
-   Section titles should not end with ":".
-   List items should start at the beginning of the line, no spaces first. Exception is nested lists.
-   One empty line between section header and text.
-   One empty line before a list and after.
-   No more than one empty line between any two non-empty lines.
  

+++

## 4. PR Review

Notebooks go through a two step process: first step is getting into the repo, and the second step gets it into the [published tutorials](https://nasa-fornax.github.io/fornax-demo-notebooks/).

To complete the review for the first step, both a science and technical reviewer will be looking at [this checklist](https://github.com/nasa-fornax/fornax-demo-notebooks/blob/main/template/notebook_review_checklists.md) to see if the new tutorial notebook meets all of the requirements, or has a reasonable excuse not to.
Please consider these checklist requirements as you are writing your code.

+++

## About this notebook

-   Authors: Only Author names, or even teams is required here
-   Contact: [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov) with questions or problems.
-   Please edit and keep the above 2 bullet points, and remove this last line.

+++

### Acknowledgements

Did anyone help you?
Probably these teams did, so include them: MAST, HEASARC, & IRSA Fornax teams.

Did you use AI for any part of this tutorial, if so please include a statement such as:
"AI: This notebook was created with assistance from OpenAI’s ChatGPT 5 model.", which is a good time to mention that this template notebook was created with assistance from OpenAI’s ChatGPT 5 model.

### References

This work made use of:

-   STScI style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
-   Fornax tech and science review guidelines: https://github.com/nasa-fornax/fornax-demo-notebooks/blob/main/template/notebook_review_checklists.md
-   The Turing Way Style Guide: https://book.the-turing-way.org/community-handbook/style

```{code-cell} ipython3

```
