# fornax-demo-notebooks
Demo notebooks for the Fornax project

### Executive Summary
HEASARC, IRSA, and MAST jointly propose an FY22 project to demonstrate how NASA
Astrophysics mission data can be accessed from the cloud or on premises through a science
platform environment. We will center this demonstration on a limited set of data that will be
held in the NASA cloud, the AURA cloud, and on premises. We will build a suite of
containerized software, Jupyter notebooks, and Python libraries that will allow users to carry
out forced photometry on multiple NASA data sets, motivated by important science use
cases for mining survey data for extragalactic science and cosmology. This suite of data
access and analysis tools will be designed to be used in any of a number of science
platforms that are available or in development across the world. We will showcase its use in
at least two notebook environments, one of which will be cloud-hosted. We describe a simple
management structure for coordinating this work across all three archives and NASA. Finally,
we will use these experiences in further consultation with NASA to create an FY23 plan for
building an operational science platform within the NASA Cloud.


### Content contributing

In this repository we use Jupytext and MyST Markdown Notebooks. You will need ``jupytext`` installed
for your browser to recognise the markdown files as notebooks (see more about the motivation and
technicalities e.g. here: https://numpy.org/numpy-tutorials/content/pairing.html).

If you already have an ``ipynb`` file, convert it to Markdown using the following command, and commit
only the markdown file to the repo:

```
jupytext --from notebook --to myst yournotebook.ipynb
```
