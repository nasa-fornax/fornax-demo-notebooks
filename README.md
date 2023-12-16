# fornax-demo-notebooks
Tutorial notebooks of fully worked science use cases for the Fornax project

### Executive Summary
The Fornax Initiative is a NASA Astrophysics Archives project to collaboratively among the three archives HEASARC, IRSA, and MAST, create cloud systems, cloud software, and cloud standards for the astronomical community.
The Fornax Science Console is a cloud compute system near to NASA data in the cloud which provides a place where astronomers can do data intensive research with reduced barriers. The Fornax Initiative provides increased compute, increased memory, increased ease of use by pre-installing astronomical software, increased reprododicibility of big data results, increased inclusion by removing some of these barriers to entry, and tutorial notebooks and documentation.  This repo houses those tutorial notebooks of fully worked science use cases for all users.  Common goals of the use cases are archival data from all NASA archives, cross-archive work, big data, and computationally intensive science. 


### Content contributing

In this repository we use Jupytext and MyST Markdown Notebooks. You will need ``jupytext`` installed
for your browser to recognise the markdown files as notebooks (see more about the motivation and
technicalities e.g. here: https://numpy.org/numpy-tutorials/content/pairing.html).

If you already have an ``ipynb`` file, convert it to Markdown using the following command, and commit
only the markdown file to the repo:

```
jupytext --from notebook --to myst yournotebook.ipynb
```
