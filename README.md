## Fornax Demo Notebooks

The Fornax Initiative is a NASA Astrophysics Archives project to collaboratively among the three archives HEASARC, IRSA, and MAST, create cloud systems, cloud software, and cloud standards for the astronomical community.

The Fornax Science Console is a cloud compute system near to NASA data in the AWS cloud which provides a place where astronomers can do data-intensive research with reduced barriers. The Fornax Initiative provides increased compute, increased memory, increased ease of use by pre-installing astronomical software, increased reproducibility of big data results, and increased inclusion by removing some of these barriers to entry, tutorial notebooks, and documentation.

This repo houses tutorial notebooks of fully worked science use cases for all users.  Common goals of the notebooks are the usage of archival data from all NASA archives, cross-archive work, big data, and computationally intensive science. Currently, there are two major topics for which we have notebooks.  The "Photometry" notebook starts with a catalog and a set of archival images.  The notebook grabs all necessary images and measures photometry at all positions listed in the catalog on all images.  The "Time Domain" notebooks are twofold.  The first generates light curves from all available archival data for any user-supplied sample of targets.  The second notebook runs ML algorithms to classify those generated light curves.


## Documentation

The user cases and documentation of the Fornax Initiative are currently available at the https://nasa-fornax.github.io/fornax-demo-notebooks/ URL, while the source code for the documentation can be found in the [fornax-documentation](https://github.com/nasa-fornax/fornax-documentation) repository.

## Content contributing

In this repository, we follow the standard practice of the Scientific Python ecosystem and use Jupytext and MyST Markdown Notebooks.

Please visit [the upstream documentation](https://mystmd.org/guide/md-vs-ipynb) to learn more about the reasoning behind the choice.
In summary, we chose MyST Markdown because we need a clear and human-readable format that makes version control, diffs, and collaborative reviews of code and narrative straightforward.
It is also ideal for tutorials and CI-backed projects where reproducible, build-time execution of code ensures continued functionality.

We also highlight that you can easily have the same user experience in JupyterLab if these two dependencies are installed:
 - [jupytext library](https://pypi.org/project/jupytext/)
 - [jupyterlab-myst JupyterLab extension](https://pypi.org/project/jupyterlab-myst/)

If you already have an ``ipynb`` file, convert it to Markdown using the following command, and commit
only the markdown file to the repo:

```
jupytext --from notebook --to myst yournotebook.ipynb
```
