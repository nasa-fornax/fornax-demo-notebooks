# Configuration file for the Sphinx documentation builder.
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os


# -- Project information -----------------------------------------------------

project = 'Fornax Demo Notebooks'
copyright = '2022-2025, Fornax developers'
author = 'Fornax developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx_copybutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'notes', '.tox', '.tmp', '.pytest_cache']

# Top level README file's sole purpose is for the repo. We also don't include
# the data and output directories that are to be populated while running the notebooks.
exclude_patterns += ['README.md', 'notebook_review_checklists.md', '*/data/*', '*/output/*']

# Not yet included in the rendering:

exclude_patterns += ['spectroscopy/explore*', '*/code_src/*']

# Myst-NB configuration
# Override kernelspec.name for rendering for all the notebooks.
# "python3" kernel is created by ipython.
nb_kernel_rgx_aliases = {".*": "python3"}

nb_merge_streams = True

nb_execution_timeout = 900
nb_execution_excludepatterns = []

# Don't execute the forced photometry notebook until we base the CI on
# the actual fornax image instead of the fresh installs
# (aka tractor install pain).
nb_execution_excludepatterns += ['multiband_photometry.md',]

# We use the non-public IRSA bucket for ZTF data, cannot execute the collector
# notebook until https://github.com/nasa-fornax/fornax-demo-notebooks/issues/311 is addressed
nb_execution_excludepatterns += ['light_curve_collector.md',]


if 'CIRCLECI' in os.environ:
    # Workaround for e.g. https://github.com/nasa-fornax/fornax-demo-notebooks/issues/475
    # Some of the notebooks run into a DeadKernelError (hitting the memory limit) on CircleCI,
    # but do execute and render on GHA. Ignore them here.
    nb_execution_excludepatterns += ['light_curve_classifier.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_title = 'Fornax Demo Notebooks'
html_logo = '_static/fornax_logo.png'
html_favicon = '_static/fornax_favicon.ico'
html_theme_options = {
    "github_url": "https://github.com/nasa-fornax/fornax-demo-notebooks",
    "repository_url": "https://github.com/nasa-fornax/fornax-demo-notebooks",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "home_page_in_toc": True,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# myst configurations
myst_heading_anchors = 4
