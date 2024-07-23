# Configuration file for the Sphinx documentation builder.
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Fornax Demo Notebooks'
copyright = '2022-2024, Fornax developers'
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
exclude_patterns += ['README.md', '*/data/*', '*/output/*']

# We exclude the documentation index.md as its sole purpose is for their CI.
exclude_patterns += ['documentation/index.md',]

# Not yet included in the rendering:
exclude_patterns += ['documentation/notebook_review_process.md', 'spectroscopy/*', '*/code_src/*']

# Myst-NB configuration
nb_execution_timeout = 900
#nb_execution_excludepatterns = ['multiband_photometry.md']

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
    "use_edit_page_button": True,
    "home_page_in_toc": True,
#    "logo_link_url": "https://astroML.org",
#    "logo_url": "http://www.astroml.org/_images/plot_moving_objects_1.png"
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# myst configurations
myst_heading_anchors = 4
