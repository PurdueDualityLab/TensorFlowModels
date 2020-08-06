# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../venv/lib/python3.7/site-packages'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Purdue CAM2 TensorFlow Model Gardeners'
copyright = '2020, Purdue CAM2'
author = 'Purdue CAM2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.napoleon',
'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster' #'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# https://github.com/mr-ubik/tensorflow-intersphinx
# https://stackoverflow.com/a/37444321
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "tensorflow": (
        "https://www.tensorflow.org/versions/r2.2/api_docs",
        "https://github.com/mr-ubik/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
}

napoleon_use_param = False
