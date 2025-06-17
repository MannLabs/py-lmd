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

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "py-lmd"
copyright = "2023 Georg Wallmann, Sophia Mädler and Niklas Schmacke"
author = "Georg Wallmann, Sophia Mädler and Niklas Schmacke"

# The full version, including alpha/beta/rc tags
release = "1.0.2"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx_rtd_theme", "nbsphinx", "sphinx_copybutton"]
nbsphinx_allow_errors = True

autodoc_mock_imports = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"

html_theme_options = {
    "collapse_navigation": False,
    "show_toc_level": 3,
    "navigation_depth": 4,
    "logo": {
        "image_light": "_static/pyLMD_text.svg",
        "image_dark": "_static/py-lmd_logo.png",
    },
}

html_title = "py-lmd"
html_static_path = ["_static"]

# placeholder to add favicon logo needs to be a png
# html_favicon = "_static/pyLMD_text.svg"

##----- OPtions for Latex output

latex_engine = "pdflatex"

latex_elements = {
    "extraclassoptions": "openany,oneside",
    "papersize": "a4paper",
    "pointsize": "10pt",
}
