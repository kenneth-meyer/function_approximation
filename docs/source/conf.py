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
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'Function Approximation Exploration'
copyright = '2024, Kenneth Meyer'
author = 'Kenneth Meyer'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------
extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.napoleon',
'sphinx.ext.autosummary',
'sphinx_markdown_builder',
'myst_parser',
'sphinx.ext.mathjax',
'sphinx_toolbox.decorators',
'sphinx_math_dollar'
]


mathjax3_config = {
    "tex": {
        "inlineMath": [['\\(', '\\)']],
        "displayMath": [["\\[", "\\]"]],
    }
}

templates_path = ['_templates']
exclude_patterns = []

# adding this from the dolfinx conf.py
myst_enable_extensions = [
    "dollarmath",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # alabaster
html_static_path = ['_static']

# process any tutorials by calling the method jupytext_process.process()
sys.path.insert(0,'.') # current directory; need to be able to find jupytext_process