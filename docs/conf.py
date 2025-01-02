# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aide'
copyright = '2024, Evan Komp'
author = 'Evan Komp, Gregg T. Beckham'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.todo', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Add autodoc options to exclude private members
autodoc_default_options = {
    'members': True,
    'undoc-members': False,  # Don't include members without docstrings
    'private-members': False,  # Don't include _private members
    'special-members': False,  # Don't include __special__ members
    'inherited-members': True,
    'show-inheritance': True,
    'exclude-members': '_abc_impl'  # Exclude specific members if needed
}

# Skip all _private members
def skip_private_members(app, what, name, obj, skip, options):
    if name.startswith('_'):
        return True
    return skip

# Connect the function to the autodoc-skip-member event
def setup(app):
    app.connect('autodoc-skip-member', skip_private_members)


napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Add these configurations
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Myst settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
# Enable math rendering and processing
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
myst_dmath_double_inline = True
myst_update_mathjax = True
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
