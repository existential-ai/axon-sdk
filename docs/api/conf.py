import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../../'))  # path to your SDK

# -- Project information -----------------------------------------------------

project = 'Axon SDK'
copyright = f'{datetime.now().year}, Neucom ApS'
author = 'Neucom ApS'
release = '0.1.0'  # or dynamically get this from your package

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # for Google-style/NumPy-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary'
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'  # or 'sphinx_rtd_theme', 'furo', etc.
#html_static_path = ['_static']

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True
}

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
