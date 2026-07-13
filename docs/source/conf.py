# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "pyAOBS"
copyright = "2024–2026, Haibo Huang"
author = "Haibo Huang"
release = "3.0.0rc2"
version = "3.0.0rc2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "../../images/logo.png"
html_favicon = None
html_title = f"{project} {release} documentation"
html_short_title = "pyAOBS"

autodoc_mock_imports = [
    "PySide6",
    "dearpygui",
    "pygmt",
    "obspy",
    "burnman",
    "sklearn",
    "seaborn",
    "openpyxl",
    "numba",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

language = "en"
