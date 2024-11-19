# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'proton_diffusion'
copyright = '2024, Atsushi Ishikawa'
author = 'Atsushi Ishikawa'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',      # 数式(latex)を使う場合
    'sphinx.ext.githubpages',  # githubを使う場合
    'myst_parser',             # markdownを使う場合
]

myst_enable_extensions = ["amsmath"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # テーマを変更
html_static_path = ['_static']
html_show_copyright = False # copyrightを消す場合
html_show_sphinx = False    # sphinxで作成...を消す場合

