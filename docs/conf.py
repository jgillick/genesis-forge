# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
import os

# Add genesis_forge to the import path
sys.path.insert(0, str(Path("..").resolve()))

# Mock imports for documentation building
autodoc_mock_imports = [
    "genesis",
    "torch",
    "numpy",
    "gymnasium",
    "pygame",
    "skrl",
    "rsl_rl",
    "gstaichi",
    "tensordict",
]

# Don't fail on missing references
need_sphinx = "7.0"
autodoc_default_options = {
    "class-doc-from": "both",
    "members": True,
    "inherited-members": True,
    "member-order": "groupwise",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "ignore-module-all": False,
}

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Genesis Forge"
copyright = "2025, Jeremy Gillick"
author = "Jeremy Gillick"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Autosummary settings - disabled for now as modules are still in development
autosummary_generate = False  # Disable autosummary to avoid import errors
autosummary_imported_members = False
autosummary_generate_overwrite = False
# autodoc_typehints = "none"  # Don't process type hints
autodoc_member_order = "groupwise"

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# PyData theme options
html_theme_options = {
    "github_url": "https://github.com/jgillick/genesis_forge",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "header_links_before_dropdown": 4,
    "primary_sidebar_end": ["indices.html"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "jgillick",
    "github_repo": "genesis-forge",
    "github_version": "main",
    "doc_path": "docs",
}

html_logo = "../images/logo.png"
html_favicon = "../images/logo.png"
html_title = "Genesis Forge Documentation"
