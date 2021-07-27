"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# pylint: disable=protected-access

import importlib
import re

import luigi
from pkg_resources import get_distribution

import morphval
import synthesis_workflow
import synthesis_workflow.tasks
from synthesis_workflow.tasks.cli import _PARAM_NO_VALUE
from synthesis_workflow.tasks.cli import _process_param

# -- Project information -----------------------------------------------------

project = "synthesis-workflow"

# The short X.Y version
version = get_distribution("synthesis_workflow").version

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "m2r2",
]

autoapi_dirs = [
    "../../src/morphval",
    "../../src/synthesis_workflow",
    "../../src/synthesis_workflow/tasks",
]
autoapi_ignore = [
    "*version.py",
]
autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "imported-members",
    "members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "undoc-members",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx-bluebrain-theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_theme_options = {
    "metadata_distribution": "synthesis-workflow",
}

html_title = "Synthesis Workflow"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# autosummary settings
autosummary_generate = True

# autodoc settings
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autoclass_content = "both"

add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "luigi": ("https://luigi.readthedocs.io/en/stable", None),
}

# Auto-API customization

SKIP = [
    r".*\.L$",
    r".*tasks\..*\.requires$",
    r".*tasks\..*\.run$",
    r".*tasks\..*\.output$",
    r".*tasks\.config.propagate$",
]

IMPORT_MAPPING = {
    "morphval": morphval,
    "synthesis_workflow": synthesis_workflow,
    "tasks": synthesis_workflow.tasks,
}


# pylint: disable=unused-argument,protected-access
def maybe_skip_member(app, what, name, obj, skip, options):
    """Skip and update documented objects."""
    skip = None
    for pattern in SKIP:
        if re.match(pattern, name) is not None:
            skip = True
            break

    if not skip:
        try:
            package, module, *path = name.split(".")
            root_package = IMPORT_MAPPING[package]
            actual_module = importlib.import_module(root_package.__name__ + "." + module)
            task = getattr(actual_module, path[-2])
            actual_obj = getattr(task, path[-1])
            if isinstance(actual_obj, luigi.Parameter):
                if hasattr(actual_obj, "description") and actual_obj.description:
                    help_str, param_type, choices, interval, optional = _process_param(actual_obj)
                    if optional:
                        help_str = "(optional) " + help_str
                    if param_type is not None:
                        help_str += f"\n\n:type: {param_type}"
                    if choices is not None:
                        help_str += f"\n\n:choices: {choices}"
                    if interval is not None:
                        help_str += f"\n\n:permitted values: {interval}"
                    if (
                        hasattr(actual_obj, "_default")
                        and actual_obj._default not in _PARAM_NO_VALUE
                    ):
                        help_str += f"\n\n:default value: {actual_obj._default}"
                    obj.docstring = help_str
        except Exception:  # pylint: disable=broad-except
            pass

    return skip


def setup(app):
    """Sphinx events setup."""
    app.connect("autoapi-skip-member", maybe_skip_member)
