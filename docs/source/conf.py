"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import importlib
import re
import subprocess
from importlib import metadata
from pathlib import Path

import luigi

import morphval
import synthesis_workflow
import synthesis_workflow.tasks
from synthesis_workflow.tasks import cli

# -- Project information -----------------------------------------------------

project_name = "Synthesis Workflow"
package_name = "synthesis-workflow"

# The short X.Y version
version = metadata.version(package_name)

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
    "sphinx.ext.todo",
    "sphinxarg.ext",
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
autoapi_keep_files = True
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

todo_include_todos = True

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
    "metadata_distribution": package_name,
}

html_title = project_name

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
    "data-validation-framework": (
        "https://data-validation-framework.readthedocs.io/en/stable",
        None,
    ),
    "diameter-synthesis": ("https://diameter-synthesis.readthedocs.io/en/stable", None),
    "luigi": ("https://luigi.readthedocs.io/en/stable", None),
    "luigi-tools": ("https://luigi-tools.readthedocs.io/en/stable", None),
    "neurom": ("https://neurom.readthedocs.io/en/stable", None),
    "neuror": ("https://neuror.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "python": ("https://docs.python.org/3", None),
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
            if (
                isinstance(actual_obj, luigi.Parameter)
                and hasattr(actual_obj, "description")
                and actual_obj.description
            ):
                obj.docstring = cli.format_description(
                    actual_obj,
                    default_str="{doc}\n\n:default value: {default}",
                    optional_str="(Optional) {doc}",
                    type_str="{doc}\n\n:type: {type}",
                    choices_str="{doc}\n\n:choices: {choices}",
                    interval_str="{doc}\n\n:permitted values: {interval}",
                )
        except Exception:  # pylint: disable=broad-except
            pass

    return skip


def generate_images(*args, **kwargs):
    """Generate images of the workflows."""
    cur_cwd = Path(__file__).parent

    # Create the required directories
    (Path(__file__).parent / "autoapi/tasks/validation").mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent / "autoapi/tasks/vacuum_synthesis").mkdir(parents=True, exist_ok=True)

    TEST_DIR = Path(*Path(__file__).parts[:-3]) / "tests"

    # Convert images
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-f",
            "5",
            "-l",
            "5",
            TEST_DIR / "data/in_small_O1/out/validation/scales/statistics.pdf",
            str(cur_cwd / "autoapi/tasks/validation/scale_statistics"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-f",
            "1",
            "-l",
            "1",
            TEST_DIR / "data/in_small_O1/out/validation/path_distance_fit.pdf",
            str(cur_cwd / "autoapi/tasks/validation/path_distance_fit"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-f",
            "1",
            "-l",
            "1",
            TEST_DIR
            / "data/in_vacuum/out/validation/morphometrics/morphometrics_basal_dendrite.pdf",
            str(cur_cwd / "autoapi/tasks/validation/morphometrics"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-f",
            "1",
            "-l",
            "1",
            TEST_DIR / "data/in_small_O1/out/validation/score_matrix_reports.pdf",
            str(cur_cwd / "autoapi/tasks/validation/score_matrix_reports"),
        ],
        check=True,
    )

    # Generate dependency graphs
    luigi_config = luigi.configuration.get_config()
    luigi_config.read(str(TEST_DIR / "data/in_small_O1/luigi.cfg"))
    cli.main(
        [
            "-dg",
            str(cur_cwd / "autoapi/tasks/workflows/ValidateSynthesis.dot"),
            "ValidateSynthesis",
        ]
    )
    cli.main(
        [
            "-dg",
            str(cur_cwd / "autoapi/tasks/workflows/ValidateRescaling.dot"),
            "ValidateRescaling",
        ]
    )
    luigi_config.read(str(TEST_DIR / "data/in_vacuum/luigi.cfg"))
    cli.main(
        [
            "-dg",
            str(cur_cwd / "autoapi/tasks/workflows/ValidateVacuumSynthesis.dot"),
            "ValidateVacuumSynthesis",
        ]
    )


def setup(app):
    """Setup Sphinx by connecting functions to events."""
    app.connect("builder-inited", generate_images)
    app.connect("autoapi-skip-member", maybe_skip_member)
