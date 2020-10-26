#!/usr/bin/env python3

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

reqs = [
    "atlas_analysis",
    "brainbuilder",
    "bluepyefe",
    "bluepyopt",
    "bluepymm",
    "diameter_synthesis",
    "gitpython",
    "h5py",
    "joblib",
    "luigi",
    "matplotlib",
    "morph_validator",
    "morphio",
    "neuroc",
    "neurom",
    "pandas",
    "placement_algorithm",
    "region_grower",
    "scipy",
    "seaborn",
    "tns",
    "tmd",
    "tqdm",
    "voxcell",
]

doc_reqs = [
    "sphinx",
    "sphinx-bluebrain-theme",
]

test_reqs = [
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-mpl",
]

VERSION = imp.load_source("", "synthesis_workflow/version.py").__version__

setup(
    name="synthesis-workflow",
    author="bbp-ou-cell",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    version=VERSION,
    description="Workflow used for synthesis and its validation.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/synthesis-workflow",
    license="BBP-internal-confidential",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=reqs,
    extras_require={"docs": doc_reqs, "test": test_reqs},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": ["synthesis_workflow=synthesis_workflow.tasks.cli:main"]
    },
)
