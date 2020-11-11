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
    "atlas_analysis>0.0.1",
    "brainbuilder>=0.14",
    "bluepyefe",
    "bluepyopt",
    "bluepymm",
    "diameter_synthesis>=0.1.7",
    "gitpython",
    "h5py<3",
    "joblib",
    "luigi",
    "matplotlib",
    "morph_validator",
    "morphio",
    "neuroc",
    "neurom!=2.0.1.dev4",
    "pandas",
    "placement_algorithm>=2.1.1",
    "region_grower>=0.1.10",
    "scipy",
    "seaborn",
    "tns>=2.2.7",
    "tmd",
    "tqdm",
    "voxcell>=2.7.3",
]

doc_reqs = [
    "sphinx",
    "sphinx-autoapi",
    "sphinx-bluebrain-theme",
]

test_reqs = [
    "diff_pdf_visually",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-xdist",
]

VERSION = imp.load_source("", "src/version.py").VERSION

setup(
    name="synthesis-workflow",
    author="bbp-ou-cell",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    version=VERSION,
    description="Workflow used for synthesis and its validation.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/synthesis-workflow",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "ssh://bbpcode.epfl.ch/cells/synthesis-workflow",
    },
    license="BBP-internal-confidential",
    packages=find_packages("src", exclude=["tests"]),
    package_dir={"": "src"},
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
    include_package_data=True,
)
