#!/usr/bin/env python3

import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

# Read the requirements
with open("requirements.pip") as f:
    reqs = f.read().splitlines()

# Read the doc requirements
with open("requirements-doc.pip") as f:
    doc_reqs = f.read().splitlines()

# Read the test requirements
with open("requirements-test.pip") as f:
    test_reqs = f.read().splitlines()

setup(
    name="synthesis-workflow",
    author="bbp-ou-cell",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    use_scm_version=True,
    description="Workflow used for synthesis and its validation.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/synthesis-workflow",
    license="BBP-internal-confidential",
    install_requires=reqs,
    packages=find_packages(),
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs
    },
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
)
