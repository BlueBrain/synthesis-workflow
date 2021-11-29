"""Setup for the synthesis-workflow package."""
import importlib
import sys

from setuptools import find_packages
from setuptools import setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

# Read the requirements
with open("requirements/base.pip") as f:
    reqs = f.read().splitlines()

# Read the requirements for doc
with open("requirements/doc.pip") as f:
    doc_reqs = f.read().splitlines()

# Read the requirements for tests
with open("requirements/test.pip") as f:
    test_reqs = f.read().splitlines()

VERSION = importlib.import_module("src.version").VERSION

setup(
    name="synthesis-workflow",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    version=VERSION,
    description="Workflow used for synthesis and its validation.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/synthesis-workflow",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow",
    },
    license="BBP-internal-confidential",
    packages=find_packages("src", exclude=["tests"]),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=reqs,
    test_requires=test_reqs,
    extras_require={"docs": doc_reqs, "test": test_reqs},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "synthesis_workflow=synthesis_workflow.tasks.cli:main",
            "morph_validation=morphval.cli:main",
        ]
    },
    include_package_data=True,
)
