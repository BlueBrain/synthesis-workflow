"""Setup for the synthesis-workflow package."""
import importlib
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "src.version",
    "src/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

# Read the requirements
with open("requirements/base.pip", "r", encoding="utf-8") as f:
    reqs = f.read().splitlines()

# Read the requirements for doc
with open("requirements/doc.pip", "r", encoding="utf-8") as f:
    doc_reqs = f.read().splitlines()

# Read the requirements for tests
with open("requirements/test.pip", "r", encoding="utf-8") as f:
    test_reqs = f.read().splitlines()

setup(
    name="synthesis-workflow",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    description="Workflow used for synthesis and its validation.",
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/synthesis-workflow",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow",
    },
    license="BBP-internal-confidential",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    version=VERSION,
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "synthesis_workflow=synthesis_workflow.tasks.cli:main",
            "morph_validation=morphval.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
