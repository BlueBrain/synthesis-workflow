"""Setup for the synthesis-workflow package."""

from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

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
    author="Blue Brain Project, EPFL",
    description="Workflow used for synthesis and its validation.",
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    url="https://synthesis-workflow.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/synthesis-workflow/issues",
        "Source": "https://github.com/BlueBrain/synthesis-workflow",
    },
    license="Apache License 2.0",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "synthesis-workflow=synthesis_workflow.tasks.cli:main",
            "morph-validation=morphval.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        # TODO: Update to relevant classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
