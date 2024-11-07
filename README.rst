Synthesis Workflow
==================

This project contains several workflows used for neuron synthesis and the validation of this process.
It is divided into two packages:

* **synthesis-workflow**, which contains the workflow tasks and tools.
* **MorphVal**, which is a library used for morphology validation and can be used as a standalone.


Installation
------------
To install:

.. code::

    pip install synthesis-workflow


Usage
-----

Synthesis workflow
~~~~~~~~~~~~~~~~~~

The usual command is the following:

.. code::

    synthesis_workflow <workflow>

You can get help and complete parameter description with the following commands:

.. code::

    synthesis_workflow --help
    synthesis_workflow <workflow> --help

You can also run a complete ``luigi`` command in order to fine-control task parameters:

.. code::

    luigi --module synthesis_workflow.tasks.workflows --help
    luigi --module synthesis_workflow.tasks.workflows <workflow> --help
    luigi --module synthesis_workflow.tasks.workflows <workflow> [specific arguments]

.. note::

	The ``synthesis_workflow`` command (or the complete ``luigi`` command) must be
	executed from a directory containing a ``luigi.cfg`` file.
	A simple example of such file is given in the ``examples`` directory.

Morphology validation
~~~~~~~~~~~~~~~~~~~~~

The usual command is the following:

.. code::

    morph_validation -t <path to reference data> -r <path to test data> -o <output path> -c <YAML config file> --bio-compare

You can get help and complete parameter description with the following command:

.. code::

    morph_validation --help

Funding & Acknowledgment
~~~~~~~~~~~~~~~~~~~~~~~~

The development of this software was supported by funding to the Blue Brain Project,
a research center of the École polytechnique fédérale de Lausanne (EPFL),
from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2022-2024 Blue Brain Project/EPFL
