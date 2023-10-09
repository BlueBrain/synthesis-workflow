Synthesis Workflow
==================

This project contains several workflows used for neuron synthesis and the validation of this process.
It is divided into two packages:

* **synthesis-workflow**, the main package, which contains the workflow tasks and tools.
* **MorphVal**, which is a legacy library used for morphology validation.

The complete documentation of this project is available here: `<https://bbpteam.epfl.ch/documentation/projects/synthesis-workflow/latest/index.html>`_


Installation
------------

This package rely on several internal requirements which can be found in the BBP devpi index.
This index must be specified to pip:

.. code::

    pip install --index-url https://bbpteam.epfl.ch/repository/devpi/simple synthesis-workflow


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
	Complete examples for BBP usecases are provided here: `<https://bbpcode.epfl.ch/browse/code/project/proj82/tree/entities/synthesis_workflow?h=refs/heads/master>`_

Morphology validation
~~~~~~~~~~~~~~~~~~~~~

The usual command is the following:

.. code::

    morph_validation -t <path to reference data> -r <path to test data> -o <output path> -c <YAML config file> --bio-compare

You can get help and complete parameter description with the following command:

.. code::

    morph_validation --help
