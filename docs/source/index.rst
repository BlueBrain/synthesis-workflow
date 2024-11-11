.. include:: ../../README.rst

Scientific methodology for synthesis
------------------------------------

The :doc:`./synthesis_methodology` page contains detailed documentation of the methodology
on which this package is based.


Workflows
---------

The package :py:mod:`synthesis_workflow` contains many tasks that are organized in workflows.
The two main workflows are the following:

* :py:class:`tasks.workflows.ValidateVacuumSynthesis`: this workflow generates cells
  in vacuum (e.g. with no atlas information) and compute some simple validation features.
* :py:class:`tasks.workflows.ValidateSynthesis`: this workflow generates cells in a
  given atlas and compute many validation features.

All workflows need configuration files. Some examples are given in :doc:`./config_files`.

More details on these workflows and their tasks can be found in the :ref:`API Reference`.


.. _API Reference:

API Reference
-------------

The :doc:`./api_ref` page contains detailed documentation of:

* :py:mod:`main workflows<tasks.workflows>`
* :py:mod:`all workflow tasks<tasks>`
* :py:mod:`all synthesis_workflow functions<synthesis_workflow>`
* :py:mod:`all morphval functions<morphval>`

.. note::
	:py:mod:`tasks` appears as a package in this documentation but it is actually a subpackage
	of the :py:mod:`synthesis_workflow` package.


.. toctree::
   :hidden:
   :maxdepth: 2

   Home <self>
   synthesis_methodology
   config_files
   cli
   api_ref
   changelog
   contributing
