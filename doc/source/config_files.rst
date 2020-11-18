Configuration files
===================

Each workflow needs a different configuration, depending on the tasks that are included in it.

Example for the :py:class:`tasks.workflows.ValidateVacuumSynthesis` workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../tests/data/in_vacuum/luigi.cfg
  :language: ini

Example for the :py:class:`tasks.workflows.ValidateSynthesis` workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../tests/data/in_small_O1/luigi.cfg
  :language: ini
