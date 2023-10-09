Synthesis methodology
=====================

This page presents the scientific procedure to follow in order to synthesize morphologies.

Overview
--------

Production workflow
~~~~~~~~~~~~~~~~~~~
For specific mtypes and brain regions, the standard procedure to synthesize morphologiesis the following:

* collect biological data via an at least curated morphology release with the code
  available at https://github.com/BlueBrain/morphology-workflows.
* run this workflow and change default parameters via files in the git repository SynthDB
  available at https://bbpgitlab.epfl.ch/neuromath/synthdb.
* once the parametrisation is satisfactory, save the parameter and distributions json
  files on the SynthDB database.
* these files are then available in SynthDB to run production workflows, such as circuit-build


Internal workflows
~~~~~~~~~~~~~~~~~~

The main workflow that should be used is: :py:class:`tasks.workflows.ValidateSynthesis`.
This workflow runs all the required tasks to calibrate and validate the synthesis models.
It is divided in several subtasks (more details on these tasks are given in
:doc:`autoapi/tasks/index`):

.. graphviz:: autoapi/tasks/workflows/ValidateSynthesis.dot


If no information on the atlas is available, one can instead run the simpler vacuum
synthesis workflow: :py:class:`tasks.workflows.ValidateVaccumSynthesis`. It is also
divided in several subtasks (more details on these tasks are given in
:doc:`autoapi/tasks/index`):

.. graphviz:: autoapi/tasks/workflows/ValidateSynthesis.dot

Calibration parameters
----------------------

!!! Discuss custom_parameters.csv file here.

The calibration step should create the two parameter files used by
:py:class:`placement_algorithm.app.synthesize_morphologies.Master`:

* ``tmd_parameters.json`` which contains the model parameters for each ``mtype``.
* ``tmd_distributions.json`` which contains the distribution properties of each ``mtype``.

Details on the content of these files can be found here: <URL> (does not exist yet)

Synthesis in atlas
------------------

When cells are synthesized inside an atlas, their shapes must be adapted according to their
positions in order to fit in this atlas. Currently, the shapes are just rescaled in order
to fit in a defined interval. This interval depends on the ``mtype`` and on the cell position
because the depths of the atlas layers also depend on this position. The information on
how each ``mtype`` can be rescaled are inserted in the ``tmd_parameters.json`` file by the task
:py:class:`tasks.synthesis.AddScalingRulesToParameters`.

For a given ``mtype``, the parameters specific to the scaling process are contained in a
``context_constraints`` key. The keys in this object should be neurity types (one of
``[apical, basal, axon]``) and should contain the interval in which it must fit. This interval
is defined relatively to the atlas layers, so its two boundaries should contain a ``layer``
entry, which should be an integer in ``[1, 6]``, and an optional ``fraction`` entry, which
should be a float in ``[0, 1]`` (0 for the bottom boundary of the layer and 1 for the top
boundary).

For apical rescaling, another optional ``extent_to_target`` entry should be added. This
entry contains a target ``layer`` entry and an optional ``fraction`` entry as described
before. But it also contains the fit parameters of the path distance of the cell as a
function of its extent along its principal direction. This is needed because the stopping
criterion used in the TNS model only takes the path distance into account. This fit is
linear and is thus described by a ``slope`` and a ``intercept`` entries.

Here is an example of such ``context_constraints`` entry:

.. code-block:: json

    {
        "<mtype>": {
            "context_constraints": {
                "apical": {
                    "extent_to_target": {
                        "fraction": 0.8,
                        "intercept": 0,
                        "layer": 1,
                        "slope": 1.5
                    },
                    "hard_limit_max": {
                        "fraction": 1,
                        "layer": 1
                    },
                    "hard_limit_min": {
                        "fraction": 0.1,
                        "layer": 1
                    }
                },
                "basal": {
                    "hard_limit_max": {
                        "fraction": 0.5,
                        "layer": 1
                    }
                }
            }
        }
    }

More details on the models can be found here:

* `TNS <https://bbpteam.epfl.ch/documentation/projects/TNS/latest/index.html>`_
* <region-grower doc URL> (does not exist yet)
* <placement-algorithm.app.synthesis doc URL> (does not exist yet)
