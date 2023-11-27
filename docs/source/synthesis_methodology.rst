Synthesis methodology
=====================

This page presents the scientific procedure to follow in order to synthesize morphologies.


How to use synthesis from scratch ?
-----------------------------------

For specific mtypes and brain regions, the standard procedure to synthesize morphologies is as follow:

1. Collect biological data via an curated morphology release (see https://github.com/BlueBrain/morphology-workflows) that consists of a folder of morphologies and a neurondb.xml file (see https://github.com/BlueBrain/morph-tool/blob/master/morph_tool/morphdb.py).
2. Create a Pull Request on https://bbpgitlab.epfl.ch/neuromath/synthdb with a new folder under 'insitu_synthesis_inputs' to store specific configuration files.
3. Run the workflow :py:class:`tasks.workflows.ValidateVacuumSynthesis` to by pointing the `GetSynthesisInputs` task to this repository/branch/folder. Assess if synthesis works as expected with default values. If not, the synthesis parameters of `tmd_parameters.json` can be modified via a `custom_parameters.csv` file (see below). Additional files include `substitution_rules.yaml` to create or complete mtypes and `scaling_rules.yaml` to control barcode scalings insitu.
4. (Optional, to run insitu synthesis) Run the workflow :py:class:`tasks.workflows.ValidateSynthesis` after having added more files to the config repository, including `cell_composition.yaml`, `mtype_taxonomy.tsv` and `region_structure.yaml`. The latter corresponds to the atlas/region under consideration.
5. Once the parametrisation is satisfactory, save a reduced version of the luigi.cfg in SynthDB under `synthesis_inputs/luigi_configs` and run the cli `synthb synthesis-inputs build` to create and save the parameter and distributions json files on the SynthDB database.
6. Use `synthdb pull` cli to get the necessary synthesis input files. Also copy the other configuration files and run either region-grower cli (see https://bbpgitlab.epfl.ch/neuromath/region-grower) or circuit-build (see https://bbpgitlab.epfl.ch/nse/circuit-build) to synthesize an entire region.


Two synthesis workflows
-----------------------

1. When no information on the atlas is available, one can run the vacuum
synthesis workflow :py:class:`tasks.workflows.ValidateVacuumSynthesis`.
This workflow runs all the required tasks to calibrate and validate the synthesis models.
It is divided in several subtasks (more details on these tasks are given in
:doc:`autoapi/tasks/index`):

.. graphviz:: autoapi/tasks/workflows/ValidateSynthesis.dot

2. For insitu synthesis the workflow is :py:class:`tasks.workflows.ValidateSynthesis`.
This workflow runs all the required tasks to calibrate and validate the synthesis models.
It is divided in several subtasks (more details on these tasks are given in
:doc:`autoapi/tasks/index`):

.. graphviz:: autoapi/tasks/workflows/ValidateSynthesis.dot


How to calibrating synthesis parameters ?
-----------------------------------------

The workflow will create the parameter file ``tmd_parameters.json`` from the default values in https://github.com/BlueBrain/NeuroTS/blob/main/neurots/extract_input/input_parameters.py.
It will then modify a few entryes. One time with scaling rules, if a `scaling_rules.yaml` file is present and another time with trunk angle data obtained from fits.

All parameters can be modified via the file ``custom_parameters.csv`` in SynthDB as follow.
Each line corresponds to a parameter modification. It should have three columns ``mtype``, ``entry``, ``value``. The ``mtype`` column should match the ``mtype`` under consideration, ``entry`` column points to the parameter entry, with ``.`` referencing nested dictionaries. ``value`` is the parameter value to assign. If the value is a list, then ``entry`` can be of the form ``parameter.data[1]`` to access the second element.
These rules are appled to ``tmd_parameters.json`` after the scaling rules, and before the trunk angles (as trunk angle is not enabled by default, see below).

Setting non-default trunk angle algorithm
------------------------------------------

There are two algorithms to assign trunk angles, the default one, which is not configurable and another one that takes into account relative angles between neurite types.
Notice that if one neurite_type uses this non-default algorithm, all other neurite_types must use it as well.
To enable the second one, one can set specific ``orientation`` parameters in ``custom_parameters.csv`` file as follow.

The orientations are always assigned with respect to a direction, the pia or apical.
For the pia, use for example:

.. code-block::

    L1_DAC,basal_dendrite.orientation.mode,pia_constraint

and for apical

.. code-block::

    L2_IPC,basal_dendrite.orientation.mode,apical_constraint

with these modes, the algorithm will try to fit a probability distribution to the data, save the fit values in the ``tmd_parameters.json`` to later be used during synthesis.
By default, the fit function is a single step, corresponding to unimodal angle distribution, but if it is bimodal, one can use another function ``double_step`` as

.. code-block::

    L23_LBC,basal_dendrite.orientation.values.form,double_step


Another mode exists, that does not perform any fit. It is used to set particular directions for trunks, such as apical trunks, for example:

.. code-block::

    L5_TPC:A,apical_dendrite.orientation.mode,normal_pia_constraint
    L5_TPC:A,apical_dendrite.orientation.values.direction.mean,0.0
    L5_TPC:A,apical_dendrite.orientation.values.direction.std,0.0

will assign apical trunks to exactly the pia direction. Std will allow some randomness. If ``mean`` is not ``0`` of ``pi`` (and ``std=0``, the angles are on a cone).

Another example is for inverted  PC cell

.. code-block::

    L2_IPC,apical_dendrite.orientation.mode,normal_pia_constraint
    L2_IPC,apical_dendrite.orientation.values.direction.mean,3.1415
    L2_IPC,apical_dendrite.orientation.values.direction.std,0.3
    L6_IPC,basal_dendrite.orientation.mode,apical_constraint
    L6_IPC,grow_types[0],apical_dendrite
    L6_IPC,grow_types[1],basal_dendrite

which has basal dendrite trunk relative to the apical trunk, but as by default the basal are generated first (basal, apical then axon if available), one must revert the ordering in ``grow_types``.

Finally, for multiple apical trunks, one can use lists for ``mean`` and ``std`` parameters, as for BPC cells:

.. code-block::

    L6_BPC,apical_dendrite.orientation.mode,normal_pia_constraint
    L6_BPC,apical_dendrite.orientation.values.direction.mean[0],0.0
    L6_BPC,apical_dendrite.orientation.values.direction.std[0],0.2
    L6_BPC,apical_dendrite.orientation.values.direction.mean[1],2.5
    L6_BPC,apical_dendrite.orientation.values.direction.std[1],0.3


Scaling rules for basic insitu synthesis
----------------------------------------

When cells are synthesized inside an atlas, their shapes must be adapted according to their
positions in order to fit in this atlas.  These rules are the synthesis version of placement hint algorithm of https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/methodology.html.
With scaling rules, only the barcodes are rescaled in order to fit the atlas.
To define the scaling rules, one must have a file ``scaling_rules.yaml`` in synthdb folder with the following form.

.. code-block:: yaml

    default:  # this will be applied on all mtypes if corresponding key is not present below
        apical_dendrite:
            hard_limit_max:
                layer: L1
                fraction: 0.99
        basal_dendrite:
            hard_limit_max:
                layer: L1
                fraction: 0.99

    L2_TPC:A:
        apical_dendrite:
            hard_limit_min:
                layer: L1
                fraction: 0.1
            extent_to_target:
                layer: L1
                fraction: 0.8


For each ``mtype`` and ``neurite_type`` there can be ``hard_limit_min``, ``hard_limit_max``
or ``extent_to_target`` rules. For each the entry the ``layer`` argument of the form ``L[1-6]``
(not the names of the layer, just number from top to bottom) specifies the layer where the rule applies,
and ``fraction`` refines it to a fraction of it (0 for bottom, 1 for top).
For the ``hard_limit`` rules, the synthesized cells will be rescaled to so that there maximum/minimum extent
fit the rule.

The other mode ``extent_to_target``, mostly used for apical dendrites uses a fit of expected extent from barcodes,
and rescales the barcodes so that it will fit the rule. This rescaling requires expected thicknesses of synthesis, that can be provided in ``region_structure.yaml``, see below.


Minimal region structure information
-------------------------------------

In addition to a working atlas folder (with at least ``hierarchy.json``, ``[PH][layers].nrrd``, ``brain_region.nrrd``, ``orientations.nrrd``, if no cell densities file are present, we will add uniform ones in appropriate layers),
one needs additional information to run synthesis in this atlas, which we encode in ``region_structure.yaml`` file.
An example of this file for an single column, considered are a region names ``O0`` is:

.. code-block:: yaml

  O0:
    layers:
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    names:
        1: layer 1
        2: layer 2
        3: layer 3
        4: layer 4
        5: layer 5
        6: layer 6
    region_queries:
        1: '@.*1$'
        2: '@.*2[a|b]?$'
        3: '@.*3$'
        4: '@.*4$'
        5: '@.*5$'
        6: '@.*6[a|b]?$'
    thicknesses:
        1: 165
        2: 149
        3: 353
        4: 190
        5: 525
        6: 700

The entry ``layers`` contains the name of layers (not int values, but str in general) that corresponds to ``[PH][layers].nrrd``, ordered by depth, from top to bottom. The next entries are dictionaries where keys are the layers in ``layers`` entry.
The ``names`` entry contains human readable names that can be used for plotting, it is optional, mostly used for legend of collage plots.
The entry ``region_queries`` contains regexes for querying the atlas ``hierarchy.json`` to find ids or layers present in ``brain_region.nrrd``.
Finally, the entry ``thicknesses`` contains expected thicknesses of synthesis in vacuum which will be used to apply the rescaling algorithm. If the ``thicknesses`` entry is absent, no scaling rule ``extent_to_target`` will be applied, even if the rule is present.

For more subtle insitu synthesis, see the next two sections which describe two algorithms based on accept-reject mechanisms during growth.

Insitu synthesis with directions
--------------------------------

Under a region block (such as ``O0`` above) of ``region_structure.yaml``, one can add a ``directions`` block to control the growing directions  of sections during synthesis via atlas orientation field.

.. code-block:: yaml

  directions:
    - mtypes:
      - L1_HAC
      - L1_SAC
      neurite_types:
        - axon
      processes:
        - major
      params:
        direction: [0, 1, 0]
        power : 2.0
        mode: perpendicular
        layers: [1, 2]

This block contains a list of rules, with the following entries.  ``mtypes`` is the list of mtypes to apply this rule, ``neurite_types`` is the list of neurite_types to apply this rule. ``processes`` is optional and is the list of type of sections in NeuroTS (``major`` or ``secondary``) to differentiate between trunk (``major``) and obliques or collaterals (``secondary``).

The entry ``params`` is a dictionary to parametrize the rule. First, we specify the ``direction`` with a 3-vector, where ``[0, 1, 0]`` is the pia direction and ``[0, -1, 0]`` is opposite to the pia. For non-cortical regions, pia generalises to ``y`` coordinate of the orientation vector in ``orientation.nrrd``.
Then, the ``mode`` selects between ``parallel`` (default if omitted) to follow the direction, and ``perpendicular`` to follow the perpendicular directions, hence a plane.
The optional ``power`` value is to set how strong the direction constraint is. The underlying algorithm converts the angle between the next point to grow and the direction into a probability function. If ``power=1`` (default) the relation is linear, otherwise it is a power of it (see ``get_directions`` in ``region-grower/region_grower/context.py``).
Finally, this rule can be applied into only specific layers, via the list in ``layers`` entry (default to all layers).

Insitu synthesis with boundaries
--------------------------------

Under a region block (such as ``O0`` above) of ``region_structure.yaml``, one can add a ``boundaries`` block to control the growing directions of trunks and sections during synthesis via atlas based meshes.

.. code-block:: yaml

  boundaries:
    - mtypes:
      - L2_TPC:A
      neurite_types:
        - apical_dendrite
        - basal_dendrite
        - axon
      params_section:
        d_min: 5
        d_max: 50
      params_trunk:
        d_min: 5.0
        d_max: 1000
        power: 3.0
      mode: repulsive
      path: pia_mesh.obj

This block contains a list of rules for boundary constraints, similar to the direction for ``mtypes`` and ``neurite_types`` entries.
Each rule must have a ``path`` entry to a mesh (readabe by https://github.com/mikedh/trimesh) in either voxel id or coordinates. To select between the two  ``mesh_type`` entry can be used with value ``voxel`` (default) for voxel ids or ``spatial`` for coordinates.
If the path is relative, it will be interpreted as relative to the location of ``region_structure.yaml`` file.
If the ``path`` is a folder, then it must contain mesh files which will be used for this rule. The way the mesh are selected to act as boundary depends on the rule parametrized by ``multimesh_mode``, which can be set to ``closest`` (default) for selecting the closest mech to the soma as the unique mesh, or ``inside`` to select the mesh surrounding the soma (used for barrel cortext for example).

There are two main modes for these rules, parametrized by ``modes``. Either ``repulsive`` (default) where the mesh will act as a repulsive wall/boundary, or ``attractive`` where the mesh will attract the growing sections (more experimental, used for glomeruli spherical meshes for example).

This rule can then be applied to either the section growing with ``params_section`` or trunk placements with ``params_trunk`` (only if the non-default trunk angle method is selected, see above).
In both cases, the algorithm uses ray tracing to compute the distance to the mesh in the direction of the growth, and convert it to a probability function. The probability will be ``0`` below a distance of ``d_min``, and ``1`` above the distance of ``d_max``. This distance is from the previous point (soma for trunk), and the direction is to the next point (first neurite point for trunk). The ``power`` argument is as above, to have a nonlinear function of distance.
If ``d_min`` is close negative, there will be a probability of going though the mesh, hence making it leaky.
The mesh are considered as non-oriented, hence there is no notion of side, so is a branch passes through, it will have no effect, unless the growing turns back and hit the mesh again from the other side.

Meshes can be generated with trimesh package directly (or any other means), or via the atlas based helper here: https://bbpgitlab.epfl.ch/neuromath/neurocollage/-/blob/main/neurocollage/mesh_helper.py.
