"""Luigi tasks for morphology synthesis."""
import json
import logging
import os
import re
import warnings
from functools import partial
from pathlib import Path

import luigi
import morphio
import numpy as np
import pandas as pd
import yaml
from voxcell import VoxelData

from atlas_analysis.planes.planes import (_create_planes,
                                          create_centerline_planes,
                                          save_planes_centerline)
from diameter_synthesis.build_models import build as build_diameter_models
from region_grower.utils import NumpyEncoder
from tns import extract_input

from .utils_tasks import (
    diametrizerconfigs,
    circuitconfigs,
    synthesisconfigs,
    pathconfigs,
)
from .utils_tasks import get_morphs_df, ensure_dir
from .circuit_slicing import generic_slicer, slice_circuit
from .synthesis import (
    get_neurite_types,
    apply_substitutions,
    build_distributions,
    create_axon_morphologies_tsv,
    run_placement_algorithm,
    get_mean_neurite_lengths,
    grow_vacuum_morphologies,
    plot_vacuum_morphologies,
    rescale_morphologies,
    add_scaling_rules_to_parameters,
)

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
warnings.filterwarnings("ignore")
morphio.set_maximum_warnings(0)


class ApplySubstitutionRules(luigi.Task):
    """Apply substitution rules to the morphology dataframe.

    Args:
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies
    """

    substitution_rules_path = luigi.Parameter(default="substitution_rules.yaml")

    def run(self):
        """"""
        with open(self.substitution_rules_path, "rb") as sub_file:
            substitution_rules = yaml.full_load(sub_file)

        substituted_morphs_df = apply_substitutions(
            pd.read_csv(pathconfigs().morphs_df_path), substitution_rules
        )
        ensure_dir(self.output().path)
        substituted_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(pathconfigs().substituted_morphs_df_path)


class BuildSynthesisParameters(luigi.Task):
    """Build the tmd_parameter.json for synthesis.

    Args:
        to_use_flag (str): column name in morphology dataframe to select subset of cells
    """

    tmd_parameters_path = luigi.Parameter(default="tmd_parameters.json")
    to_use_flag = luigi.Parameter(default="all")

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""
        mtypes = list(
            set(get_morphs_df(self.input().path, to_use_flag=self.to_use_flag).mtype)
        )
        neurite_types = get_neurite_types(synthesisconfigs().pc_in_types_path, mtypes)

        if synthesisconfigs().custom_tmd_parameters_path is not None:
            L.info("Using custom tmd parameters")
            with open(synthesisconfigs().custom_tmd_parameters_path, "r") as f:
                custom_tmd_parameters = json.load(f)

        tmd_parameters = {}
        for mtype in mtypes:
            if synthesisconfigs().custom_tmd_parameters_path is None:
                tmd_parameters[mtype] = extract_input.parameters(
                    neurite_types=neurite_types[mtype],
                    diameter_parameters=diametrizerconfigs().config_diametrizer,
                )
            else:
                try:
                    tmd_parameters[mtype] = custom_tmd_parameters[mtype]
                except KeyError:
                    L.error("%s is not in the given tmd_parameter.json", mtype)
                    tmd_parameters[mtype] = {}
                tmd_parameters[mtype][
                    "diameter_params"
                ] = diametrizerconfigs().config_diametrizer
                tmd_parameters[mtype]["diameter_params"]["method"] = "external"

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4)

    def output(self):
        """"""
        return luigi.LocalTarget(self.tmd_parameters_path)


class BuildSynthesisDistributions(luigi.Task):
    """Build the tmd_distribution.json for synthesis.

    Args:
        to_use_flag (str): column name in morphology dataframe to select subset of cells
        morphology_path (str): column name in morphology dataframe to access morphology paths
        h5_path (str): base path to h5 morphologies for TNS distribution extraction
    """

    to_use_flag = luigi.Parameter(default="all")
    morphology_path = luigi.Parameter(default="morphology_path")
    h5_path = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""

        morphs_df = get_morphs_df(
            self.input().path,
            to_use_flag=self.to_use_flag,
            morphology_path=self.morphology_path,
            h5_path=self.h5_path,
        )
        mtypes = list(set(morphs_df.mtype))
        neurite_types = get_neurite_types(synthesisconfigs().pc_in_types_path, mtypes)

        diameter_model_function = partial(
            build_diameter_models, config=diametrizerconfigs().config_model
        )

        tmd_distributions = build_distributions(
            mtypes,
            morphs_df,
            neurite_types,
            diameter_model_function,
            self.morphology_path,
            synthesisconfigs().cortical_thickness,
        )

        with self.output().open("w") as f:
            json.dump(tmd_distributions, f, cls=NumpyEncoder, indent=4)

    def output(self):
        """"""
        return luigi.LocalTarget(synthesisconfigs().tmd_distributions_path)


class BuildSynthesisModels(luigi.WrapperTask):
    """Only build both json files for synthesis."""

    def requires(self):
        """"""
        return [BuildSynthesisParameters(), BuildSynthesisDistributions()]


class CreateAtlasPlanes(luigi.Task):
    """Crerate plane cuts of an atlas."""

    plane_type = luigi.Parameter(default="centerline")
    plane_count = luigi.IntParameter(default=10)
    inter_plane_count = luigi.IntParameter(default=200)

    use_half = luigi.Parameter(default=True)
    half_axis = luigi.IntParameter(default=0)
    half_side = luigi.IntParameter(default=0)

    tmp_nrrd_path = luigi.Parameter(default="layer_annotation.nrrd")

    centerline_first_bound = luigi.ListParameter(default=[126, 181, 220])
    centerline_last_bound = luigi.ListParameter(default=[407, 110, 66])

    centerline_axis = luigi.IntParameter(default=0)
    centerline_start = luigi.FloatParameter(3000)
    centerline_end = luigi.FloatParameter(10000)

    atlas_planes_path = luigi.Parameter(default="atlas_planes")

    def run(self):
        """"""
        layer_annotation_path = Path(circuitconfigs().atlas_path) / "layers.nrrd"
        layer_annotation = VoxelData.load_nrrd(layer_annotation_path)
        if self.use_half:
            layer_annotation.raw = halve_atlas(
                layer_annotation.raw, axis=self.half_axis, side=self.half_side
            )
        ensure_dir(self.tmp_nrrd_path)
        layer_annotation.save_nrrd(self.tmp_nrrd_path)

        if self.plane_type == "centerline":
            bounds = [self.centerline_first_bound, self.centerline_last_bound]
            create_centerline_planes(
                self.tmp_nrrd_path,
                self.output().path,
                bounds,
                plane_count=self.inter_plane_count,
            )

        if self.plane_type == "aligned":
            centerline = np.zeros([100, 3])
            centerline[:, self.centerline_axis] = np.linspace(
                self.centerline_start, self.centerline_end, 100
            )
            planes = _create_planes(centerline, plane_count=self.inter_plane_count)
            save_planes_centerline(self.output().path, planes, centerline)

        all_planes = np.load(self.output().path)
        selected_planes = []
        di = int(self.inter_plane_count / self.plane_count)
        for i in range(self.plane_count):
            selected_planes.append(all_planes["planes"][di * i])
            selected_planes.append(all_planes["planes"][di * i + 1])
        np.savez(
            self.output().path,
            planes=np.array(selected_planes),
            centerline=all_planes["centerline"],
            plane_format=all_planes["plane_format"],
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.atlas_planes_path + ".npz")


class SliceCircuit(luigi.Task):
    """Create a smaller circuit .mvd3 file for subsampling.

    Args:
        sliced_circuit_path (str): path to save sliced circuit somata mvd3
        mtypes (list): list of mtypes to consider
        n_cells (int): number of cells per mtype to consider
    """

    sliced_circuit_path = luigi.Parameter(default="sliced_circuit_somata.mvd3")
    mtypes = luigi.ListParameter(default=["all"])
    n_cells = luigi.IntParameter(default=10)
    hemisphere = luigi.Parameter(default="left")

    def requires(self):
        """"""
        return CreateAtlasPlanes()

    def run(self):
        """"""
        if "all" in self.mtypes:  # pylint: disable=unsupported-membership-test
            self.mtypes = None

        slicer = partial(
            generic_slicer,
            n_cells=self.n_cells,
            mtypes=self.mtypes,
            planes=np.load(self.input().path)["planes"],
            hemisphere=self.hemisphere,
        )

        ensure_dir(self.output().path)
        cells = slice_circuit(
            circuitconfigs().circuit_somata_path, self.output().path, slicer
        )

        if len(cells.index) == 0:
            raise Exception("No cells will be synthtesized, better stop here.")

    def output(self):
        """"""
        return luigi.LocalTarget(self.sliced_circuit_path)


class Synthesize(luigi.Task):
    """Run placement-algorithm to synthesize morphologies.

    Args:
        out_circuit_path (str): path to circuit mvd3 with morphology data
        ext (str): extension for morphology files
        axon_morphs_path (str): path to .tsv file for axon grafting
        axon_morphs_base_dir (str): base dir for morphology used for axon (.h5 files)
        apical_points_path (str): path to .yaml file for recording apical points
    """

    out_circuit_path = luigi.Parameter(default="sliced_circuit_morphologies.mvd3")
    ext = luigi.Parameter(default=".asc")
    axon_morphs_path = luigi.Parameter(default="axon_morphs_path.tsv")
    axon_morphs_base_dir = luigi.Parameter(default="None")
    apical_points_path = luigi.Parameter(default="apical_points.yaml")
    nb_jobs = luigi.IntParameter(default=-1)

    def requires(self):
        """"""

        return {
            "substituted_cells": ApplySubstitutionRules(),
            "circuit": SliceCircuit(),
            "tmd_parameters": AddScalingRulesToParameters(),
            "tmd_distributions": BuildSynthesisDistributions(),
        }

    def run(self):
        """"""

        ensure_dir(self.axon_morphs_path)
        ensure_dir(self.output().path)
        ensure_dir(self.apical_points_path)
        ensure_dir(pathconfigs().synth_output_path)

        axon_morphs_base_dir = create_axon_morphologies_tsv(
            self.input()["circuit"].path,
            self.input()["substituted_cells"].path,
            axon_morphs_path=self.axon_morphs_path,
        )
        if self.axon_morphs_base_dir != "None":
            axon_morphs_base_dir = self.axon_morphs_base_dir

        args = {
            "cells_path": self.input()["circuit"].path,
            "tmd_parameters": self.input()["tmd_parameters"].path,
            "tmd_distributions": self.input()["tmd_distributions"].path,
            "atlas": circuitconfigs().atlas_path,
            "out_mvd3": self.output().path,
            "out_apical": self.apical_points_path,
            "out_morph_ext": str(self.ext)[1:],
            "out_morph_dir": pathconfigs().synth_output_path,
            "overwrite": True,
            "no_mpi": True,
            "morph-axon": self.axon_morphs_path,
            "base-morph-dir": axon_morphs_base_dir,
            "max_drop_ratio": str(0.1),
            "seed": str(0),
        }

        run_placement_algorithm(args, nb_jobs=self.nb_jobs)

    def output(self):
        """"""
        return luigi.LocalTarget(self.out_circuit_path)


class GetMeanNeuriteLengths(luigi.Task):
    """Get the mean neurite lenghts of a neuron population, per mtype and neurite type."""

    neurite_types = luigi.ListParameter(default=["apical"])
    mtypes = luigi.ListParameter(default=["all"])
    morphology_path = luigi.Parameter(default="morphology_path")
    mean_lengths_path = luigi.Parameter(default="mean_neurite_lengths.yaml")

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""

        morphs_df = pd.read_csv(self.input().path)
        mean_lengths = {
            neurite_type: get_mean_neurite_lengths(
                morphs_df,
                neurite_type=neurite_type,
                mtypes=self.mtypes,
                morphology_path=self.morphology_path,
            )
            for neurite_type in self.neurite_types
        }

        with self.output().open("w") as f:
            yaml.dump(mean_lengths, f)

    def output(self):
        """"""
        return luigi.LocalTarget(self.mean_lengths_path)


class GetSynthetisedNeuriteLengths(luigi.Task):
    """Get the mean neurite lenghts of a neuron population, per mtype and neurite type."""

    neurite_types = luigi.ListParameter(default=["apical"])
    mtypes = luigi.ListParameter(default=["all"])
    morphology_path = luigi.Parameter(default="morphology_path")
    mean_lengths_path = luigi.Parameter(default="mean_neurite_lengths.yaml")
    percentile = luigi.Parameter(default=50)

    def requires(self):
        """"""
        return VacuumSynthesize()

    def run(self):
        """"""

        synth_morphs_df = pd.read_csv(self.input().path)
        mean_lengths = {
            neurite_type: get_mean_neurite_lengths(
                synth_morphs_df,
                neurite_type=neurite_type,
                mtypes=self.mtypes,
                morphology_path=self.morphology_path,
                percentile=self.percentile,
            )
            for neurite_type in self.neurite_types
        }

        L.info("Lengths: {}".format(mean_lengths))

        with self.output().open("w") as f:
            yaml.dump(mean_lengths, f)

    def output(self):
        """"""
        return luigi.LocalTarget(self.mean_lengths_path)


class AddScalingRulesToParameters(luigi.Task):
    """Add scaling rules to tmd_parameter.json."""

    scaling_rules_path = luigi.Parameter(default="scaling_rules.yaml")
    hard_limits_path = luigi.Parameter(default="hard_limits.yaml")
    tmd_parameters_with_scaling_path = luigi.Parameter(
        default="tmd_parameters_with_scaling.json"
    )

    def requires(self):
        """"""
        return {
            "tmd_parameters": BuildSynthesisParameters(),
            "mean_lengths": GetMeanNeuriteLengths(),
        }

    def run(self):
        """"""
        tmd_parameters = json.load(self.input()["tmd_parameters"].open("r"))
        mean_lengths = yaml.full_load(self.input()["mean_lengths"].open("r"))
        scaling_rules = yaml.full_load(open(self.scaling_rules_path, "r"))

        def _get_target_layer(target_layer_str):
            if re.match("^L", target_layer_str):
                position = 0.5
                if len(target_layer_str) == 3:
                    position = 1
                return int(target_layer_str[1]), position
            raise Exception("Scaling rule not understood: " + str(target_layer_str))

        for neurite_type in mean_lengths:
            for mtype, mean_length in mean_lengths[neurite_type].items():
                if (
                    scaling_rules[mtype] is not None
                    and neurite_type in scaling_rules[mtype]
                ):
                    tmd_parameters[mtype][neurite_type][
                        "expected_max_length"
                    ] = mean_length
                    layer, position = _get_target_layer(
                        scaling_rules[mtype][neurite_type]
                    )
                    tmd_parameters[mtype][neurite_type]["target_layer"] = layer
                    tmd_parameters[mtype][neurite_type][
                        "target_layer_position"
                    ] = position

        hard_limits = yaml.full_load(open(self.hard_limits_path, "r"))

        add_scaling_rules_to_parameters(
            tmd_parameters, mean_lengths, scaling_rules, hard_limits
        )

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4)

    def output(self):
        """"""
        return luigi.LocalTarget(self.tmd_parameters_with_scaling_path)


class VacuumSynthesize(luigi.Task):
    """Grow cells in vacuum, for annotation tasks."""

    mtypes = luigi.ListParameter(default=["all"])
    vacuum_synth_morphology_path = luigi.Parameter(default="vacuum_synth_morphologies")
    vacuum_synth_morphs_df_path = luigi.Parameter(default="vacuum_synth_morphs_df.csv")
    n_cells = luigi.IntParameter(default=10)

    def requires(self):
        """"""

        return {
            "tmd_parameters": BuildSynthesisParameters(),
            "tmd_distributions": BuildSynthesisDistributions(),
        }

    def run(self):
        """"""
        tmd_parameters = json.load(self.input()["tmd_parameters"].open())
        tmd_distributions = json.load(self.input()["tmd_distributions"].open())

        if self.mtypes[0] == "all":  # pylint: disable=unsubscriptable-object
            mtypes = list(tmd_parameters.keys())
        else:
            mtypes = self.mtypes

        Path(self.vacuum_synth_morphology_path).mkdir(parents=True, exist_ok=True)
        morphology_base_path = Path(self.vacuum_synth_morphology_path).absolute()
        vacuum_synth_morphs_df = grow_vacuum_morphologies(
            mtypes,
            self.n_cells,
            tmd_parameters,
            tmd_distributions,
            morphology_base_path,
        )
        vacuum_synth_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(self.vacuum_synth_morphs_df_path)


class PlotVacuumMorphologies(luigi.Task):
    """Plot morphologies to obtain annotations."""

    pdf_filename = luigi.Parameter(default="vacuum_morphologies.pdf")
    morphology_path = luigi.Parameter(default="vacuum_morphology_path")

    def requires(self):
        """"""
        return {"vacuum": VacuumSynthesize(), "mean_lengths": GetMeanNeuriteLengths()}

    def run(self):
        """"""
        vacuum_synth_morphs_df = pd.read_csv(self.input()["vacuum"].path)
        mean_lengths = yaml.full_load(self.input()["mean_lengths"].open())
        ensure_dir(self.output().path)
        plot_vacuum_morphologies(
            vacuum_synth_morphs_df,
            self.output().path,
            self.morphology_path,
            mean_lengths,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.pdf_filename)


class RescaleMorphologies(luigi.Task):
    """Rescale morphologies for synthesis input."""

    morphology_path = luigi.Parameter(default="morphology_path")
    rescaled_morphology_path = luigi.Parameter(default="rescaled_morphology_path")
    rescaled_morphology_base_path = luigi.Parameter(default="rescaled_morphologies")
    scaling_rules_path = luigi.Parameter(default="scaling_rules.yaml")
    rescaled_morphs_df_path = luigi.Parameter(default="rescaled_morphs_df.csv")
    scaling_mode = luigi.Parameter(default="y")

    def requires(self):
        """"""
        return ApplySubstitutionRules()

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input().path)
        scaling_rules = yaml.full_load(open(self.scaling_rules_path, "r"))
        rescaled_morphs_df = rescale_morphologies(
            morphs_df,
            scaling_rules,
            json.loads(synthesisconfigs().cortical_thickness),
            self.morphology_path,
            self.rescaled_morphology_base_path,
            self.rescaled_morphology_path,
            scaling_mode=self.scaling_mode,
        )

        rescaled_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(self.rescaled_morphs_df_path)
