"""Utils functions."""

import json
import logging
from pathlib import Path

import dictdiffer
import lxml.etree
import numpy as np
import pandas as pd
from jsonpath_ng import parse
from pkg_resources import resource_filename
from tqdm import tqdm

# pylint:disable=too-many-nested-blocks

_TEMPLATES = Path(
    resource_filename(
        "synthesis_workflow",
        "_templates",
    )
)


class DisableLogger:
    """Context manager to disable logging."""

    def __init__(self, log_level=logging.CRITICAL, logger=None):
        self.log_level = log_level
        self.logger = logger
        if self.logger is None:
            self.logger = logging

    def __enter__(self):
        """Disabling the logger when entering the context manager."""
        self.logger.disable(self.log_level)

    def __exit__(self, *args):
        """Enabling the logger when exiting the context manager."""
        self.logger.disable(0)


def setup_logging(
    log_level=logging.DEBUG,
    log_file=None,
    log_file_level=None,
    log_format=None,
    date_format=None,
    logger=None,
):
    """Setup logging."""
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    # Setup logging formatter
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s -- %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Setup console logging handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)
    logger.addHandler(console)

    # Setup file logging handler
    if log_file is not None:
        if log_file_level is None:
            log_file_level = log_level
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def create_parameter_diff(param, param_spec):
    """Create a dataframe with a diff between two parameter dict."""
    custom_values = pd.DataFrame()
    i = 0
    for mtype in param:
        diff = dictdiffer.diff(
            param[mtype], param_spec[mtype], ignore=set(["diameter_params", "grow_types"])
        )
        for d in diff:
            if d[0] == "change":
                if isinstance(d[1], list):
                    entry = ""
                    for _d in d[1]:
                        if isinstance(_d, str):
                            if len(entry) > 0:
                                entry += "."
                            entry += _d
                        if isinstance(_d, int):
                            entry += f"[{_d}]"
                else:
                    entry = d[1]
                custom_values.loc[i, "mtype"] = mtype
                custom_values.loc[i, "entry"] = entry
                custom_values.loc[i, "value"] = d[2][1]
                i += 1
            if d[0] == "add":
                for _d in d[2]:
                    custom_values.loc[i, "mtype"] = mtype
                    if isinstance(_d[0], str):
                        custom_values.loc[i, "entry"] = ".".join([d[1], _d[0]])
                        custom_values.loc[i, "value"] = _d[1]
                    else:
                        custom_values.loc[i, "entry"] = f"{d[1]}[{_d[0]}]"
                        custom_values.loc[i, "value"] = json.dumps(_d[1])
                    i += 1
    return custom_values


def _create_entry(param, entry):
    """Create a dict entry if it does not exist."""
    if entry[0].endswith("]"):
        # add list for specific orientations
        if entry[0].split("[")[0] not in param:
            param[entry[0].split("[")[0]] = [None]
        else:
            param[entry[0].split("[")[0]].append([None])
    elif entry[0] in param and not isinstance(param[entry[0]], dict):
        # erase anything that is not a dict to be able to go deeper
        param[entry[0]] = {}
    elif entry[0] not in param:
        # add empty entries depending on depths
        if len(entry) > 1:
            param[entry[0]] = {}
        else:
            param[entry[0]] = None

    if len(entry) > 1:
        # go deeper if needed
        _create_entry(param[entry[0]], entry[1:])


def apply_parameter_diff(param, custom_values):
    """Apply a parameter diff from 'create_parameter_diff' to a parameter dict."""
    for mtype in param:
        df = custom_values[custom_values.mtype == mtype]
        for gid in df.index:
            entry = parse(f"$.{df.loc[gid, 'entry']}")
            if not entry.find(param[mtype]):
                _create_entry(param[mtype], df.loc[gid, "entry"].split("."))
            val = df.loc[gid, "value"]

            try:
                val = json.loads(val)
            except (json.decoder.JSONDecodeError, TypeError):
                pass

            if val in ("True", "False"):
                val = val == "True"

            entry.update(param[mtype], val)


def save_planes(planes, path):
    """Save planes to txt files."""
    np.savetxt(path / "centerline.dat", planes["centerline"])
    _planes_left = []
    _planes_center = []
    _planes_right = []
    for plane in planes["planes"]:
        _planes_left.append(np.hstack([plane["left"].point, plane["left"].normal]))
        _planes_center.append(np.hstack([plane["center"].point, plane["center"].normal]))
        _planes_right.append(np.hstack([plane["right"].point, plane["right"].normal]))
    np.savetxt(path / "planes_left.dat", _planes_left)
    np.savetxt(path / "planes_center.dat", _planes_center)
    np.savetxt(path / "planes_right.dat", _planes_right)


def create_circuit_config(nodes_file, morphology_path):
    """Create simple circuit config."""
    return {
        "networks": {
            "nodes": [
                {
                    "nodes_file": str(nodes_file),
                    "populations": {
                        "default": {
                            "type": "biophysical",
                            "biophysical_neuron_models_dir": ".",
                            "alternate_morphologies": {
                                "neurolucida-asc": str(morphology_path),
                            },
                        }
                    },
                }
            ],
            "edges": [],
        }
    }


def parse_annotations(filepath):
    """Parse XML with morphology annotations."""
    etree = lxml.etree.parse(filepath)
    result = {}
    for elem in etree.findall("placement"):
        attr = dict(elem.attrib)
        rule_id = attr.pop("rule")
        if rule_id in result:
            raise KeyError(f"Duplicate annotation for rule '{rule_id}'")
        result[rule_id] = attr
    return result


def parse_morphdb(filepath):
    """Parse (ext)neuronDB.dat file."""
    columns = ["morphology", "layer", "mtype"]
    first_row = pd.read_csv(filepath, sep=r"\s+", header=None, nrows=1)
    if first_row.shape[1] > 3:
        columns.append("etype")
    return pd.read_csv(
        filepath, sep=r"\s+", names=columns, usecols=columns, na_filter=False, dtype={"layer": str}
    )


def collect_annotations(annotation_dir, morphdb_path):
    """Collect annotations from given directory."""
    result = {}
    if morphdb_path is None:
        for filepath in tqdm(Path(annotation_dir).glob("*.xml")):
            result[Path(filepath).stem] = parse_annotations(filepath)
    else:
        morphdb = parse_morphdb(morphdb_path)
        for morph in tqdm(morphdb["morphology"].unique()):
            filepath = Path(annotation_dir) / (morph + ".xml")
            result[morph] = parse_annotations(filepath)
    return result
