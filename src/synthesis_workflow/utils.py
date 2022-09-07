"""Utils functions."""
import json
import logging

import dictdiffer
import pandas as pd
from jsonpath_ng import parse

# pylint:disable=too-many-nested-blocks


class DisableLogger:
    """Context manager to disable logging."""

    def __init__(self, log_level=logging.CRITICAL, logger=None):
        self.log_level = log_level
        self.logger = logger
        if self.logger is None:
            self.logger = logging

    def __enter__(self):
        self.logger.disable(self.log_level)

    def __exit__(self, *args):
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
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

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
    root.addHandler(console)

    # Setup file logging handler
    if log_file is not None:
        if log_file_level is None:
            log_file_level = log_level
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_file_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)


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
        param[entry[0].split("[")[0]].append([None])
    elif entry[0] not in param:
        param[entry[0]] = None
    if len(entry) > 1:
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
