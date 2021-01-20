import json
import yaml
from pathlib import Path
from hashlib import md5

import numpy as np
import pandas as pd
from diff_pdf_visually import pdfdiff
from voxcell import CellCollection
from voxcell import VoxelData


def nested_round(obj, precision=6):
    """Round all floats (recursively) in a nested dictionary"""
    if isinstance(obj, np.ndarray):
        return [nested_round(i, precision) for i in obj.tolist()]
    elif isinstance(obj, np.floating):
        if precision is not None:
            return np.round(obj, precision)
        else:
            return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, float):
        if precision is not None:
            return round(obj, precision)
        else:
            return obj
    elif isinstance(obj, dict):
        return dict((k, nested_round(v, precision)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return [nested_round(i, precision) for i in obj]
    return obj


def compare_dicts(ref, test, precision=None):
    if precision is not None:
        ref = nested_round(ref, precision)
        test = nested_round(test, precision)
    return json.dumps(ref, sort_keys=True) == json.dumps(test, sort_keys=True)


def compare_json_files(ref_path, test_path, precision=6):
    """Compare data from two *.json files"""
    with open(ref_path) as f:
        ref = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    res = compare_dicts(ref, test, precision)
    if res is True:
        return True
    else:
        return (
            f"The files {ref_path} and {test_path} are different "
            f"(with precision={precision}):\n{str(ref)}\n!=\n{str(test)}"
        )


def compare_yaml_files(ref_path, test_path, precision=6):
    """Compare data from two *.yaml files"""
    with open(ref_path) as f:
        ref = yaml.load(f)
    with open(test_path) as f:
        test = yaml.load(f)
    res = compare_dicts(ref, test, precision)
    if res is True:
        return True
    else:
        return f"The files {ref_path} and {test_path} are different:\n{str(ref)}\n !=\n{str(test)}"


def compare_dataframes(ref, test, rtol=None, atol=None, ignore_columns=None):
    """Compare two Pandas.DataFrames"""
    if ignore_columns is not None:
        ref.drop(columns=ignore_columns, inplace=True, errors="ignore")
        test.drop(columns=ignore_columns, inplace=True, errors="ignore")

    kwargs = {}
    if rtol is not None or atol is not None:
        kwargs["check_exact"] = False
        if rtol is not None:
            kwargs["rtol"] = rtol
        if atol is not None:
            kwargs["atol"] = atol
    else:
        kwargs["check_exact"] = True

    try:
        pd.testing.assert_frame_equal(ref, test, **kwargs)
        return True
    except AssertionError as e:
        return e.args[0]


def compare_csv_files(
    ref_path,
    test_path,
    ref_sep=",",
    test_sep=",",
    rtol=None,
    atol=None,
    ignore_columns=None,
):
    """Compare data from two *.csv / *.tsv / *.dat files"""
    ref = pd.read_csv(ref_path, sep=ref_sep)
    test = pd.read_csv(test_path, sep=test_sep)

    res = compare_dataframes(ref, test, rtol, atol, ignore_columns)
    if res is True:
        return True
    else:
        return f"The files {ref_path} and {test_path} are different:\n{res}"


def compare_nrrd_files(ref_path, test_path, precision=None):
    """Compare data from two *.nrrd files

    Note: *.nrrd files can contain their creation date, so their hashes are depends on
    this creation date, even if the data is the same.
    """
    ref = VoxelData.load_nrrd(ref_path).raw
    test = VoxelData.load_nrrd(test_path).raw
    try:
        if precision is not None:
            np.testing.assert_array_almost_equal(ref, test, decimal=precision)
        else:
            np.testing.assert_array_equal(ref, test)
        return True
    except AssertionError as e:
        return f"The files {ref_path} and {test_path} are different:\n{e.args[0]}"


def compare_mvd3_files(ref_path, test_path, rtol=None, atol=None, ignore_columns=None):
    """Compare data from two *.mvd3 files

    Note: *.mvd3 files can contain their creation date, so their hashes are depends on
    this creation date, even if the data is the same.
    """
    ref = CellCollection.load_mvd3(ref_path).as_dataframe()
    test = CellCollection.load_mvd3(test_path).as_dataframe()
    res = compare_dataframes(ref, test, rtol, atol, ignore_columns)
    if res is True:
        return True
    else:
        return f"The files {ref_path} and {test_path} are different:\n{res}"


def compare_pdf_files(ref_path, test_path, *args, **kwargs):
    """Compare two *.pdf files"""
    res = pdfdiff(ref_path, test_path, *args, **kwargs)
    if res is True:
        return True
    else:
        return f"The files {ref_path} and {test_path} are different"


COMPARATORS = {
    ".csv": compare_csv_files,
    ".json": compare_json_files,
    ".mvd3": compare_mvd3_files,
    ".nrrd": compare_nrrd_files,
    ".pdf": compare_pdf_files,
    ".tsv": compare_csv_files,
    ".yaml": compare_yaml_files,
}


def compare_tree(ref_path, comp_path, comparators=None, specific_args=None, verbose=False):
    """Compare all files from 2 different directory trees"""
    if comparators is None:
        comparators = COMPARATORS

    ref_path = Path(ref_path)
    comp_path = Path(comp_path)

    if specific_args is None:
        specific_args = {}

    # Loop over all files and call the correct comparator
    different_files = []
    for ref_file in ref_path.glob("**/*"):
        if ref_file.is_dir():
            continue
        suffix = ref_file.suffix
        relative_path = ref_file.relative_to(ref_path)
        comp_file = comp_path / relative_path
        if verbose:
            print("Compare:\n\t%s\n\t%s" % (ref_file, comp_file))
        if comp_file.exists():
            if suffix in COMPARATORS:
                args = specific_args.get(relative_path.as_posix(), {}).get("args", [])
                kwargs = specific_args.get(relative_path.as_posix(), {}).get("kwargs", {})
                try:
                    res = COMPARATORS[suffix](ref_file, comp_file, *args, **kwargs)
                    if res is not True:
                        different_files.append(res)
                except Exception as e:
                    different_files.append(
                        f"The file '{relative_path}' is different from the reference for the "
                        "following reason reason: %s " % "\n".join(e.args)
                    )
            else:
                # If no comparator is given for this suffix, test with MD5 hashes
                with ref_file.open("rb") as f:
                    ref_md5 = md5(f.read()).hexdigest()
                with comp_file.open("rb") as f:
                    comp_md5 = md5(f.read()).hexdigest()
                if ref_md5 != comp_md5:
                    msg = (
                        f"The MD5 hashes are different for the file '{relative_path}': "
                        f"{ref_md5} != {comp_md5}"
                    )
                    different_files.append(msg)
        else:
            msg = f"The file '{relative_path}' does not exist in '{comp_path}'"
            different_files.append(msg)

    # Test that all files are equal and raise the formatted message if there are differences
    assert len(different_files) == 0, "\n".join(different_files)
