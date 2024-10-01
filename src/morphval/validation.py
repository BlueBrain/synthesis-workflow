"""Statistical validation tools."""

import copy
from collections import namedtuple
from decimal import Decimal

import numpy as np
from neurom import get
from neurom import stats
from neurom.core.types import NeuriteType

DICTDATA = dict.fromkeys(["name", "data_type", "data", "labels"])

DICTALLDATA = dict.fromkeys(["datasets", "description", "charts", "version", "result", "type"])

DESCR = (
    "Morphology validation against reference morphologies. "
    "Comparison of the %s of the two populations. "
    "The sample sizes of the set to be validated and the reference set "
    "are %d and %d respectively. The %s statistical test has "
    "been used for measuring the similarity between the two datasets. "
    "The corresponding distance between the distributions "
    "is: %f (p-value = %f). The test result is %s "
    "for a comparison of the pvalue with the accepted threshold %f."
)

# Redefine a NamedTuple to store NeuroM stat results so it can be pickled
Stats = namedtuple("Stats", ["dist", "pvalue"])


def extract_hist(data, bins=20):
    """Extract a histogram distribution from data.

    Args:
        data: the data from which the histogram is computed
        bins: select the bins, according to numpy.histogram guidelines.
    """
    bin_data, edges = np.histogram(data, bins, density=True)

    edges_centers = [float(Decimal(f"{e:.2f}")) for e in list((edges[1:] + edges[:-1]) / 2)]

    return list(bin_data), list(edges_centers)


def load_stat_test(test_name):
    """Load stat test object from test name."""
    obj = stats
    for attr in test_name.split("."):
        obj = getattr(obj, attr)
    return obj


def stat_test(validation_data, reference_data, test, fargs=0.1, val_crit="pvalue"):
    """Run the selected statistical test.

    Returns:
        the results(distance, pvalue) along with a PASS - FAIL statement
        according to the selected threshold.
    """
    res_tmp = stats.compare_two(validation_data, reference_data, test)
    results = Stats(*res_tmp)

    res = bool(getattr(results, val_crit) > fargs)

    status = "PASS" if res else "FAIL"

    return results, status


def write_hist(data, feature, name, bins=20):
    """Write the histogram in the format expected by the validation report."""
    bin_data, edges = extract_hist(data, bins=bins)

    pop_data = copy.deepcopy(DICTDATA)

    pop_data["name"] = name
    pop_data["data_type"] = "Histogram1D"
    pop_data["data"] = {"bin_center": edges, "entries": bin_data}
    pop_data["labels"] = {
        "bin_center": feature.capitalize().replace("_", " "),
        "entries": "Fraction",
    }

    return pop_data


def unpack_config_data(config, name, component, feature):
    """Return values needed for statistical tests from config file."""
    base_config = config[name][component][feature]
    return (
        base_config["bins"],
        load_stat_test(base_config["stat_test"]),
        base_config["threshold"],
        base_config["criterion"],
    )


def write_all(validation_data, reference_data, component, feature, name, config):
    """Write the histogram in the format expected by the validation report."""
    all_data = copy.deepcopy(DICTALLDATA)

    bins, test, thresh, val_crit = unpack_config_data(
        config=config, name=name, component=component, feature=feature
    )

    valid = write_hist(data=validation_data, feature=feature, name=name + "-test", bins=bins)

    refer = write_hist(data=reference_data, feature=feature, name=name + "-reference", bins=bins)

    results, status = stat_test(
        validation_data, reference_data, test=test, fargs=thresh, val_crit=val_crit
    )

    all_data["datasets"] = {"validation": valid, "reference": refer}
    all_data["description"] = DESCR % (
        feature,
        len(validation_data),
        len(reference_data),
        test.name,
        results.dist,
        results.pvalue,
        status,
        thresh,
    )
    all_data["charts"] = {"Data - Model Comparison": ["validation", "reference"]}
    all_data["version"] = "0.1"
    all_data["result"] = {"status": status, "probability": results.pvalue}
    all_data["type"] = "validation"

    return all_data


def extract_feature(test_population, ref_population, component, feature):
    """Extract the distributions of the selected feature.

    The distributions are extracted from the test and reference populations.
    """
    neurite_type = getattr(NeuriteType, component)

    ref_data = get(feature, ref_population, neurite_type=neurite_type)
    test_data = get(feature, test_population, neurite_type=neurite_type)

    return test_data, ref_data
