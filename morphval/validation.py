'''Statistical validation tools'''
import numpy as np
import copy
from decimal import Decimal

from neurom import stats
from neurom import get
from neurom.core.types import NeuriteType

DICTDATA = dict.fromkeys(["name", "data_type", "data", "labels"])

DICTALLDATA = dict.fromkeys(
    ["datasets", "description", "charts", "version", "result", "type"])

DESCR = ("Morphology validation against reference morphologies. "
         "Comparison of the %s of the two populations. "
         "The sample sizes of the set to be validated and the reference set "
         "are %d and %d respectively. The %s statistical test has "
         "been used for measuring the similarity between the two datasets. "
         "The corresponding distance between the distributions "
         "is: %f (p-value = %f). The test result is %s "
         "for a comparison of the pvalue with the accepted threshold %f.")


def extract_hist(data, bins=20):
    '''Extracts a histogram distribution
       from data. Choose bins to select the bins,
       according to numpy.histogram guidelines.
    '''

    bin_data, edges = np.histogram(data, bins, normed=True)

    edges_centers = [
        float(Decimal("%.2f" % e)) for e in list((edges[1:] + edges[:-1]) / 2)
    ]

    return list(bin_data), list(edges_centers)


def load_stat_test(test_name):
    obj = stats
    for attr in test_name.split('.'):
        obj = getattr(obj, attr)
    return obj


def stat_test(validation_data,
              reference_data,
              test,
              fargs=0.1,
              val_crit='pvalue'):
    '''Runs the selected statistical test
       and returns the results(distance, pvalue)
       along with a PASS - FAIL statement
       according to the selected threshold.
    '''
    results = stats.compare_two(validation_data, reference_data, test)

    res = bool(getattr(results, val_crit) > fargs)

    status = 'PASS' if res else 'FAIL'

    return results, status


def write_hist(data, feature, name, bins=20):
    '''Writes the histogram in the format
       expected by the validation report.
    '''
    bin_data, edges = extract_hist(data, bins=bins)

    pop_data = copy.deepcopy(DICTDATA)

    pop_data["name"] = name
    pop_data["data_type"] = "Histogram1D"
    pop_data["data"] = {"bin_center": edges, "entries": bin_data}
    pop_data["labels"] = {
        "bin_center": feature.capitalize().replace('_', ' '),
        "entries": "Fraction"
    }

    return pop_data


def unpack_config_data(config, name, component, feature):
    '''Returns values needed for statistical tests
       from config file.
    '''
    base_config = config[name][component][feature]
    return (base_config['bins'], load_stat_test(base_config['stat_test']),
            base_config['threshold'], base_config['criterion'],
            )


def write_all(validation_data, reference_data, component, feature, name,
              config):
    '''Writes the histogram in the format
       expected by the validation report.
    '''
    all_data = copy.deepcopy(DICTALLDATA)

    bins, test, thresh, val_crit = unpack_config_data(
        config=config, name=name, component=component, feature=feature)

    valid = write_hist(
        data=validation_data,
        feature=feature,
        name=name + '-test',
        bins=bins)

    refer = write_hist(
        data=reference_data,
        feature=feature,
        name=name + '-reference',
        bins=bins)

    results, status = stat_test(
        validation_data,
        reference_data,
        test=test,
        fargs=thresh,
        val_crit=val_crit)

    all_data["datasets"] = {"validation": valid, "reference": refer}
    all_data["description"] = DESCR % (
        feature, len(validation_data), len(reference_data), test.name,
        results.dist, results.pvalue, status, thresh)
    all_data["charts"] = {
        "Data - Model Comparison": ["validation", "reference"]
    }
    all_data["version"] = "0.1"
    all_data["result"] = {"status": status, "probability": results.pvalue}
    all_data["type"] = "validation"

    return all_data


def extract_feature(test_population, ref_population, component, feature):
    '''Extracts the distributions of the selected
       feature from the test and reference populations.
    '''
    neurite_type = getattr(NeuriteType, component)

    ref_data = get(feature, ref_population, neurite_type=neurite_type)
    test_data = get(feature, test_population, neurite_type=neurite_type)

    return test_data, ref_data
