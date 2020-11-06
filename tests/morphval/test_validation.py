import pytest
from copy import deepcopy

from morphval import validation
from neurom import stats
from neurom.core.types import NeuriteType


@pytest.fixture
def CONFIG():
    return {
        "mtype_test": {
            "neurite_test": {
                "feature_test": {
                    "stat_test": "StatTests.ks",
                    "threshold": 0.1,
                    "bins": 4,
                    "criterion": "dist",
                }
            }
        }
    }


@pytest.fixture
def test_single():
    return {
        "data": {"bin_center": [1.38, 2.12, 2.88, 3.62], "entries": [1 / 3] * 4},
        "data_type": "Histogram1D",
        "labels": {"bin_center": "Feature test", "entries": "Fraction"},
        "name": "mtype_test",
    }


@pytest.fixture
def test_dict(test_single):
    return {
        "charts": {"Data - Model Comparison": ["validation", "reference"]},
        "datasets": {
            "reference": deepcopy(test_single),
            "validation": deepcopy(test_single),
        },
        "description": (
            "Morphology validation against reference morphologies. Comparison of the "
            "feature_test of the two populations. The sample sizes of the set to be validated "
            "and the reference set are 4 and 4 respectively. The ks statistical test has been "
            "used for measuring the similarity between the two datasets. The corresponding "
            "distance between the distributions is: 0.000000 (p-value = 1.000000). The test "
            "result is FAIL for a comparison of the pvalue with the accepted threshold 0.100000."
        ),
        "result": {"probability": 1.0, "status": "FAIL"},
        "type": "validation",
        "version": "0.1",
    }


def test_extract_hist():
    data, bins = validation.extract_hist([1, 2, 3, 4], bins=4)
    assert data == [1 / 3] * 4
    assert bins == [1.38, 2.12, 2.88, 3.62]


def test_stat_test():
    results, res = validation.stat_test(
        [1, 2, 3, 4], [1, 2, 3, 4], stats.StatTests.ks, fargs=0.1
    )
    assert results.dist == 0.0
    assert results.pvalue == 1.0
    assert res == "PASS"


def test_write_hist(test_single):
    results = validation.write_hist(
        [1, 2, 3, 4], feature="feature_test", name="mtype_test", bins=4
    )
    assert results.keys() == test_single.keys()
    for i in results.keys():
        assert results[i] == test_single[i]


def test_unpack_config_data(CONFIG):
    bins, stat_test, thresh, criterion = validation.unpack_config_data(
        CONFIG, component="neurite_test", name="mtype_test", feature="feature_test"
    )
    assert bins == 4
    assert stat_test == stats.StatTests.ks
    assert thresh == 0.1
    assert criterion == "dist"


def test_write_all(CONFIG, test_dict):
    results = validation.write_all(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        component="neurite_test",
        feature="feature_test",
        name="mtype_test",
        config=CONFIG,
    )
    assert results.keys() == test_dict.keys()
    test_dict["datasets"]["reference"]["name"] = "-".join(
        [test_dict["datasets"]["reference"]["name"], "reference"]
    )
    test_dict["datasets"]["validation"]["name"] = "-".join(
        [test_dict["datasets"]["validation"]["name"], "test"]
    )
    for i in results.keys():
        assert results[i] == test_dict[i]