"""Test the `morphval.validation` module."""

from copy import deepcopy

import pytest
from neurom import stats

from morphval import validation


@pytest.fixture
def CONFIG():
    """The configuration used in tests."""
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
    """A single result entry."""
    return {
        "data": {"bin_center": [1.38, 2.12, 2.88, 3.62], "entries": [1 / 3] * 4},
        "data_type": "Histogram1D",
        "labels": {"bin_center": "Feature test", "entries": "Fraction"},
        "name": "mtype_test",
    }


@pytest.fixture
def test_dict(test_single):
    """A validation result."""
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
    """Test the `extract_hist()` function."""
    data, bins = validation.extract_hist([1, 2, 3, 4], bins=4)
    assert data == [1 / 3] * 4
    assert bins == [1.38, 2.12, 2.88, 3.62]


def test_stat_test():
    """Test the `stat_test()` function."""
    results, res = validation.stat_test([1, 2, 3, 4], [1, 2, 3, 4], stats.StatTests.ks, fargs=0.1)
    assert results.dist == 0.0
    assert results.pvalue == 1.0
    assert res == "PASS"


def test_write_hist(test_single):
    """Test the `write_hist()` function."""
    results = validation.write_hist([1, 2, 3, 4], feature="feature_test", name="mtype_test", bins=4)
    assert results.keys() == test_single.keys()
    for i in results.keys():
        assert results[i] == test_single[i]


def test_unpack_config_data(CONFIG):
    """Test the `unpack_config_data()` function."""
    bins, stat_test, thresh, criterion = validation.unpack_config_data(
        CONFIG, component="neurite_test", name="mtype_test", feature="feature_test"
    )
    assert bins == 4
    assert stat_test == stats.StatTests.ks
    assert thresh == 0.1
    assert criterion == "dist"


def test_write_all(CONFIG, test_dict):
    """Test the `write_all()` function."""
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
