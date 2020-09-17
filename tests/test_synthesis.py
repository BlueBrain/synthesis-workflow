"""Tests for synthesis module."""
import pytest

import pandas as pd

from synthesis_workflow import synthesis


@pytest.fixture
def empty_morph_df():
    df = pd.DataFrame()
    return df


def test_get_mean_neurite_lengths(empty_morph_df):
    """Test the computation of the mean neurite lengths"""
    res = synthesis.get_mean_neurite_lengths(empty_morph_df)
    assert res == {}
