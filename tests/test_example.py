"""Sample nosetest file."""
import pytest

import pandas as pd

from synthesis_workflow import synthesis


@pytest.fixture
def simple_morph_df():
    df = pd.DataFrame()
    return df


def test_get_mean_neurite_lengths(simple_morph_df):
    """Test the computation of the mean neurite lengths"""
    res = synthesis.get_mean_neurite_lengths(simple_morph_df)
    assert res
