"""Tests for the morphval package."""

from pathlib import Path

import pkg_resources
import pytest

import morphval
import morphval.config
import morphval.validation_main

_distribution = pkg_resources.get_distribution("synthesis-workflow")
DATA = Path(__file__).parent / "data"
TEST_DIR = DATA / "test"
REF_DIR = DATA / "reference"
OUTPUT_DIR = DATA / "reports"
CONFIGS = Path(__file__).parent.parent.parent / "examples/morphval_config"


def test_import_config():
    """Test that a config is properly loaded."""
    config = morphval.config.load_config(CONFIGS / "config_bio.yaml")
    assert isinstance(config, dict)


def test_import_morphval():
    """Check that morphval loads the TEMPLATE correctly."""
    assert morphval.validation_main.TEMPLATE_FILE.endswith(
        "morphval/templates/report_template.jinja2"
    )
    assert Path(morphval.validation_main.TEMPLATE_FILE).exists()


@pytest.mark.skip(reason="TODO: Write this test")
def test_validation_conf():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_init_results():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_validate_features():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_validate_feature():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_compose_validation_criterion():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_compute_validation_scores():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_generate_report_data():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_write_report():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_compute_summary_statistics():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_compute_statistical_tests():
    """TODO: Write this test."""
    pass


@pytest.mark.skip(reason="TODO: Write this test")
def test_plot_save_feature():
    """TODO: Write this test."""
    pass
