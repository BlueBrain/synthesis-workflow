"""
Tests for the morphval package.
"""
import pkg_resources
from pathlib import Path

import morphval
import morphval.config
import morphval.validation_main


_distribution = pkg_resources.get_distribution("synthesis-workflow")
DATA = Path(__file__).parent / "data"
TEST_DIR = DATA / "test"
REF_DIR = DATA / "reference"
OUTPUT_DIR = DATA / "reports"
TEMPLATE_FILE = (
    Path(_distribution.get_resource_filename("morphval", "morphval"))
    / "templates"
    / "report_template.jinja2"
)
CONFIGS = Path(__file__).parent.parent.parent / "examples/morphval_config"


def test_import_config():
    config = morphval.config.load_config(CONFIGS / "config_bio.yaml")
    assert isinstance(config, dict)


def test_import_morphval():
    """Check if morphval loads the TEMPLATE correctly"""
    assert morphval.validation_main.TEMPLATE_FILE == TEMPLATE_FILE.as_posix()


def test_validation_conf():
    pass


def test_init_results():
    pass


def test_validate_features():
    pass


def test_validate_feature():
    pass


def test_compose_validation_criterion():
    pass


def test_compute_validation_scores():
    pass


def test_generate_report_data():
    pass


def test_write_report():
    pass


def test_compute_summary_statistics():
    pass


def test_compute_statistical_tests():
    pass


def test_plot_save_feature():
    pass


"""
def test_stat_test():
    results, res = validation.stat_test([1,2,3,4], [1,2,3,4], stats.StatTests.ks, fargs=0.1)
    nt.assert_true(results.dist==0.0)
    nt.assert_true(results.pvalue==1.0)
    nt.assert_true(res=="PASS")

def test_write_hist():
    results = validation.write_hist([1,2,3,4], feature='test', name='T', bins=4)
    nt.assert_true(results.keys() == test_single.keys())
    for i in results.keys():
        nt.assert_true(results[i] == test_single[i])

def test_unpack_config_data():
    bins, stat_test, thresh = validation.unpack_config_data(CONFIG, name='T', feature='test')
    nt.assert_true(bins==4)
    nt.assert_true(stat_test==stats.StatTests.ks)
    nt.assert_true(thresh==0.1)

def test_write_all():
    results = validation.write_all([1,2,3,4], [1,2,3,4], feature='test', name='T', config=CONFIG)
    nt.assert_true(results.keys() == test_dict.keys())
    for i in results.keys():
        nt.assert_true(results[i] == test_dict[i]) """
