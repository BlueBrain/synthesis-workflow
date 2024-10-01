"""Main module of MorphVal package."""

import collections
import os
import time
from pathlib import Path

import jinja2
import neurom
import numpy as np
import pandas as pd
import pkg_resources
from joblib import Parallel
from joblib import cpu_count
from joblib import delayed

from morphval import common
from morphval import validation

TEMPLATE_FILE = pkg_resources.resource_filename("morphval", "templates/report_template.jinja2")
SUMMARY_TEMPLATE_FILE = pkg_resources.resource_filename(
    "morphval", "templates/report_summary_template.jinja2"
)


def save_csv(dir_name, feature, data):
    """Save data to CSV file."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_name = os.path.join(dir_name, feature + ".csv")
    np.savetxt(file_name, data, delimiter=",")
    return file_name


def load_template(template_file):
    """Load a template."""
    t_dir, t_file = os.path.split(template_file)
    templateLoader = jinja2.FileSystemLoader(searchpath=t_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    return templateEnv.get_template(t_file)


def count_passing_validations(features):
    """Count the number of passing, and total validations for a dict of features.

    Returns:
        returns tuple(feature_pass, features_total)
    """
    num_pass = sum(v["validation_criterion"]["status"] == "PASS" for v in features.values())
    return num_pass, len(features)


def compute_validation_criterion(config, stat_test_results):
    """Based on the config thresholds and criterion, computes if validation passed or failed."""
    ret = collections.OrderedDict(
        [
            ("threshold", config["threshold"]),
            ("criterion", config["criterion"]),
        ]
    )

    if config["criterion"] == "pvalue":
        ret["value"] = value = stat_test_results[1]
        ret["status"] = "FAIL" if value < config["threshold"] else "PASS"
    elif config["criterion"] == "dist":
        ret["value"] = value = stat_test_results[0]
        ret["status"] = "FAIL" if value > config["threshold"] else "PASS"
    return ret


def do_validation(validation_config, ref_population, test_population):
    """Validate a test population against a reference population.

    Args:
        validation_config(dict): {component: {feature: {...}}}
        ref_population(NeuroM morph population): reference population
        test_population(NeuroM morph population): test population

    Returns:
        tuple of morphometrics, results, where morphometrics is a dictionary containing
        the raw feature values for the morphologies, and the results is a dictionary
        containing statistical results on this raw data
    """
    results = collections.OrderedDict()
    morphometrics = {}
    for component_name, features in validation_config.items():
        results[component_name] = component_results = collections.OrderedDict()
        morphometrics[component_name] = component_metrics = {}
        for feature_name, feature_config in features.items():
            component_results[feature_name] = feature_results = collections.OrderedDict()

            test_data, ref_data = validation.extract_feature(
                test_population, ref_population, component_name, feature_name
            )

            component_metrics[feature_name] = {
                "test": test_data,
                "ref": ref_data,
            }

            feature_results["test_summary_statistics"] = compute_summary_statistics(test_data)
            feature_results["ref_summary_statistics"] = compute_summary_statistics(ref_data)

            test_name = feature_config["stat_test"]

            feature_results["statistical_tests"] = test_results = compute_statistical_tests(
                test_data,
                ref_data,
                feature_config["stat_test"],
                feature_config["threshold"],
            )

            feature_results["validation_criterion"] = compute_validation_criterion(
                feature_config, test_results[test_name]["results"]
            )

    return morphometrics, results


def write_morphometrics(output_dir, morphometrics):
    """Dump CSV of morphometrics for each of the features."""
    for component_name, features in morphometrics.items():
        for feature_name, feature_metrics in features.items():
            for kind in ("test", "ref"):
                dir_name = os.path.join(output_dir, component_name, "morphometrics", kind)
                save_csv(dir_name, feature_name, feature_metrics[kind])


def create_morphometrics_histograms(output_dir, morphometrics, config, notebook_desc=None):
    """Create histograms based on morphometrics."""
    m_items = common.add_progress_bar(morphometrics.items(), "[{}] Histograms", notebook_desc)
    for component_name, features in m_items:
        figure_dir = os.path.join(output_dir, component_name, "figures")

        c_name = f"-- {common.pretty_name(component_name).capitalize()}s"
        f_items = common.add_progress_bar(features.items(), c_name, notebook_desc)
        for feature_name, feature_metrics in f_items:
            test_data, ref_data = feature_metrics["test"], feature_metrics["ref"]
            plot_save_feature(
                figure_dir,
                test_data,
                ref_data,
                feature_name,
                config[component_name][feature_name]["bins"],
            )


def validate_feature(
    mtype, config, output_dir, ref_files, test_files, cell_figure_count, notebook=False
):
    """Validate one feature."""
    ref_population = neurom.load_morphologies(ref_files, cache=True)
    test_population = neurom.load_morphologies(test_files, cache=True)

    morphometrics, result_mtype = do_validation(config, ref_population, test_population)

    write_morphometrics(output_dir, morphometrics)

    notebook_desc = mtype if notebook else None
    create_morphometrics_histograms(output_dir, morphometrics, config, notebook_desc)

    ref_plots_mtype, test_plots_mtype = common.plot_normalized_neurons(
        os.path.join(output_dir, "figures"),
        ref_population,
        test_population,
        cell_figure_count,
        config.keys(),
        notebook_desc,
    )

    return mtype, result_mtype, ref_plots_mtype, test_plots_mtype


class Validation:
    """Validation state object.

    This class holds the state information of a validation run.
    """

    def __init__(
        self,
        config,
        test_data,
        ref_data,
        output_dir,
        create_timestamp_dir=True,
        notebook=False,
    ):
        """Create a new Validation object.

        Args:
            config (dict): which validations to perform, and how:
                ex {'L23_PC':       # cell type
                     {'soma':        # component
                       {'soma_radii': # feature
                         {'stat_test': 'StatTests.ks',  # test configuration
                          'threshold': 0.1,
                          'bins': 40,
                          'criterion': 'dist'
                          }}}}
            test_data (str path or pandas.DataFrame): directory to the files under test or a
                pandas.DataFrame with 'mtype' and 'filepath' columns
            ref_data (str path or pandas.DataFrame): directory to the reference files or a
                pandas.DataFrame with 'mtype' and 'filepath' columns
            output_dir (str path): where the output should be written
            create_timestamp_dir (bool): whether a directory should be created in output_dir,
                with the timestamp of the run
            notebook (bool): trigger the Jupyter notebook mode
        """
        self.timestamp = time.strftime("%Y%m%d-%H%M")

        self.config = config

        # list files
        self.ref_dir, self.ref_files = self._arg_to_file_list(ref_data, "ref_data")
        self.test_dir, self.test_files = self._arg_to_file_list(test_data, "test_data")

        # Setup output directory
        self.output_dir = Path(output_dir)
        if create_timestamp_dir:
            self.output_dir = self.output_dir / ("validation_results-" + self.timestamp)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = None
        self.results_file = None

        # list paths to plots, keyed on mtype
        self.test_plots = {}
        self.ref_plots = {}

        self.notebook = notebook

    def _arg_to_file_list(self, arg, arg_name):
        if isinstance(arg, str):
            arg_dir = Path(arg).absolute()
            arg_files = self._list_files(arg_dir)
        elif isinstance(arg, pd.DataFrame):
            arg_files = arg
            if not arg_files.empty:
                arg_dir = Path(arg_files.iloc[0]["filepath"]).parent
            else:
                raise ValueError(f"The {arg_name} DataFrame is empty")
        else:
            raise TypeError(f"The {arg_name} argument must be a string path or a pandas.DataFrame")
        return arg_dir, arg_files

    def _list_files(self, directory):
        files = collections.defaultdict(list)
        d = Path(directory)
        for mtype in self.config:
            # List files
            sub_d = d / mtype
            for file in sub_d.iterdir():
                files["mtype"].append(mtype)
                files["filepath"].append(file.absolute())
        return pd.DataFrame(files)

    def validate_features(self, cell_figure_count=100, nb_jobs=-1, joblib_verbose=0):
        """Validate all features."""
        self.results = results = collections.OrderedDict()

        batch_size = 1 + int(len(self.config) / (nb_jobs if nb_jobs > 0 else cpu_count()))

        for res_mtype, res, ref_p, test_p in Parallel(
            nb_jobs,
            verbose=joblib_verbose,
            backend="multiprocessing",
            batch_size=batch_size,
        )(
            delayed(validate_feature)(
                mtype,
                config,
                self.output_dir / mtype,
                self.ref_files.loc[self.ref_files["mtype"] == mtype, "filepath"].tolist(),
                self.test_files.loc[self.test_files["mtype"] == mtype, "filepath"].tolist(),
                cell_figure_count,
                False,
            )
            for mtype, config in self.config.items()
        ):
            results[res_mtype] = res
            self.ref_plots[res_mtype] = ref_p
            self.test_plots[res_mtype] = test_p

        self.results_file = common.dump2json(self.output_dir, "validation_results", results)
        return self.results_file

    def generate_report_data(self, mtype):
        """Generate dictionary with the text that will fill the template.

        It contains all data of the results dictionary in text form with
        additional information on the directories where the data come from.
        tt is a shortcut for template text.
        """
        tt = {}

        # c is a shortcut for component (NeuriteType) and f for feature.
        mtype_results = self.results[mtype]
        total_num_pass, total_num_features = 0, 0
        for c, component_results in mtype_results.items():
            num_pass, num_features = count_passing_validations(component_results)
            total_num_pass += num_pass
            total_num_features += num_features
            tt[c] = {
                "name": c.capitalize().replace("_", " "),
                # pass the validation scores per component to the
                # template text (tt) as a string
                "num_pass": num_pass,
                "num_features": num_features,
                "pass_percentage": f"{(100.0 * num_pass) / num_features:5.2f}",
            }

            for f, feature_results in component_results.items():
                tt[c][f] = self.merge_results_features(
                    mtype=mtype,
                    component=c,
                    feature_name=f,
                    feature_config=self.config[mtype][c][f],
                    feature_results=feature_results,
                )

        tt["num_pass"] = total_num_pass
        tt["num_features"] = total_num_features
        tt["pass_percentage"] = f"{(100.0 * total_num_pass) / total_num_features:5.2f}"

        return tt

    @staticmethod
    def merge_results_features(mtype, component, feature_name, feature_config, feature_results):
        """Merge result features."""
        stat_test = feature_config["stat_test"]
        stat_test_results = feature_results["statistical_tests"][stat_test]["results"]
        results_validation_criterion = feature_results["validation_criterion"]
        ret = {
            "feature_histogram_file": os.path.join(
                "..", mtype, component, "figures", feature_name + ".png"
            ),
            "name": feature_name.capitalize().replace("_", " "),
            "stat_test": stat_test.split(".")[1].upper() + " test",
            "stat_test_result": (f"{stat_test_results[0]:10.4f}", f"{stat_test_results[1]:1.4g}"),
            "validation_criterion": {
                "value": f"{results_validation_criterion['value']:1.5g}",
                "threshold": f"{results_validation_criterion['threshold']:g}",
                "status": results_validation_criterion["status"],
                "criterion": results_validation_criterion["criterion"],
            },
        }

        ret["test_summary_statistics"] = dict(
            (k, f"{feature_results['test_summary_statistics'][k]:10.2f}")
            for k in feature_results["test_summary_statistics"]
        )

        ret["ref_summary_statistics"] = dict(
            (k, f"{feature_results['ref_summary_statistics'][k]:10.2f}")
            for k in feature_results["ref_summary_statistics"]
        )

        return ret

    def write_report(self, validation_report=True, template_file=TEMPLATE_FILE, prefix="report-"):
        """For each mtype in the results, write out its report.

        Args:
            validation_report(bool): True if 'validation' to be shown in report,
                False if p-values shown instead, with no 'Pass/Fail' information
            template_file(str): template file name
            prefix(str): report is saved as <prefix> + <mtype> + '.html'
        """
        report_dir = os.path.join(self.output_dir, "html")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        output_files = []
        for mtype in self.results:
            output_text = self.render_mtype_report(template_file, mtype, validation_report)
            output_files.append(os.path.join(report_dir, prefix + mtype + ".html"))
            with open(output_files[-1], "w", encoding="utf-8") as outputFile:
                outputFile.write(output_text)
        return output_files

    def write_report_summary(
        self,
        validation_report=True,
        template_file=SUMMARY_TEMPLATE_FILE,
        prefix="report-summary-",
    ):
        """Write summary report."""
        return self.write_report(
            validation_report=validation_report,
            template_file=template_file,
            prefix=prefix,
        )

    def render_mtype_report(self, template_file, mtype, validation_report):
        """Render mtype report."""
        template = load_template(template_file)
        templateText = self.generate_report_data(mtype)

        config_file = common.dump2json(self.output_dir, "validation_config", self.config)

        templateVars = {
            "output_title": "Validation report: " + mtype,
            "description": f"Validation summary for cell type {mtype}",
            "timestamp": self.timestamp,
            "results_dir": self.output_dir,
            "test_dir": self.test_dir.as_posix(),
            "ref_dir": self.ref_dir.as_posix(),
            "config_file": config_file,
            "template_file": template_file,
            "mtype": mtype,
            "mtype_results": self.results[mtype],
            "templateText": templateText,
            "test_cell_files": {
                comp: [
                    path.replace(self.output_dir.as_posix(), "..")
                    for path in self.test_plots[mtype][comp]
                ]
                for comp in self.test_plots[mtype]
            },
            "ref_cell_files": {
                comp: [
                    path.replace(self.output_dir.as_posix(), "..")
                    for path in self.ref_plots[mtype][comp]
                ]
                for comp in self.ref_plots[mtype]
            },
            "validation_report": validation_report,
        }

        return template.render(templateVars)


def compute_summary_statistics(data):
    """Compute the summary statistics of a feature.

    Args:
        data : the feature data array

    Returns:
        The dictionary summary_statistics which contains sample size,
        mean, standard deviation and median.
    """
    summary_statistics = collections.OrderedDict(
        [
            ("sample_size", len(data)),
            ("mean", np.mean(data)),
            ("std", np.std(data)),
            ("median", np.median(data)),
        ]
    )
    return summary_statistics


def compute_statistical_tests(test_data, ref_data, test_name, thresh):
    """Compute the test statistic and the p-value of a statistical test.

    Args:
        test_data: the test feature array
        ref_data: the reference feature array
        test: the statistical test (ex.: KS test)
        thresh: the threshold

    Returns:
        the dictionary : statistical tests
    """
    test = validation.load_stat_test(test_name)
    results, status = validation.stat_test(test_data, ref_data, test, thresh)

    ret = collections.OrderedDict(
        [
            (
                test_name,
                collections.OrderedDict(
                    [
                        ("results", results),
                        ("status", status),
                    ]
                ),
            ),
        ]
    )
    return ret


def plot_save_feature(figures_dir, test_data, ref_data, feature, bin_count):
    """Plot and saves the figure of the overlaid histograms of feature distributions.

    The result image is a .png figure.

    Args:
        figures_dir (str): the directory where the figures are stored,
        test_data (np.array): test feature array
        ref_data (np.array): reference feature array
        feature (str): the feature name
        bin_count (int): the number of histogram bins
    """
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # define the bins from the validation report
    bins_defined = np.linspace(
        min(np.min(test_data), np.min(ref_data)),
        max(np.max(test_data), np.max(ref_data)),
        bin_count,
    )

    with common.pyplot_non_interactive():
        with common.get_agg_fig() as fig:
            ax = fig.add_subplot(111)
            ax.hist(
                [ref_data, test_data],
                bins_defined,
                alpha=0.5,
                density=True,
                color=["red", "blue"],
                label=["Reference Data", "Test Data"],
            )
            # ax.xticks(fontsize=18)
            # ax.yticks(fontsize=18)
            ax.legend(fontsize=18)
            fig.savefig(os.path.join(figures_dir, feature + ".png"))
