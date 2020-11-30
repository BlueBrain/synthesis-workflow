"""CLI for validation workflows."""
import argparse
import inspect
import logging
import os
import re
import sys
from pathlib import Path

import luigi
from luigi_tools.util import get_dependency_graph

import synthesis_workflow
from synthesis_workflow.tasks import workflows
from synthesis_workflow.utils import setup_logging


L = logging.getLogger(__name__)


WORKFLOW_TASKS = {
    "ValidateSynthesis": workflows.ValidateSynthesis,
    "ValidateVacuumSynthesis": workflows.ValidateVacuumSynthesis,
    "ValidateRescaling": workflows.ValidateRescaling,
}

LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

LUIGI_PARAMETERS = ["workers", "local_scheduler", "log_level"]


_PARAM_NO_VALUE = [luigi.parameter._no_value, None]  # pylint: disable=protected-access


def _process_param(param):
    desc = param.description
    choices = None
    interval = None
    optional = False
    if isinstance(param, luigi.OptionalParameter):
        optional = True
    if isinstance(param, luigi.ChoiceParameter):
        desc, choices = desc.rsplit("Choices: ", 1)
    if isinstance(param, luigi.NumericalParameter):
        desc, interval = desc.rsplit("permitted values: ", 1)
    try:
        param_type, param_doc = re.match("(:.*?:)? *(.*)", desc).groups()
    except AttributeError:
        param_type = None
        param_doc = desc
    return param_doc, param_type, choices, interval, optional


class ArgParser:
    """Class to build parser and parse arguments."""

    def __init__(self):
        self.parsers = self._get_parsers()

    @property
    def parser(self):
        """Return the root parser."""
        return self.parsers["root"]

    def _get_parsers(self):
        """Return the main argument parser."""
        parser = argparse.ArgumentParser(
            description="Run the synthesis workflow",
        )

        parser.add_argument("-c", "--config-path", help="Path to the Luigi config file")

        parser.add_argument(
            "-l",
            "--local-scheduler",
            default=False,
            action="store_true",
            help="Use Luigi's local scheduler instead of master scheduler.",
        )

        parser.add_argument(
            "-ll",
            "--log-level",
            default="INFO",
            choices=LOGGING_LEVELS,
            help="Logger level.",
        )

        parser.add_argument("-lf", "--log-file", help="Logger file.")

        parser.add_argument(
            "-w",
            "--workers",
            type=int,
            default=1,
            help="Number of workers that luigi can summon.",
        )

        parser.add_argument(
            "-dg",
            "--create-dependency-graph",
            help=(
                "Create the dependency graph of a workflow instead of running it. "
                "Pass either 'ascii' to print the graph to screen or a path to render "
                "it as an image (depending on the extension of the given path)."
            ),
        )

        return self._get_workflow_parsers(parser)

    @staticmethod
    def _get_workflow_parsers(parser=None):
        """Return the workflow argument parser.

        If parser is None, a new parser is created with the workflows as subparsers,
        otherwise if it is supplied, the parsers are added as subparsers.

        For each task listed in WORKFLOW_TASKS, a subparser is created as if it was
        created by luigi.
        """
        if not parser:
            parser = argparse.ArgumentParser()

        parsers = {"root": parser}

        workflow_parser = parser.add_subparsers(
            help="Possible workflows", dest="workflow"
        )

        def format_description(param):
            try:
                param_doc, param_type, choices, interval, optional = _process_param(
                    param
                )
                if optional:
                    param_doc = "(optional) " + param_doc
                if param_type is not None:
                    param_type = f"({param_type.replace(':', '')})"
                    param_doc = f"{param_type} {param_doc}"
                if choices is not None:
                    param_doc = f"{param_doc} Choices: {choices}."
                if interval is not None:
                    param_doc = f"{param_doc} Permitted values: {interval}."
                # pylint: disable=protected-access
                if hasattr(param, "_default") and param._default not in _PARAM_NO_VALUE:
                    param_doc = f"{param_doc} Default value: {param._default}."
            except AttributeError:
                param_doc = param.description
            return param_doc

        for workflow_name, task in WORKFLOW_TASKS.items():
            try:
                task_name = task.__name__
                doc = task.__doc__
                subparser = workflow_parser.add_parser(workflow_name, help=doc)
                for param, param_obj in task.get_params():
                    param_name = "--" + param.replace("_", "-")
                    subparser.add_argument(
                        param_name,
                        help=format_description(param_obj),
                        # pylint: disable=protected-access
                        **param_obj._parser_kwargs(param_name, task_name),
                    )
                parsers[workflow_name] = subparser
            except (AttributeError, TypeError):
                pass

        return parsers

    def parse_args(self, argv):
        """Parse the arguments, and return a argparse.Namespace object."""
        args = self.parser.parse_args(argv)

        return args


def _setup_logging(log_level, log_file=None, log_file_level=None):
    """Setup logging."""
    setup_logging(log_level, log_file, log_file_level)


def main(arguments=None):
    """Main function."""
    if arguments is None:
        arguments = sys.argv[1:]

    # Parse arguments
    parser = ArgParser()
    args = parser.parse_args(arguments)

    L.debug("Args: %s", args)

    # Check that one workflow is in arguments
    if args is None or args.workflow is None:
        L.critical(
            "Arguments must contain one workflow. Check help with -h/--help argument."
        )
        parser.parser.print_help()
        sys.exit()

    # Set luigi.cfg path
    if args.config_path is not None:
        os.environ["LUIGI_CONFIG_PATH"] = args.config_path

    # Get arguments to configure luigi
    luigi_config = {k: v for k, v in vars(args).items() if k in LUIGI_PARAMETERS}

    # Prepare workflow task and aguments
    task = WORKFLOW_TASKS[args.workflow]
    args_dict = {k: v for k, v in vars(args).items() if k in task.get_param_names()}

    # Export the dependency graph of the workflow instead of running it
    if args.create_dependency_graph is not None:
        try:
            from graphviz import Digraph  # pylint: disable=import-outside-toplevel
        except ImportError:
            print("Could not import GraphViz, please install it.")
            sys.exit()
        task = WORKFLOW_TASKS[args.workflow](**args_dict)
        g = get_dependency_graph(task)
        default_graph_attrs = {
            "rankdir": "TB",
            "size": "7.0, 15.0",
            "bgcolor": "transparent",
        }
        default_node_attrs = {
            "shape": "box",
            "fontsize": "9",
            "height": "0.25",
            "fontname": '"Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans"',
            "style": "setlinewidth(0.5),filled",
            "fillcolor": "white",
        }
        default_edge_attrs = {
            "arrowsize": "0.5",
            "style": "setlinewidth(0.5)",
        }
        dot = Digraph(
            comment="Dependency graph",
            strict=True,
            graph_attr=default_graph_attrs,
            node_attr=default_node_attrs,
            edge_attr=default_edge_attrs,
        )
        dot.node(task.__class__.__name__, color="red", penwidth="1.5")
        base_f = Path(inspect.getfile(synthesis_workflow)).parent
        for parent, child in g:
            url = (
                Path(inspect.getfile(child.__class__))
                .relative_to(base_f)
                .with_suffix("")
                / "index.html"
            )
            anchor = "#" + ".".join(
                child.__module__.split(".")[1:] + [child.__class__.__name__]
            )
            dot.node(child.__class__.__name__, URL="../../" + url.as_posix() + anchor)
            dot.edge(
                parent.__class__.__name__,
                child.__class__.__name__,
                **default_edge_attrs,
            )
        filepath = Path(args.create_dependency_graph)
        filename = filepath.with_suffix("")
        pattern = re.compile(r"\.?(.*)")
        match = re.match(pattern, filepath.suffix)
        dot.render(
            filename=filename,
            format=match.group(1),
            cleanup=True,
        )
        sys.exit()

    # Run the luigi task
    luigi.build([WORKFLOW_TASKS[args.workflow](**args_dict)], **luigi_config)


if __name__ == "__main__":
    main()
