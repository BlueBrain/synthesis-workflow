"""CLI for validation workflows."""
import argparse
import logging
import os
import sys

import luigi

from synthesis_workflow.tasks import workflows


L = logging.getLogger(__name__)


WORKFLOW_TASKS = {
    "ValidateSynthesis": workflows.ValidateSynthesis,
    "ValidateVacuumSynthesis": workflows.ValidateVacuumSynthesis,
    "ValidateRescaling": workflows.ValidateRescaling,
}

LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

LUIGI_PARAMETERS = ["workers", "local_scheduler", "log_level"]


class ArgParser:
    """Class to build parser and parse arguments"""

    def __init__(self):
        self.parsers = self._get_parsers()

    @property
    def parser(self):
        """Return the root parser"""
        return self.parsers["root"]

    def _get_parsers(self):
        """Return the main argument parser"""
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

        return self._get_workflow_parsers(parser)

    @staticmethod
    def _get_workflow_parsers(parser=None):
        """Return the workflow argument parser

        If parser is None, a new parser is created with the workflows as subparsers,
        otherwise if it is supplied, the parsers are added as subparsers.

        For each task listed in WORKFLOW_TASKS, a subparser is created as if it was
        created by luigi.
        """
        if not parser:
            parser = argparse.ArgumentParser()

        parsers = {"root": parser}

        workflow_parser = parser.add_subparsers(
            help="Possible workflows", dest="workflow", required=True
        )

        for workflow_name, task in WORKFLOW_TASKS.items():
            try:
                cls = task()
                task_name = cls.__class__.__name__
                doc = cls.__class__.__doc__
                subparser = workflow_parser.add_parser(workflow_name, help=doc)
                for param, param_obj in cls.get_params():
                    param_name = "--" + param.replace("_", "-")
                    subparser.add_argument(
                        param_name,
                        help=param_obj.description,
                        # pylint: disable=protected-access
                        **param_obj._parser_kwargs(param_name, task_name)
                    )
                parsers[workflow_name] = subparser
            except (AttributeError, TypeError):
                pass

        return parsers

    def parse_args(self, argv):
        """Parse the arguments, and return a argparse.Namespace object"""
        args = self.parser.parse_args(argv)

        return args


def _setup_logging(log_level, log_file=None, log_file_level=None):
    """Setup logging"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Setup logging formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s -- %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Setup console logging handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)
    root.addHandler(console)

    # Setup file logging handler
    if log_file is not None:
        if log_file_level is None:
            log_file_level = log_level
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_file_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def main(arguments=None):
    """Main function"""

    if arguments is None:
        arguments = sys.argv[1:]

    # Parse arguments
    parser = ArgParser()
    args = parser.parse_args(arguments)

    # Setup logger
    _setup_logging(args.log_level)
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

    # Use current logger configuration for luigi logger
    luigi.setup_logging.InterfaceLogging.config.set(
        "core", "no_configure_logging", "True"
    )

    # Get arguments to configure luigi
    luigi_config = {k: v for k, v in vars(args).items() if k in LUIGI_PARAMETERS}

    # Prepare workflow task and aguments
    task = WORKFLOW_TASKS[args.workflow]
    args_dict = {k: v for k, v in vars(args).items() if k in task.get_param_names()}

    # Run the luigi task
    luigi.build([WORKFLOW_TASKS[args.workflow](**args_dict)], **luigi_config)


if __name__ == "__main__":
    main()
