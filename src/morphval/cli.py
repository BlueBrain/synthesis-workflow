#!/usr/bin/env python
"""CLI for MorphVal package."""
import argparse

import pkg_resources

from morphval import config
from morphval.validation_main import Validation


def get_parser():
    """Return the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--test-dir", required=True, help="full path to directory with test data"
    )
    parser.add_argument(
        "-r",
        "--ref-dir",
        required=True,
        help="full path to directory with reference data",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="full path to directory for the validation results",
    )
    parser.add_argument("-c", "--config", required=True, help="full path to yaml config file")
    parser.add_argument(
        "--example-config",
        action="store_true",
        help=(
            "print out an example config in which the 'mtype' key should be replaced "
            "by an actual mtype"
        ),
    )
    parser.add_argument(
        "--bio-compare",
        action="store_true",
        default=False,
        help="Use the bio compare template",
    )
    parser.add_argument(
        "--cell-figure-count",
        default=10,
        type=int,
        help="Number of example cells to show",
    )
    return parser


def main():
    """Main function of MorphVal package."""
    args = get_parser().parse_args()

    if args.example_config:
        print(
            pkg_resources.resource_string(
                "synthesis_workflow", "defaults/morphval_default_config.yaml"
            )
        )
        return

    my_config = config.load_config(args.config)
    validation = Validation(my_config, args.test_dir, args.ref_dir, args.output_dir)
    validation.validate_features(cell_figure_count=args.cell_figure_count)
    validation.write_report(validation_report=not args.bio_compare)


if __name__ == "__main__":
    main()
