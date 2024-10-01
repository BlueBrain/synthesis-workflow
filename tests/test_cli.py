"""Tests for the synthesis_workflow.cli module."""

import re

from synthesis_workflow.tasks import cli


class TestCLI:
    """Test the CLI of the synthesis-workflow package."""

    def test_help(self, capsys):
        """Test the --help argument."""
        try:
            cli.main(arguments=["--help"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert (
            re.match(
                r"usage: \S+ .*Run the workflow\n\npositional arguments:\s*"
                r"{ValidateSynthesis,ValidateVacuumSynthesis,ValidateRescaling}\s*"
                r"Possible workflows.*",
                captured.out,
                flags=re.DOTALL,
            )
            is not None
        )

    def test_dependency_graph(self, vacuum_working_directory):
        """Test the --create-dependency-graph argument."""
        root_dir = vacuum_working_directory[0].parent
        cli.main(
            arguments=[
                "--create-dependency-graph",
                str(root_dir / "dependency_graph.png"),
                "ValidateVacuumSynthesis",
            ]
        )

        assert (root_dir / "dependency_graph.png").exists()


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run("synthesis-workflow", "--version")
    assert ret.success
    assert ret.stdout.startswith("synthesis-workflow, version ")
    assert ret.stderr == ""
