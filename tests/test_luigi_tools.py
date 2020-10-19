"""Tests for luigi tools."""
import os

import luigi
import pytest

from synthesis_workflow.tasks import luigi_tools


def test_copy_params(tmpdir):
    class TaskA(luigi.Task):
        """"""

        a = luigi.Parameter(default="default_value_a")

        def run(self):
            print(self.a)
            return self.a

        def output(self):
            return luigi.LocalTarget(tmpdir)

    @luigi_tools.copy_params(
        aa=luigi_tools.ParamLink(TaskA, "a"),
        a_default=luigi_tools.ParamLink(TaskA, "a", "given_default_value"),
    )
    class TaskB(luigi.Task):
        """"""

        b = luigi.Parameter(default="b")

        def run(self):
            print(self.aa, self.a_default, self.b)
            return self.aa, self.a_default, self.b

        def output(self):
            return luigi.LocalTarget(tmpdir)

    # Test with default value
    task = TaskB()
    res = task.run()

    assert res == ("default_value_a", "given_default_value", "b")

    # Test with another value
    task = TaskB(aa="z", a_default="new_default", b="bb")
    res = task.run()

    assert res == ("z", "new_default", "bb")


@pytest.mark.skipif(
    "LUIGI_CONFIG_PATH" not in os.environ,
    reason="The 'LUIGI_CONFIG_PATH' environment variable must be set",
)
def test_copy_params_with_globals():
    class TaskA(luigi_tools.GlobalParamTask):
        """"""

        a = luigi.Parameter(default="a")
        a_cfg = luigi.Parameter(default="a_cfg")

        def run(self):
            print(os.environ["LUIGI_CONFIG_PATH"])
            print(self.a)
            assert self.a == "a"
            assert self.a_cfg == "default_value_in_cfg"

        def output(self):
            return luigi.LocalTarget("not_existing_file")

    @luigi_tools.copy_params(
        aa=luigi_tools.ParamLink(TaskA, "a_cfg"),
        a_default=luigi_tools.ParamLink(TaskA, "a", "given_default_value"),
    )
    class TaskB(luigi_tools.GlobalParamTask):
        """"""

        b = luigi.Parameter(default="b")
        mode = luigi.Parameter(default="default")

        def run(self):
            print(os.environ["LUIGI_CONFIG_PATH"])
            print(self.mode, self.aa, self.a_default, self.b)
            if self.mode == "default":
                assert self.aa == "default_value_in_cfg"
                assert self.a_default == "given_default_value"
                assert self.b == "b"
            else:
                assert self.aa == "constructor_value"
                assert self.a_default == "new_default"
                assert self.b == "bb"
            return self.aa, self.a_default, self.b

        def output(self):
            return luigi.LocalTarget("not_existing_file")

    # Test with default value
    task = TaskB()
    res = task.run()

    assert res == ("default_value_in_cfg", "given_default_value", "b")

    # Test with another value
    task = TaskB(
        aa="constructor_value", a_default="new_default", b="bb", mode="constructor"
    )
    res = task.run()

    assert res == ("constructor_value", "new_default", "bb")

    assert luigi.build([TaskA(), TaskB()], local_scheduler=True)
    assert luigi.build(
        [
            TaskA(),
            TaskB(
                aa="constructor_value",
                a_default="new_default",
                b="bb",
                mode="constructor",
            ),
        ],
        local_scheduler=True,
    )
