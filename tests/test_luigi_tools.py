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


def test_copy_params_with_globals(luigi_tools_working_directory):
    class TaskA(luigi_tools.GlobalParamTask):
        """"""

        a = luigi.Parameter(default="a")
        a_cfg = luigi.Parameter(default="a_cfg")

        def run(self):
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


def create_empty_file(filename):
    with open(filename, "w") as f:
        pass


def check_empty_file(filename):
    with open(filename) as f:
        return f.read() == ""


def create_not_empty_file(filename):
    with open(filename, "w") as f:
        f.write("NOT EMPTY")


def check_not_empty_file(filename):
    with open(filename) as f:
        return f.read() == "NOT EMPTY"


def set_new_state(task_class):
    """Set a new state to a luigi.Task class to force luigi to check this class again"""
    task_class.counter = luigi.IntParameter(default=task_class().counter + 1)


def test_forceable_tasks(tmpdir):
    class TaskA(luigi_tools.WorkflowTask):
        """"""

        counter = luigi.IntParameter(default=0)
        rerun = luigi.BoolParameter()

        def run(self):
            for i in luigi.task.flatten(self.output()):
                create_not_empty_file(i.path)

        def output(self):
            return luigi.LocalTarget(tmpdir / "TaskA.target")

    class TaskB(luigi_tools.WorkflowTask):
        """"""

        counter = luigi.IntParameter(default=0)
        rerun = luigi.BoolParameter()

        def requires(self):
            return TaskA()

        def run(self):
            for i in luigi.task.flatten(self.output()):
                create_not_empty_file(i.path)

        def output(self):
            return [
                luigi.LocalTarget(tmpdir / "TaskB.target"),
                [
                    luigi.LocalTarget(tmpdir / "TaskB2.target"),
                    luigi.LocalTarget(tmpdir / "TaskB3.target"),
                ],
            ]

    class TaskC(luigi_tools.WorkflowTask):
        """"""

        counter = luigi.IntParameter(default=0)
        rerun = luigi.BoolParameter()

        def requires(self):
            return TaskA()

        def run(self):
            for i in luigi.task.flatten(self.output()):
                create_not_empty_file(i.path)

        def output(self):
            return {
                "first_target": luigi.LocalTarget(tmpdir / "TaskC.target"),
                "second_target": luigi.LocalTarget(tmpdir / "TaskC2.target"),
            }

    class TaskD(luigi_tools.WorkflowTask):
        """"""

        counter = luigi.IntParameter(default=0)
        rerun = luigi.BoolParameter()

        def requires(self):
            return [TaskB(), TaskC()]

        def run(self):
            for i in luigi.task.flatten(self.output()):
                create_not_empty_file(i.path)

        def output(self):
            return [
                luigi.LocalTarget(tmpdir / "TaskD.target"),
                luigi.LocalTarget(tmpdir / "TaskD2.target"),
            ]

    class TaskE(luigi_tools.WorkflowTask):
        """"""

        counter = luigi.IntParameter(default=0)
        rerun = luigi.BoolParameter()

        def requires(self):
            return TaskD()

        def run(self):
            for i in luigi.task.flatten(self.output()):
                create_not_empty_file(i.path)

        def output(self):
            return {
                "first_target": luigi.LocalTarget(tmpdir / "TaskE.target"),
                "other_targets": {
                    "second_target": luigi.LocalTarget(tmpdir / "TaskE2.target"),
                    "third_target": luigi.LocalTarget(tmpdir / "TaskE3.target"),
                },
            }

    all_targets = {}
    for task in [TaskA(), TaskB(), TaskC(), TaskD(), TaskE()]:
        all_targets[task.__class__.__name__] = task.output()

    # Test that everything is run when all rerun are False and targets are missing
    print("=================== FIRST BUILD ====================")
    for task_class in [TaskA, TaskB, TaskC, TaskD, TaskE]:
        set_new_state(task_class)

    assert luigi.build([TaskE()], local_scheduler=True)

    assert all([check_not_empty_file(i.path) for i in luigi.task.flatten(all_targets)])

    # Test that nothing is run when all rerun are False and targets are present
    for i in luigi.task.flatten(all_targets):
        create_empty_file(i.path)

    print("=================== SECOND BUILD ====================")
    for task_class in [TaskA, TaskB, TaskC, TaskD, TaskE]:
        set_new_state(task_class)
    assert luigi.build([TaskE()], local_scheduler=True)

    assert all([check_empty_file(i.path) for i in luigi.task.flatten(all_targets)])

    # Test that everything is run when rerun = True for the root task and targets are present
    for i in luigi.task.flatten(all_targets):
        create_empty_file(i.path)

    print("=================== THIRD BUILD ====================")
    for task_class in [TaskA, TaskB, TaskC, TaskD, TaskE]:
        set_new_state(task_class)
    TaskA.rerun = luigi.BoolParameter(default=True)
    assert luigi.build([TaskE()], local_scheduler=True)
    TaskA.rerun = luigi.BoolParameter()

    assert all([check_not_empty_file(i.path) for i in luigi.task.flatten(all_targets)])

    # Test that only the parents of the task with rerun = True are run
    for i in luigi.task.flatten(all_targets):
        create_empty_file(i.path)

    print("=================== FORTH BUILD ====================")
    for task_class in [TaskA, TaskB, TaskC, TaskD, TaskE]:
        set_new_state(task_class)
    TaskB.rerun = luigi.BoolParameter(default=True)
    assert luigi.build([TaskE()], local_scheduler=True)
    TaskB.rerun = luigi.BoolParameter()

    assert all(
        [
            check_not_empty_file(i.path)
            for task_name, targets in all_targets.items()
            for j in luigi.task.flatten(targets)
            if task_name not in ["TaskA", "TaskC"]
        ]
    )
    assert all(
        [
            check_empty_file(j.path)
            for task_name, targets in all_targets.items()
            for j in luigi.task.flatten(targets)
            if task_name in ["TaskA", "TaskC"]
        ]
    )

    # Test that calling a task inside another one does not remove its targets

    class TaskF(luigi_tools.WorkflowTask):
        """"""

        counter = luigi.IntParameter(default=0)
        rerun = luigi.BoolParameter()

        def requires(self):
            return TaskE()

        def run(self):
            # Call A inside F but the targets of A should not be removed
            _ = TaskA(counter=999)

            for i in luigi.task.flatten(self.output()):
                create_not_empty_file(i.path)

        def output(self):
            return {
                "first_target": luigi.LocalTarget(tmpdir / "TaskF.target"),
                "other_targets": {
                    "second_target": luigi.LocalTarget(tmpdir / "TaskE2.target"),
                    "third_target": luigi.LocalTarget(tmpdir / "TaskE3.target"),
                },
            }

    for i in luigi.task.flatten(all_targets):
        create_empty_file(i.path)

    print("=================== FORTH BUILD ====================")
    for task_class in [TaskA, TaskB, TaskC, TaskD, TaskE]:
        set_new_state(task_class)
    TaskB.rerun = luigi.BoolParameter(default=True)
    assert luigi.build([TaskF()], local_scheduler=True)
    TaskB.rerun = luigi.BoolParameter()

    assert all(
        [
            check_not_empty_file(i.path)
            for task_name, targets in all_targets.items()
            for j in luigi.task.flatten(targets)
            if task_name not in ["TaskA", "TaskC"]
        ]
    )
    assert all(
        [
            check_empty_file(j.path)
            for task_name, targets in all_targets.items()
            for j in luigi.task.flatten(targets)
            if task_name in ["TaskA", "TaskC"]
        ]
    )


def test_remove_folder_target(tmpdir):
    class TaskA(luigi_tools.WorkflowTask):
        """"""

        def run(self):
            for i in luigi.task.flatten(self.output()):
                i.makedirs()

            os.makedirs(self.output()[0].path)
            create_not_empty_file(self.output()[1].path)
            create_not_empty_file(self.output()[0].path + "/file.test")

            for i in luigi.task.flatten(self.output()):
                assert i.exists()
                luigi_tools.target_remove(i)
                assert not i.exists()

        def output(self):
            return [
                luigi.LocalTarget(tmpdir / "TaskA"),
                luigi.LocalTarget(tmpdir / "TaskA_bis" / "file.test"),
            ]

    assert luigi.build([TaskA()], local_scheduler=True)


def test_output_target(tmpdir):
    """
    4 tests for the OutputLocalTarget class:
        * using explicit prefix, so the default prefix is ignored
        * using absolute path, so the prefix is ignored
        * using explicit prefix with relative paths, so the default prefix is ignored
        * using default prefix
    """

    class TaskA_OutputLocalTarget(luigi_tools.WorkflowTask):
        """"""

        def run(self):
            """"""

            for i in luigi.task.flatten(self.output()):
                os.makedirs(i.ppath.parent, exist_ok=True)
                create_not_empty_file(i.path)
                assert i.exists()
                luigi_tools.target_remove(i)
                assert not i.exists()

        def output(self):
            return [
                luigi_tools.OutputLocalTarget("output_target.test", prefix=tmpdir),
                luigi_tools.OutputLocalTarget(
                    tmpdir / "absolute_output_target_no_prefix.test"
                ),
                luigi_tools.OutputLocalTarget(
                    "relative_output_target.test", prefix=tmpdir / "test" / ".."
                ),
                luigi_tools.OutputLocalTarget("output_target_default_prefix.test"),
                luigi_tools.OutputLocalTarget(
                    tmpdir / "absolute_output_target_prefix.test",
                    prefix=tmpdir / "test",
                ),
            ]

    try:
        current_prefix = luigi_tools.OutputLocalTarget._prefix
        luigi_tools.OutputLocalTarget.set_default_prefix(tmpdir / "subdir")
        assert luigi.build([TaskA_OutputLocalTarget()], local_scheduler=True)
    finally:
        luigi_tools.OutputLocalTarget.set_default_prefix(current_prefix)
