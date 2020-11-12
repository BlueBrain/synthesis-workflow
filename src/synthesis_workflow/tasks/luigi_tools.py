"""utils functions for luigi parameters."""
import logging
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path

import luigi


L = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Exception raised when the workflow is not consistent."""


def recursive_check(task, attr="rerun"):
    """Check if a task or any of its recursive dependencies has a given attribute set to True."""
    val = getattr(task, attr, False)

    for dep in task.deps():
        val = val or getattr(dep, attr, False) or recursive_check(dep, attr)

    return val


def target_remove(target, *args, **kwargs):
    """Remove a given target by calling its 'exists()' and 'remove()' methods."""
    try:
        if target.exists():
            target.remove()
    except AttributeError as e:
        raise AttributeError(
            "The target must have 'exists()' and 'remove()' methods"
        ) from e


def apply_over_luigi_iterable(luigi_iterable, func):
    """Apply the given function to a luigi iterable (task.input() or task.output())."""
    try:
        for key, i in luigi_iterable.items():
            func(i, key)
    except AttributeError:
        for i in luigi.task.flatten(luigi_iterable):
            func(i)


def apply_over_inputs(task, func):
    """Apply the given function to all inputs of a luigi task.

    The given function should accept the following arguments:
    * luigi_iterable: the inputs or outputs of the task
    * key=None: the key when the iterable is a dictionnary
    """
    try:
        inputs = task.input()
    except AttributeError:
        return

    apply_over_luigi_iterable(inputs, func)


def apply_over_outputs(task, func):
    """Apply the given function to all outputs of a luigi task.

    The given function should accept the following arguments:
    * luigi_iterable: the inputs or outputs of the task
    * key=None: the key when the iterable is a dictionnary
    """
    try:
        outputs = task.output()
    except AttributeError:
        return

    apply_over_luigi_iterable(outputs, func)


@luigi.Task.event_handler(luigi.Event.SUCCESS)
def log_targets(task):
    """Hook to log output target of the task."""

    def log_func(target, key=None):
        class_name = task.__class__.__name__
        if key is None:
            L.debug("Output of %s task: %s", class_name, target.path)
        else:
            L.debug("Output %s of %s task: %s", key, class_name, target.path)

    apply_over_outputs(task, log_func)


@luigi.Task.event_handler(luigi.Event.START)
def log_parameters(task):
    """Hook to log actual parameter values considering their global processing."""
    class_name = task.__class__.__name__
    L.debug("Attributes of %s task after global processing:", class_name)
    for name in task.get_param_names():
        try:
            L.debug("Atribute: %s == %s", name, getattr(task, name))
        except Exception:  # pylint: disable=broad-except
            L.debug("Can't print '%s' attribute for unknown reason", name)


class ForceableTask(luigi.Task):
    """A luigi task that can be forced running again by setting the 'rerun' parameter to True."""

    rerun = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if recursive_check(self):
            apply_over_outputs(self, target_remove)


class GlobalParamTask(luigi.Task):
    """Mixin used to add customisable global parameters."""

    def __getattribute__(self, name):
        tmp = super().__getattribute__(name)
        if tmp is not None:
            return tmp
        if hasattr(self, "_global_params"):
            global_param = self._global_params.get(name)
            if global_param is not None:
                return getattr(global_param.cls(), global_param.name)
        return tmp

    def __setattr__(self, name, value):
        try:
            global_params = self._global_params
        except AttributeError:
            global_params = {}
        if value is None and name in global_params:
            L.warning(
                "The Parameter '%s' of the task '%s' is set to None, thus the global "
                "value will be taken frow now on",
                name,
                self.__class__.__name__,
            )
        return super().__setattr__(name, value)


class WorkflowTask(GlobalParamTask, ForceableTask):
    """Default task used in workflows.

    This task can be forced running again by setting the 'rerun' parameter to True.
    It can also use copy and link parameters from other tasks.
    """


class WorkflowWrapperTask(WorkflowTask, luigi.WrapperTask):
    """Base wrapper class with global parameters."""


class ExtParameter(luigi.Parameter):
    """Class to parse file extension parameters."""

    def parse(self, x):
        pattern = re.compile(r"\.?(.*)")
        match = re.match(pattern, x)
        return match.group(1)


class RatioParameter(luigi.NumericalParameter):
    """Class to parse ratio parameters.

    The argument must be a float between 0 and 1.
    The operators to include or exclude the boundaries can be set with 'left_op' and
    'right_op' parameters. Defaults operators include the boundaries.
    """

    def __init__(
        self,
        *args,
        left_op=luigi.parameter.operator.le,
        right_op=luigi.parameter.operator.le,
        **kwargs
    ):
        super().__init__(
            *args,
            min_value=0,
            max_value=1,
            var_type=float,
            left_op=left_op,
            right_op=right_op,
            **kwargs
        )


class OptionalParameter(luigi.OptionalParameter):
    """Mixin to make a parameter class optional."""

    def __init__(self, *args, **kwargs):
        self._cls = self.__class__
        self._base_cls = self.__class__.__bases__[-1]
        if OptionalParameter in (self._cls, self._base_cls):
            raise TypeError(
                "OptionalParameter can only be used as a mixin (must not be the rightmost "
                "class in the class definition)"
            )
        super().__init__(*args, **kwargs)

    def parse(self, x):
        if x and x.lower() != "null":
            return self._base_cls.parse(self, x)
        else:
            return None

    def _warn_on_wrong_param_type(self, param_name, param_value):
        if self.__class__ != self._cls:
            return
        if not isinstance(param_value, int) and param_value is not None:
            warnings.warn(
                '{} "{}" with value "{}" is not of type int or None.'.format(
                    self._cls.__name__, param_name, param_value
                )
            )


class OptionalIntParameter(OptionalParameter, luigi.IntParameter):
    """Class to parse optional int parameters."""


class OptionalNumericalParameter(OptionalParameter, luigi.NumericalParameter):
    """Class to parse optional int parameters."""


class BoolParameter(luigi.BoolParameter):
    """Class to parse boolean parameters and set explicit parsing when default is True."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._default is True:
            self.parsing = self.__class__.EXPLICIT_PARSING


class OutputLocalTarget(luigi.LocalTarget):
    """A target that adds a prefix before the given path.

    If prefix is not given, the current working directory is taken.
    """

    _prefix = None

    def __init__(self, *args, prefix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_prefix(self, prefix)

    @property
    def path(self):
        """The path stored in this target."""
        return str(self.ppath)

    @path.setter
    def path(self, path):
        self._path = Path(path)

    @property
    def ppath(self):
        """The path stored in this target returned as a ``pathlib.Path`` object."""
        if self._prefix is not None:
            return self._prefix / self._path
        else:
            return self._path

    @classmethod
    def set_default_prefix(cls, prefix):
        """Set the default prefix to the class."""
        OutputLocalTarget._reset_prefix(cls, prefix)

    @staticmethod
    def _reset_prefix(obj, prefix):
        # pylint: disable=protected-access
        if prefix is not None:
            obj._prefix = Path(prefix).absolute()
        elif obj._prefix is None:
            obj._prefix = Path(os.getcwd())
        else:
            obj._prefix = Path(obj._prefix)


class ParamLink:
    """Class to store parameter linking informations."""

    def __init__(self, cls, name=None, default=None):
        self.cls = cls
        self.name = name
        self.default = default


class copy_params:
    """Copy a parameter from another Task.

    If no value is given to this parameter, the value from the other task is taken.

    **Usage**:

    .. code-block:: python

        class AnotherTask(luigi.Task):
            m = luigi.IntParameter(default=1)

        @copy_params(m=ParamLink(AnotherTask))
        class MyFirstTask(luigi.Task):
            def run(self):
               print(self.m) # this will be defined and print 1
               # ...

        @copy_params(another_m=ParamLink(AnotherTask, "m"))
        class MySecondTask(luigi.Task):
            def run(self):
               print(self.another_m) # this will be defined and print 1
               # ...

        @copy_params(another_m=ParamLink(AnotherTask, "m", 5))
        class MyFirstTask(luigi.Task):
            def run(self):
               print(self.another_m) # this will be defined and print 5
               # ...

        @copy_params(another_m=ParamLink(AnotherTask, "m"))
        class MyFirstTask(GlobalParamTask):
            def run(self):
               print(self.another_m) # this will be defined and print 1 if self.another_m is None
               # ...
    """

    def __init__(self, **params_to_copy):
        super().__init__()
        if not params_to_copy:
            raise TypeError("params_to_copy cannot be empty")

        self.params_to_copy = params_to_copy

    def __call__(self, task_that_inherits):
        # Get all parameters
        for param_name, attr in self.params_to_copy.items():
            # Check if the parameter exists in the inheriting task
            if not hasattr(task_that_inherits, param_name):
                if attr.name is None:
                    attr.name = param_name
                par = getattr(attr.cls, attr.name)

                # Copy param
                new_param = deepcopy(par)

                # Set default value is required
                if attr.default is not None:
                    new_param._default = attr.default
                elif (
                    issubclass(task_that_inherits, GlobalParamTask)
                    and attr.default is None
                ):
                    new_param._default = None

                # Add it to the inheriting task with new default values
                setattr(task_that_inherits, param_name, new_param)

                # Add link to global parameter
                if issubclass(task_that_inherits, GlobalParamTask):
                    task = task_that_inherits
                    if not hasattr(task, "_global_params"):
                        task._global_params = {}
                    task._global_params[param_name] = attr

        return task_that_inherits


def get_dependency_graph(task):
    """Compute dependency graph of a given task.

    Args:
        task (luigi.Task): the task from which the dependency graph is computed.

    Returns:
        list(luigi.task): A list of (parent, child) tuples
    """
    childs = []
    for t in task.deps():
        childs.append((task, t))
        childs.extend(get_dependency_graph(t))
    return childs
