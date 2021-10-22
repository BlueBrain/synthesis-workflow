"""Utils functions for the luigi library."""
# pylint: disable=unused-import
import logging

import luigi
from luigi_tools import task
from luigi_tools.parameter import BoolParameter
from luigi_tools.parameter import ExtParameter
from luigi_tools.parameter import OptionalFloatParameter
from luigi_tools.parameter import OptionalIntParameter
from luigi_tools.parameter import OptionalNumericalParameter
from luigi_tools.parameter import OptionalParameter
from luigi_tools.parameter import OptionalPathParameter
from luigi_tools.parameter import OptionalRatioParameter
from luigi_tools.parameter import PathParameter
from luigi_tools.parameter import RatioParameter
from luigi_tools.target import OutputLocalTarget
from luigi_tools.task import ParamRef
from luigi_tools.task import WorkflowWrapperTask
from luigi_tools.task import copy_params
from luigi_tools.util import WorkflowError

L = logging.getLogger(__name__)


class WorkflowTask(task.LogTargetMixin, task.WorkflowTask):
    """Default :class:`luigi.Task` used in workflows.

    This task can be forced running again by setting the 'rerun' parameter to True.
    It can also use copy and link parameters from other tasks.
    """


class OptionalChoiceParameter(OptionalParameter, luigi.ChoiceParameter):
    """Class to parse optional choice parameters."""

    expected_type = str
