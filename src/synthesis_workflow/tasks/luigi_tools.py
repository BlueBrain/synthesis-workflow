"""Utils functions for the luigi library."""
# pylint: disable=unused-import
import logging

import luigi

from luigi_tools import tasks
from luigi_tools.parameters import BoolParameter
from luigi_tools.parameters import ExtParameter
from luigi_tools.parameters import OptionalFloatParameter
from luigi_tools.parameters import OptionalIntParameter
from luigi_tools.parameters import OptionalNumericalParameter
from luigi_tools.parameters import OptionalParameter
from luigi_tools.parameters import OptionalRatioParameter
from luigi_tools.parameters import RatioParameter
from luigi_tools.targets import OutputLocalTarget
from luigi_tools.tasks import copy_params
from luigi_tools.tasks import ParamRef
from luigi_tools.tasks import WorkflowWrapperTask
from luigi_tools.utils import WorkflowError


L = logging.getLogger(__name__)


class WorkflowTask(tasks.LogTargetMixin, tasks.WorkflowTask):
    """Default task used in workflows.

    This task can be forced running again by setting the 'rerun' parameter to True.
    It can also use copy and link parameters from other tasks.
    """


class OptionalChoiceParameter(OptionalParameter, luigi.ChoiceParameter):
    """Class to parse optional choice parameters."""

    expected_type = str
