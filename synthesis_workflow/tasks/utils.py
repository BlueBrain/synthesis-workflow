"""utils functions for luigi tasks."""
import logging
import re

import luigi

from .config import circuitconfigs
from .config import diametrizerconfigs
from .config import pathconfigs
from .config import synthesisconfigs


L = logging.getLogger(__name__)


@luigi.Task.event_handler(luigi.Event.SUCCESS)
def log_targets(task):
    """Hook to log output target of the task"""
    class_name = task.__class__.__name__
    try:
        output = task.output()
    except AttributeError:
        return
    try:
        L.debug("Output of %s task: %s", class_name, output.path)
    except AttributeError:
        try:
            for k, i in output.items():
                L.debug("Output %s of %s task: %s", k, class_name, i.path)
        except AttributeError:
            for i in output:
                L.debug("Output of %s task: %s", class_name, i.path)


@luigi.Task.event_handler(luigi.Event.START)
def log_parameters(task):
    """Hook to log actual parameter values considering their global processing"""
    class_name = task.__class__.__name__
    L.debug("Attributes of %s task after global processing:", class_name)
    for name in task.get_param_names():
        try:
            L.debug("Atribute: %s == %s", name, getattr(task, name))
        except Exception:  # pylint: disable=broad-except
            L.debug("Can't print '%s' attribute for unknown reason", name)


class BaseTask(luigi.Task):
    """Base class used to add customisable global parameters"""

    _global_configs = [
        diametrizerconfigs,
        synthesisconfigs,
        circuitconfigs,
        pathconfigs,
    ]

    def __getattribute__(self, name):
        tmp = super().__getattribute__(name)
        if tmp is not None:
            return tmp
        for conf in self._global_configs:
            tmp_conf = conf()
            if hasattr(tmp_conf, name):
                return getattr(tmp_conf, name)
        return tmp

    def __setattr__(self, name, value):
        if value is None and name in self.get_param_names():
            L.warning(
                "The Parameter '%s' of the task '%s' is set to None, thus the global "
                "value will be taken frow now on",
                name,
                self.__class__.__name__,
            )
        return super().__setattr__(name, value)


class BaseWrapperTask(BaseTask, luigi.WrapperTask):
    """Base wrapper class with global parameters"""


class ExtParameter(luigi.Parameter):
    """Class to parse file extension parameters"""

    def parse(self, x):
        pattern = re.compile(r"\.?(.*)")
        match = re.match(pattern, x)
        return match.group(1)
