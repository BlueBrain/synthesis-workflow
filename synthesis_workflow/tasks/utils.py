"""utils functions for luigi tasks."""
import warnings

import luigi

from .config import circuitconfigs
from .config import diametrizerconfigs
from .config import logger as L
from .config import pathconfigs
from .config import synthesisconfigs


@luigi.Task.event_handler(luigi.Event.START)
def log_parameters(task):
    class_name = task.__class__.__name__
    L.debug("Attributes of {} class after global processing:".format(class_name))
    for name in task.get_param_names():
        try:
            L.debug("Atribute: {} == {}".format(name, getattr(task, name)))
        except Exception:
            L.debug("Can't print '{}' attribute for unknown reason".format(name))


class BaseTask(luigi.Task):
    """Base class used to add customisable global parameters"""
    _global_configs = [diametrizerconfigs, synthesisconfigs, circuitconfigs, pathconfigs]

    def __getattribute__(self, name):
        tmp = super(BaseTask, self).__getattribute__(name)
        if tmp is not None:
            return tmp
        for conf in self._global_configs:
            tmp_conf = conf()
            if hasattr(tmp_conf, name):
                return getattr(tmp_conf, name)
        return tmp

    def __setattr__(self, name, value):
        if value is None and name in self.get_param_names():
            msg = (
                "The Parameter '{}' of the task '{}' is set to None, thus the global "
                "value will be taken frow now on"
            ).format(name, self.__class__.__name__)
            warnings.warn(msg)
        return super(BaseTask, self).__setattr__(name, value)


class BaseWrapperTask(BaseTask, luigi.WrapperTask):
    pass
