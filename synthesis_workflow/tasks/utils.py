"""utils functions for luigi tasks."""
import logging

import luigi

from .config import CircuitConfig
from .config import DiametrizerConfig
from .config import PathConfig
from .config import RunnerConfig
from .config import SynthesisConfig


L = logging.getLogger(__name__)


class GlobalParamTask(luigi.Task):
    """Base class used to add customisable global parameters"""

    _global_configs = [
        CircuitConfig,
        DiametrizerConfig,
        PathConfig,
        RunnerConfig,
        SynthesisConfig,
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


class BaseWrapperTask(GlobalParamTask, luigi.WrapperTask):
    """Base wrapper class with global parameters"""
