"""Utils luigi tasks."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import luigi
from git import Repo

from .config import SynthesisConfig
from .luigi_tools import copy_params
from .luigi_tools import OutputLocalTarget
from .luigi_tools import ParamLink
from .luigi_tools import WorkflowTask


class GitClone(WorkflowTask):
    """Task to clone a git repository"""

    url = luigi.Parameter()
    dest = luigi.Parameter()

    def run(self):
        """"""
        Repo.clone_from(self.url, self.output().path)

    def output(self):
        return OutputLocalTarget(self.dest)


@copy_params(
    tmd_parameters_path=ParamLink(SynthesisConfig),
    tmd_distributions_path=ParamLink(SynthesisConfig),
)
class GetOfficialConfiguration(WorkflowTask):
    """Task to get official parameters from the git repository"""

    url = luigi.Parameter()
    specie = luigi.ChoiceParameter(choices=["rat", "mouse", "human"])
    version = luigi.OptionalParameter(default=None)

    def run(self):
        """"""
        with TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "tmp_repo"
            # Note: can not be called with yield here because of the TemporaryDirectory
            GitClone(url=self.url, dest=dest).run()
            if self.version is not None:
                r = Repo(dest)
                r.git.checkout(self.version)
            shutil.copy(
                dest / "entities" / "bionames" / self.specie / "tmd_parameters.json",
                self.output()["tmd_parameters"].path,
            )
            shutil.copy(
                dest / "entities" / "bionames" / self.specie / "tmd_distributions.json",
                self.output()["tmd_distributions"].path,
            )

    def output(self):
        return {
            "tmd_parameters": OutputLocalTarget(self.tmd_parameters_path),
            "tmd_distributions": OutputLocalTarget(self.tmd_distributions_path),
        }
