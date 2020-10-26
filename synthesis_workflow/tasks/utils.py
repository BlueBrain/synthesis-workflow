"""Utils luigi tasks."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import luigi
from git import Repo

from .luigi_tools import WorkflowTask
from .luigi_tools import OutputLocalTarget


class GitClone(WorkflowTask):
    """Task to clone a git repository"""

    url = luigi.Parameter()
    dest = luigi.Parameter()

    def run(self):
        """"""
        Repo.clone_from(self.url, self.output().path)

    def output(self):
        return OutputLocalTarget(self.dest)


class GetSynthesisInputs(WorkflowTask):
    """Task to get synthesis input files from a folder on git it repository.

    If no url is provided, this task will copy an existing folder to the target location.

    Args:
        url (str): url of repository, if None, git_synthesis_input_path should be an existing folder
        version (str): version of repo to checkout (optional)
        git_synthesis_input_path (str): path to folder in git repo with synthesis files
        local_synthesis_input_path (str): path to local folder to copy these files
    """

    url = luigi.Parameter(default=None)
    version = luigi.OptionalParameter(default=None)
    git_synthesis_input_path = luigi.Parameter(default="synthesis_input")
    local_synthesis_input_path = luigi.Parameter(default="synthesis_input")

    def run(self):
        """"""
        if self.url is None:
            shutil.copytree(self.git_synthesis_input_path, self.output().path)
        else:
            with TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "tmp_repo"
                # Note: can not be called with yield here because of the TemporaryDirectory
                GitClone(url=self.url, dest=dest).run()
                if self.version is not None:
                    r = Repo(dest)
                    r.git.checkout(self.version)
                shutil.copytree(
                    dest / self.git_synthesis_input_path, self.output().path
                )

    def output(self):
        return luigi.LocalTarget(self.local_synthesis_input_path)
