"""Utils luigi tasks."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import luigi
from git import Repo

from synthesis_workflow.tasks.config import PathConfig
from synthesis_workflow.tasks.luigi_tools import OutputLocalTarget
from synthesis_workflow.tasks.luigi_tools import WorkflowTask


class GitClone(WorkflowTask):
    """Task to clone a git repository."""

    url = luigi.Parameter(
        description=(
            ":str: Url of repository. If None, git_synthesis_input_path should be an existing "
            "folder."
        )
    )
    dest = luigi.Parameter(description=":str: Path to the destination directory.")

    def run(self):
        """"""
        Repo.clone_from(self.url, self.output().path)

    def output(self):
        """"""
        return OutputLocalTarget(self.dest)


class GetSynthesisInputs(WorkflowTask):
    """Task to get synthesis input files from a folder on git it repository.

    If no url is provided, this task will copy an existing folder to the target location.

    Attributes:
        local_synthesis_input_path (str): Path to local folder to copy these files.
    """

    url = luigi.OptionalParameter(
        default=None,
        description=(
            ":str: Url of repository. If None, git_synthesis_input_path should be an "
            "existing folder."
        ),
    )
    version = luigi.OptionalParameter(
        default=None, description=":str: Version of repo to checkout."
    )
    git_synthesis_input_path = luigi.Parameter(
        default="synthesis_input",
        description=":str: Path to folder in git repo with synthesis files.",
    )

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
        """"""
        # TODO: it would probably be better to have a specific target for each file
        return OutputLocalTarget(PathConfig().local_synthesis_input_path)
