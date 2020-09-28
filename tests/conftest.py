from os import devnull
from subprocess import call

import pytest


@pytest.fixture
def small_O1(tmpdir):
    """Dump a small O1 atlas in folder path"""
    atlas_dir = tmpdir / "small_O1"
    # fmt: off
    with open(devnull, "w") as f:
        call(["brainbuilder", "atlases",
              "-n", "1,2,3,4,5,6",
              "-t", "200,100,100,100,100,200",
              "-d", "100",
              "-o", str(atlas_dir),
              "column",
              "-a", "1000",
              ], stdout=f, stderr=f)
    # fmt: on

    return atlas_dir
