"""Test that the MF phase-plane demo notebook can be executed cell-by-cell."""

import pathlib

import pytest

try:
    import nbformat
    from nbclient import NotebookClient

    HAS_NBCLIENT = True
except ImportError:
    HAS_NBCLIENT = False


@pytest.mark.skipif(not HAS_NBCLIENT, reason="nbformat/nbclient not installed")
class TestNotebookExecution:
    def test_mf_phase_plane_notebook_executes(self):
        repo_root = pathlib.Path(__file__).parent.parent
        nb_path = (
            repo_root / "transfer_function" / "validation" / "MF_phase_plane.ipynb"
        )
        assert nb_path.exists(), f"Notebook not found: {nb_path}"

        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=120, kernel_name="python3")
        client.execute()
