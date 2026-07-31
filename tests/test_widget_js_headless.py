"""Headless JS tests for the vendored phase-plane widget (no browser needed)."""

import shutil
import subprocess
import tempfile

import pytest

from ntmf.phase_plane_widget import PhasePlaneWidget


def _get_js_source():
    return PhasePlaneWidget._esm


# ── Python-side string sanity checks ──────────────────────────────

class TestJSContainsExpectedPatches:
    @pytest.fixture(scope="module")
    def js_src(self):
        return _get_js_source()

    def test_contains_python_compute_guard(self, js_src):
        assert "python_compute" in js_src

    def test_contains_if_python_compute_branch(self, js_src):
        assert "if (pythonCompute)" in js_src or "if (pythonCompute === true)" in js_src

    def test_contains_trait_change_listener_loop(self, js_src):
        # The listener registers inside a forEach over data-trait names
        assert "'nullcline_x'" in js_src or '"nullcline_x"' in js_src

    def test_contains_vector_field_in_listener_array(self, js_src):
        assert "'vector_field'" in js_src or '"vector_field"' in js_src

    def test_contains_fixed_points_in_listener_array(self, js_src):
        assert "'fixed_points'" in js_src or '"fixed_points"' in js_src

    def test_contains_trajectory_in_listener_array(self, js_src):
        assert "'trajectory'" in js_src or '"trajectory"' in js_src

    def test_render_phase_plane_still_present(self, js_src):
        assert "function renderPhasePlane()" in js_src

    def test_render_time_series_still_present(self, js_src):
        assert "function renderTimeSeries()" in js_src


# ── Node.js syntax check ──────────────────────────────────────────

NODE_AVAILABLE = shutil.which("node") is not None


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not installed")
class TestJSParsesInNode:
    def test_widget_js_parses_without_syntax_error(self):
        """The vendored JS should at least evaluate without throwing a syntax error.

        The source is an ES module (uses ``export``), so we write it to a
        ``.mjs`` file and let Node.js parse it as a module.
        """
        js_src = _get_js_source()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", delete=False) as f:
            f.write(js_src)
            tmpfile = f.name

        try:
            result = subprocess.run(
                ["node", "--check", tmpfile],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"Node.js syntax check failed:\nSTDERR: {result.stderr}\n"
                f"STDOUT: {result.stdout}"
            )
        finally:
            import os

            os.unlink(tmpfile)
