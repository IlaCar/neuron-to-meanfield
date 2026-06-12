"""Tests for PhasePlaneWidget in python_compute mode (no JS/browser needed)."""

import json

import pytest

from ntmf.config import adding_K_params, get_params_model_SI
from ntmf.phase_plane import NTMFMeanField
from ntmf.phase_plane_widget import PhasePlaneWidget
from ntmf.validation import build_mf_network_config


def _build_widget():
    """Load real data and build a PhasePlaneWidget in python_compute mode."""
    network_config = build_mf_network_config(
        "config/network_config_file_v0.json",
        Q_e_nS=1.5,
        Q_i_nS=5.0,
    )

    params_SI = {
        "FS": adding_K_params(
            get_params_model_SI("FS", "neuron_models/AdEx/FS.json").copy(),
            network_config,
        ),
        "RS": adding_K_params(
            get_params_model_SI("RS", "neuron_models/AdEx/RS.json").copy(),
            network_config,
        ),
    }

    with open("transfer_function/validation/FS_MF_params.json") as f:
        fs_fits = json.load(f)
    with open("transfer_function/validation/RS_MF_params.json") as f:
        rs_fits = json.load(f)

    fs_fit = fs_fits[0]
    rs_fit = rs_fits[0]

    poly_params = {
        "FS": fs_fit["polynomial_params"],
        "RS": rs_fit["polynomial_params"],
    }
    alphas = {"FS": fs_fit["alpha"], "RS": rs_fit["alpha"]}

    ntmf = NTMFMeanField(
        params=params_SI,
        poly_params=poly_params,
        alphas=alphas,
        network_config=network_config,
        tau_f=0.01,
    )

    return PhasePlaneWidget(
        model=ntmf,
        python_compute=True,
        xlim=[0, 80],
        ylim=[0, 80],
        t_max=100.0,
    )


@pytest.fixture(scope="module")
def wc_widget():
    return _build_widget()


class TestWidgetInstantiation:
    def test_creates_without_error(self):
        w = _build_widget()
        assert w.python_compute is True
        assert w._model_instance is not None

    def test_default_params_on_sliders(self):
        w = _build_widget()
        assert "nu_ext_exc_FS" in w.params
        assert "tau_f" in w.params


class TestPythonComputePopulatesTraits:
    def test_nullclines_populated(self, wc_widget):
        assert len(wc_widget.nullcline_x) > 0
        assert len(wc_widget.nullcline_y) > 0

    def test_vector_field_populated(self, wc_widget):
        assert len(wc_widget.vector_field) > 0

    def test_fixed_points_populated(self, wc_widget):
        assert len(wc_widget.fixed_points) > 0
        for fp in wc_widget.fixed_points:
            assert len(fp) == 3

    def test_trajectory_populated(self, wc_widget):
        assert len(wc_widget.trajectory) > 0
        # trajectory stores [t, nu_FS, nu_RS]
        assert len(wc_widget.trajectory[0]) == 3


class TestTraitChangeTriggersRecompute:
    def test_param_change_updates_nullclines(self, wc_widget):
        old_nc = [row[:] for row in wc_widget.nullcline_y]
        # Change external input rate for RS (moves the RS nullcline, nc_y)
        new_params = dict(wc_widget.params)
        new_params["nu_ext_exc_RS"] = 50.0
        wc_widget.params = new_params
        # Observer fires synchronously; trait should be updated
        assert wc_widget.nullcline_y != old_nc


class TestWidgetWithoutPythonCompute:
    def test_default_mode_empty_traits(self):
        w = PhasePlaneWidget(model_name="wilson_cowan")
        # Default python_compute=False; JS hasn't run so data traits are empty
        assert len(w.nullcline_x) == 0
        assert len(w.nullcline_y) == 0
        assert len(w.vector_field) == 0
        assert len(w.fixed_points) == 0
        assert len(w.trajectory) == 0
