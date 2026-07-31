"""Tests for the NTMFMeanField phase-plane adapter."""

import json
import time

import numpy as np
import pytest

from ntmf.config import adding_K_params, get_network_config, get_params_model_SI
from ntmf.phase_plane import NTMFMeanField


# ── fixtures ──

@pytest.fixture
def mf_model():
    """Return an NTMFMeanField instance with real saved parameters."""
    network_config = get_network_config("config/network_config_file_v0.json")
    # Add Q_e / Q_i to external_input (required by MFModel)
    network_config["external_input"]["Q_e"] = 1.5
    network_config["external_input"]["Q_i"] = 5.0

    params_SI = {
        "FS": adding_K_params(
            get_params_model_SI("FS", "neuron_models/AdEx/FS.json"),
            network_config,
        ),
        "RS": adding_K_params(
            get_params_model_SI("RS", "neuron_models/AdEx/RS.json"),
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

    return NTMFMeanField(
        params=params_SI,
        poly_params=poly_params,
        alphas=alphas,
        network_config=network_config,
        tau_f=0.01,
    )


@pytest.fixture
def sample_params():
    """Default slider params with non-zero external drive."""
    return {
        "nu_ext_exc_FS": 50.0,
        "nu_ext_exc_RS": 50.0,
        "nu_ext_inh_FS": 10.0,
        "nu_ext_inh_RS": 10.0,
        "tau_f": 0.01,
        "alpha_FS": 1.0,
        "alpha_RS": 1.0,
    }


# ── RHS evaluation ──

class TestRFSEvaluation:

    def test_zero_state_zero_derivative(self, mf_model):
        """At (0,0) with zero external drive, rates should stay near zero."""
        params = mf_model.default_params.copy()
        params["nu_ext_exc_FS"] = 0.0
        params["nu_ext_exc_RS"] = 0.0
        params["nu_ext_inh_FS"] = 0.0
        params["nu_ext_inh_RS"] = 0.0
        dydt = mf_model.f(0, [0.0, 0.0], params)
        assert isinstance(dydt, list)
        assert len(dydt) == 2
        assert abs(dydt[0]) < 1e-6
        assert abs(dydt[1]) < 1e-6

    def test_finite_outputs(self, mf_model, sample_params):
        for nu_fs, nu_rs in [(0, 0), (10, 10), (50, 20), (80, 80)]:
            dydt = mf_model.f(0, [float(nu_fs), float(nu_rs)], sample_params)
            assert all(np.isfinite(v) for v in dydt)
            assert isinstance(dydt, list)
            assert len(dydt) == 2


class TestCaching:

    def test_caching_avoids_redundant_rebuilds(self, mf_model, sample_params):
        """Second call with identical params should reuse cached interpolators."""
        # First call — warm cache
        mf_model.f(0, [10.0, 10.0], sample_params)
        assert mf_model._last_key is not None

        t0 = time.perf_counter()
        for _ in range(100):
            mf_model.f(0, [10.0 + _, 10.0], sample_params)
        t_cached = time.perf_counter() - t0

        old_key = mf_model._last_key

        # Change a param — forces rebuild
        new_params = sample_params.copy()
        new_params["nu_ext_exc_RS"] = 51.0
        mf_model.f(0, [10.0, 10.0], new_params)

        assert mf_model._last_key != old_key
        assert mf_model._last_key[1] == 51.0  # nu_ext_exc_RS changed


# ── BaseModel analysis methods ──

class TestNullclines:

    def test_returns_points(self, mf_model, sample_params):
        nc_x, nc_y = mf_model.compute_nullclines(
            sample_params, [0, 80], [0, 80], n_grid=20
        )
        assert isinstance(nc_x, list)
        assert isinstance(nc_y, list)
        for pt in nc_x:
            assert len(pt) == 2
            assert all(isinstance(v, float) for v in pt)
        for pt in nc_y:
            assert len(pt) == 2
            assert all(isinstance(v, float) for v in pt)


class TestFixedPoints:

    STABILITIES = {
        "stable_node", "stable_focus",
        "unstable_node", "unstable_focus",
        "saddle",
    }

    def test_classifies(self, mf_model, sample_params):
        fps = mf_model.find_fixed_points(
            sample_params, [0, 80], [0, 80], n_grid=15
        )
        assert isinstance(fps, list)
        for fp in fps:
            assert len(fp) == 3
            assert isinstance(fp[0], float)
            assert isinstance(fp[1], float)
            assert fp[2] in self.STABILITIES


class TestVectorField:

    def test_returns_arrows(self, mf_model, sample_params):
        vf = mf_model.compute_vector_field(
            sample_params, [0, 80], [0, 80], n_grid=12
        )
        assert isinstance(vf, list)
        for arrow in vf:
            assert len(arrow) == 4
            assert all(isinstance(v, (int, float)) for v in arrow)


class TestTrajectory:

    def test_non_empty(self, mf_model, sample_params):
        traj = mf_model.compute_trajectory(
            [0.0, 0.0], sample_params, [0, 50.0], dt=0.01
        )
        assert isinstance(traj, list)
        assert len(traj) > 0
        for row in traj:
            assert len(row) == 3  # [t, nu_FS, nu_RS]
            assert all(np.isfinite(v) for v in row)


class TestJacobian:

    def test_finite_2x2(self, mf_model, sample_params):
        J = mf_model.jacobian([10.0, 10.0], sample_params)
        assert J.shape == (2, 2)
        assert np.all(np.isfinite(J))
