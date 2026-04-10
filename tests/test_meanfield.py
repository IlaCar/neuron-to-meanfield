"""Tests for the mean-field simulation (Euler and ODE)."""

import json
import numpy as np
import pytest

from ntmf.config import adding_K_params, get_network_config, get_params_model_SI
from ntmf.meanfield import simulate_MF_FS_RS, simulate_MF_FS_RS_ode


# =====================================================================
# Fixtures (module-scoped to avoid reloading)
# =====================================================================

@pytest.fixture(scope="module")
def mf_params(validation_dir, neuron_models_dir, config_dir):
    """Bundle all params needed for MF simulation."""
    params_FS = get_params_model_SI("FS", neuron_models_dir / "FS.json")
    params_RS = get_params_model_SI("RS", neuron_models_dir / "RS.json")

    net_cfg = get_network_config(validation_dir / "network_config_file_val.json")
    params_FS = adding_K_params(params_FS, net_cfg)
    params_RS = adding_K_params(params_RS, net_cfg)

    with open(validation_dir / "FS_MF_params.json") as f:
        fs_params = json.load(f)
    with open(validation_dir / "RS_MF_params.json") as f:
        rs_params = json.load(f)

    return {
        "params": {"FS": params_FS, "RS": params_RS},
        "poly_params": {
            "FS": fs_params[0]["polynomial_params"],
            "RS": rs_params[0]["polynomial_params"],
        },
        "alphas": {"FS": fs_params[0]["alpha"], "RS": rs_params[0]["alpha"]},
        "network_config": net_cfg,
    }


def _make_driving(time, exc_FS, exc_RS, inh_FS, inh_RS):
    """Helper to build driving_input dicts."""
    return {
        "excitatory": {
            "FS": np.ones_like(time) * exc_FS,
            "RS": np.ones_like(time) * exc_RS,
        },
        "inhibitory": {
            "FS": np.ones_like(time) * inh_FS,
            "RS": np.ones_like(time) * inh_RS,
        },
    }


# =====================================================================
# Common tests (Euler)
# =====================================================================

class TestMeanField:
    def test_constant_drive_converges(self, mf_params):
        """Constant drive → rates should stabilise (not diverge)."""
        T, dt = 2.0, 1e-3
        time = np.arange(0, T, dt)
        net_cfg = mf_params["network_config"]

        driving = _make_driving(
            time,
            net_cfg["rates"]["freq_exc_FS"],
            net_cfg["rates"]["freq_exc_RS"],
            net_cfg["rates"]["freq_inh_FS"],
            net_cfg["rates"]["freq_inh_RS"],
        )

        rates = simulate_MF_FS_RS(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=net_cfg,
            driving_input=driving,
        )

        assert np.all(np.isfinite(rates["FS"]))
        assert np.all(np.isfinite(rates["RS"]))

    def test_rates_non_negative(self, mf_params):
        T, dt = 1.0, 1e-3
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 2, 2, 2, 2),
        )

        assert np.all(rates["FS"] >= 0)
        assert np.all(rates["RS"] >= 0)

    def test_zero_drive_zero_rates(self, mf_params):
        T, dt = 1.0, 1e-3
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 0, 0, 0, 0),
        )

        assert rates["FS"][-1] < 1.0
        assert rates["RS"][-1] < 1.0

    def test_tau_f_dynamics(self, mf_params):
        tau_f = 0.01
        dt = 1e-3
        T = 0.2
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 2, 2, 2, 2),
            tau_f=tau_f,
        )

        idx_5tau = int(5 * tau_f / dt)
        if idx_5tau < len(rates["FS"]):
            steady = rates["FS"][-1]
            if abs(steady) > 0.1:
                rel_diff = abs(rates["FS"][idx_5tau] - steady) / abs(steady)
                assert rel_diff < 0.02

    def test_output_shape_matches_time(self, mf_params):
        T, dt = 0.5, 1e-3
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 1, 1, 0, 0),
        )

        assert rates["FS"].shape == time.shape
        assert rates["RS"].shape == time.shape


# =====================================================================
# ODE integration tests
# =====================================================================

class TestMeanFieldODE:
    def test_ode_constant_drive_converges(self, mf_params):
        """ODE with constant drive should converge."""
        T, dt = 2.0, 1e-3
        time = np.arange(0, T, dt)
        net_cfg = mf_params["network_config"]

        driving = _make_driving(
            time,
            net_cfg["rates"]["freq_exc_FS"],
            net_cfg["rates"]["freq_exc_RS"],
            net_cfg["rates"]["freq_inh_FS"],
            net_cfg["rates"]["freq_inh_RS"],
        )

        rates = simulate_MF_FS_RS_ode(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=net_cfg,
            driving_input=driving,
        )

        assert np.all(np.isfinite(rates["FS"]))
        assert np.all(np.isfinite(rates["RS"]))
        assert rates["FS"][-1] > 0 or rates["RS"][-1] > 0

    def test_ode_rates_non_negative(self, mf_params):
        T, dt = 1.0, 1e-3
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS_ode(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 2, 2, 2, 2),
        )

        assert np.all(rates["FS"] >= 0)
        assert np.all(rates["RS"] >= 0)

    def test_ode_zero_drive_near_zero(self, mf_params):
        T, dt = 1.0, 1e-3
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS_ode(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 0, 0, 0, 0),
        )

        assert rates["FS"][-1] < 1.0
        assert rates["RS"][-1] < 1.0

    def test_ode_output_shape(self, mf_params):
        T, dt = 0.5, 1e-3
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS_ode(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 1, 1, 0, 0),
        )

        assert rates["FS"].shape == time.shape
        assert rates["RS"].shape == time.shape

    def test_ode_matches_euler_constant_drive(self, mf_params):
        """ODE and Euler should converge to the same steady state."""
        T, dt = 2.0, 1e-3
        time = np.arange(0, T, dt)

        driving = _make_driving(time, 2, 2, 2, 2)

        kw = dict(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=driving,
        )

        rates_euler = simulate_MF_FS_RS(**kw)
        rates_ode = simulate_MF_FS_RS_ode(**kw)

        # Steady-state values should match closely
        for pop in ["FS", "RS"]:
            ss_euler = rates_euler[pop][-100:].mean()
            ss_ode = rates_ode[pop][-100:].mean()
            if abs(ss_euler) > 0.1:
                rel_diff = abs(ss_ode - ss_euler) / abs(ss_euler)
                assert rel_diff < 0.05, (
                    f"{pop}: Euler SS={ss_euler:.4f}, ODE SS={ss_ode:.4f}, rel_diff={rel_diff:.4f}"
                )

    def test_ode_matches_euler_step_input(self, mf_params):
        """ODE and Euler should agree on step-input dynamics."""
        T, dt = 1.0, 1e-3
        time = np.arange(0, T, dt)

        # Step: 0→5 Hz at t=0.3s for exc_FS
        exc_FS = np.zeros_like(time)
        exc_FS[time >= 0.3] = 5.0

        driving = _make_driving(time, 0, 0, 0, 0)
        driving["excitatory"]["FS"] = exc_FS

        kw = dict(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=driving,
        )

        rates_euler = simulate_MF_FS_RS(**kw)
        rates_ode = simulate_MF_FS_RS_ode(**kw)

        # After transient (last 300ms), rates should be close
        for pop in ["FS", "RS"]:
            ss_euler = rates_euler[pop][-300:].mean()
            ss_ode = rates_ode[pop][-300:].mean()
            if abs(ss_euler) > 0.1:
                rel_diff = abs(ss_ode - ss_euler) / abs(ss_euler)
                assert rel_diff < 0.05, (
                    f"{pop}: Euler SS={ss_euler:.4f}, ODE SS={ss_ode:.4f}"
                )

    def test_ode_tau_f_dynamics(self, mf_params):
        """ODE relaxation time constant should match τ_f."""
        tau_f = 0.01
        dt = 1e-3
        T = 0.2
        time = np.arange(0, T, dt)

        rates = simulate_MF_FS_RS_ode(
            time=time, neuron_models=["FS", "RS"],
            params=mf_params["params"], poly_params=mf_params["poly_params"],
            alphas=mf_params["alphas"], network_config=mf_params["network_config"],
            driving_input=_make_driving(time, 2, 2, 2, 2),
            tau_f=tau_f,
        )

        idx_5tau = int(5 * tau_f / dt)
        if idx_5tau < len(rates["FS"]):
            steady = rates["FS"][-1]
            if abs(steady) > 0.1:
                rel_diff = abs(rates["FS"][idx_5tau] - steady) / abs(steady)
                assert rel_diff < 0.02

    def test_ode_failure_raises(self, mf_params):
        """When solve_ivp fails, the RuntimeError path should trigger (line 206)."""
        from unittest.mock import patch
        from scipy.integrate._ivp.ivp import OdeResult

        T, dt = 0.1, 1e-3
        time = np.arange(0, T, dt)

        # Create a failed OdeResult
        failed_result = OdeResult(t=time, y=np.zeros((2, len(time))), success=False,
                                  message="Test failure")

        with patch("ntmf.meanfield.solve_ivp", return_value=failed_result):
            with pytest.raises(RuntimeError, match="ODE integration failed"):
                simulate_MF_FS_RS_ode(
                    time=time, neuron_models=["FS", "RS"],
                    params=mf_params["params"], poly_params=mf_params["poly_params"],
                    alphas=mf_params["alphas"], network_config=mf_params["network_config"],
                    driving_input=_make_driving(time, 2, 2, 2, 2),
                )
