"""Tests for ntmf.validation — MF-vs-NN validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import interp1d

from ntmf.config import adding_K_params, get_params_model_SI
from ntmf.meanfield import simulate_MF_FS_RS_ode
from ntmf.validation import (
    build_driving_input_from_hdf5,
    build_mf_network_config,
    compare_mf_nn,
    compute_nn_population_rates,
)


# ---------------------------------------------------------------------------
# Project-level fixtures (reuse from conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def nn_data(project_root: Path) -> dict:
    """Load the HDF5 spike data from test_0."""
    from ntmf.config import load_spike_data
    path = project_root / "neural_network_simulation" / "simulations" / "test_0" / "spikes_data.h5"
    if not path.exists():
        pytest.skip("NN spike data (test_0) not found")
    return load_spike_data(path)


@pytest.fixture(scope="session")
def mf_network_config(project_root: Path) -> dict:
    """Build network config with Q_e/Q_i for the MF simulation."""
    return build_mf_network_config(
        str(project_root / "config" / "network_config_file_v0.json"),
        Q_e_nS=1.5,
        Q_i_nS=5.0,
    )


@pytest.fixture(scope="session")
def tf_params(project_root: Path) -> dict:
    """Load best TF fitting params (index 0) for FS and RS."""
    val_dir = project_root / "transfer_function" / "validation"
    with open(val_dir / "FS_MF_params.json") as f:
        fs_data = json.load(f)
    with open(val_dir / "RS_MF_params.json") as f:
        rs_data = json.load(f)
    best_fs = fs_data[0]  # lowest mean_error
    best_rs = rs_data[0]
    return {
        "alphas": {"FS": best_fs["alpha"], "RS": best_rs["alpha"]},
        "poly_params": {
            "FS": best_fs["polynomial_params"],
            "RS": best_rs["polynomial_params"],
        },
    }


@pytest.fixture(scope="session")
def neuron_params(project_root: Path, mf_network_config: dict) -> dict:
    """Load SI neuron params and add K_e/K_i."""
    model_dir = project_root / "neuron_models" / "AdEx"
    params_FS = get_params_model_SI("FS", model_dir / "FS.json")
    params_RS = get_params_model_SI("RS", model_dir / "RS.json")
    params_FS = adding_K_params(params_FS.copy(), mf_network_config)
    params_RS = adding_K_params(params_RS.copy(), mf_network_config)
    return {"FS": params_FS, "RS": params_RS}


@pytest.fixture(scope="session")
def mf_simulation(
    nn_data: dict,
    mf_network_config: dict,
    tf_params: dict,
    neuron_params: dict,
) -> tuple[dict, np.ndarray]:
    """Run MF ODE simulation matching the NN protocol.

    Returns (mf_rates_dict, time_array).
    """
    sim_duration = nn_data["sim_duration"]
    dt = 1e-3  # 1 ms time step
    time = np.arange(0, sim_duration, dt)

    driving_input = build_driving_input_from_hdf5(
        # Use the path inferred from nn_data fixture
        hdf5_path=str(
            Path(__file__).resolve().parent.parent
            / "neural_network_simulation" / "simulations" / "test_0"
            / "spikes_data.h5"
        ),
        time=time,
        network_config=mf_network_config,
    )

    mf_rates = simulate_MF_FS_RS_ode(
        time=time,
        neuron_models=["FS", "RS"],
        params=neuron_params,
        poly_params=tf_params["poly_params"],
        alphas=tf_params["alphas"],
        network_config=mf_network_config,
        driving_input=driving_input,
        tau_f=0.01,
    )
    return mf_rates, time


@pytest.fixture(scope="session")
def nn_rates(nn_data: dict) -> dict:
    """Compute binned NN population rates."""
    N_FS = int(nn_data["network_composition"]["num_FS_neurons"])
    N_RS = int(nn_data["network_composition"]["num_RS_neurons"])
    return compute_nn_population_rates(
        nn_data,
        N_FS=N_FS,
        N_RS=N_RS,
        sim_duration=nn_data["sim_duration"],
        bin_size=0.1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDrivingInput:
    """Tests for build_driving_input_from_hdf5."""

    def test_driving_input_reconstruction(self, project_root: Path):
        """Verify driving input has correct step timings."""
        hdf5_path = str(
            project_root / "neural_network_simulation" / "simulations" / "test_0"
            / "spikes_data.h5"
        )
        dt = 1e-3
        time = np.arange(0, 10.0, dt)

        di = build_driving_input_from_hdf5(hdf5_path, time)

        # Check shapes
        assert di["excitatory"]["FS"].shape == time.shape
        assert di["excitatory"]["RS"].shape == time.shape
        assert di["inhibitory"]["FS"].shape == time.shape
        assert di["inhibitory"]["RS"].shape == time.shape

        # Background exc should be 0.3 Hz everywhere
        np.testing.assert_allclose(di["excitatory"]["FS"][0], 0.3, atol=1e-10)

        # During [1,2) FS exc should be 5.0 + 0.3 = 5.3
        mask_1 = (time >= 1.0) & (time < 2.0)
        np.testing.assert_allclose(di["excitatory"]["FS"][mask_1], 5.3, atol=1e-10)

        # During [0,1) FS exc should still be 0.3 (background only)
        mask_bg = (time >= 0.5) & (time < 1.0)
        np.testing.assert_allclose(di["excitatory"]["FS"][mask_bg], 0.3, atol=1e-10)

        # RS exc during [3,4) should be 3.0 + 0.3 = 3.3
        mask_rs = (time >= 3.0) & (time < 4.0)
        np.testing.assert_allclose(di["excitatory"]["RS"][mask_rs], 3.3, atol=1e-10)

        # Inh FS during [5,6) should be 10.0
        mask_ihfs = (time >= 5.0) & (time < 6.0)
        np.testing.assert_allclose(di["inhibitory"]["FS"][mask_ihfs], 10.0, atol=1e-10)

        # Inh RS during [7,8) should be 30.0
        mask_ihrs = (time >= 7.0) & (time < 8.0)
        np.testing.assert_allclose(di["inhibitory"]["RS"][mask_ihrs], 30.0, atol=1e-10)

    def test_inhibitory_zero_outside_stimulus(self, project_root: Path):
        """Inhibitory input should be zero outside the stimulus windows."""
        hdf5_path = str(
            project_root / "neural_network_simulation" / "simulations" / "test_0"
            / "spikes_data.h5"
        )
        time = np.arange(0, 10.0, 1e-3)
        di = build_driving_input_from_hdf5(hdf5_path, time)

        # Inh FS should be 0 outside [5,6)
        mask_before = (time >= 0.0) & (time < 4.9)
        assert np.all(di["inhibitory"]["FS"][mask_before] == 0.0)
        mask_after = (time >= 6.1) & (time < 10.0)
        assert np.all(di["inhibitory"]["FS"][mask_after] == 0.0)


class TestNNRateExtraction:
    """Tests for compute_nn_population_rates."""

    def test_nn_rate_extraction(self, nn_rates: dict):
        """Binned NN rates should be non-negative and finite."""
        for pop in ("FS", "RS"):
            centers, rates = nn_rates[pop]
            assert np.all(np.isfinite(rates)), f"{pop} rates have non-finite values"
            assert np.all(rates >= 0), f"{pop} rates have negative values"

    def test_nn_rate_bins(self, nn_rates: dict, nn_data: dict):
        """Bin centers should span [bin_size/2, sim_duration - bin_size/2]."""
        bin_size = 0.1
        sim_duration = nn_data["sim_duration"]
        for pop in ("FS", "RS"):
            centers, _ = nn_rates[pop]
            assert len(centers) == int(sim_duration / bin_size)
            np.testing.assert_allclose(centers[0], bin_size / 2, atol=1e-10)
            np.testing.assert_allclose(centers[-1], sim_duration - bin_size / 2, atol=1e-10)

    def test_nn_total_spikes_consistent(self, nn_data: dict, nn_rates: dict):
        """Sum of binned spike counts should match total spike count."""
        bin_size = 0.1
        N_FS = int(nn_data["network_composition"]["num_FS_neurons"])
        N_RS = int(nn_data["network_composition"]["num_RS_neurons"])

        for pop, N in [("FS", N_FS), ("RS", N_RS)]:
            _, rates = nn_rates[pop]
            # Total spikes from bins = sum(rates * N * bin_size)
            binned_total = np.sum(rates * N * bin_size)
            actual_total = len(nn_data["spikes"][pop]["t"])
            # Allow small floating-point tolerance
            np.testing.assert_allclose(binned_total, actual_total, rtol=1e-10)


class TestMFSimulation:
    """Tests for MF simulation output."""

    def test_mf_nn_rates_finite(self, mf_simulation: tuple):
        """All MF rates are finite and effectively non-negative.

        Tiny negative values (~1e-11 Hz) can arise from ODE solver
        numerical noise; we clamp at a generous tolerance.
        """
        mf_rates, _ = mf_simulation
        for pop in ("FS", "RS"):
            assert np.all(np.isfinite(mf_rates[pop])), f"MF {pop} rates not finite"
            assert np.all(mf_rates[pop] >= -1e-6), (
                f"MF {pop} rates have significant negative values "
                f"(min={np.min(mf_rates[pop]):.2e})"
            )


class TestValidation:
    """Tests comparing MF simulation against NN data."""

    def test_mf_nn_steady_state_background(self, mf_simulation: tuple, nn_rates: dict):
        """During background-only period [0,1) s, MF and NN rates within factor of 3."""
        mf_rates, mf_time = mf_simulation

        # NN: average rate in [0.05, 0.95) seconds (avoid edge effects)
        nn_bg_fs = nn_rates["FS"][1][(nn_rates["FS"][0] >= 0.05) & (nn_rates["FS"][0] < 0.95)]
        nn_bg_rs = nn_rates["RS"][1][(nn_rates["RS"][0] >= 0.05) & (nn_rates["RS"][0] < 0.95)]
        mean_nn_fs = float(np.mean(nn_bg_fs)) if len(nn_bg_fs) > 0 else 0.0
        mean_nn_rs = float(np.mean(nn_bg_rs)) if len(nn_bg_rs) > 0 else 0.0

        # MF: average rate in [0.2, 0.95) seconds (give MF time to settle)
        mask_mf_bg = (mf_time >= 0.2) & (mf_time < 0.95)
        mean_mf_fs = float(np.mean(mf_rates["FS"][mask_mf_bg]))
        mean_mf_rs = float(np.mean(mf_rates["RS"][mask_mf_bg]))

        # Rates must be within a factor of 3 (generous criterion for stochastic NN)
        for pop, mf_val, nn_val in [
            ("FS", mean_mf_fs, mean_nn_fs),
            ("RS", mean_mf_rs, mean_nn_rs),
        ]:
            if nn_val > 0.1:  # Only check if NN has appreciable rate
                ratio = max(mf_val, nn_val) / max(min(mf_val, nn_val), 1e-12)
                assert ratio < 3.0, (
                    f"{pop} background rate ratio {ratio:.2f} exceeds 3x "
                    f"(MF={mf_val:.3f} Hz, NN={nn_val:.3f} Hz)"
                )

    def test_mf_nn_correlation(self, mf_simulation: tuple, nn_rates: dict):
        """Pearson r > 0.5 between MF and NN binned rates."""
        mf_rates, mf_time = mf_simulation

        stats = compare_mf_nn(mf_rates, mf_time, nn_rates)

        for pop in ("FS", "RS"):
            r = stats[f"corr_{pop}"]
            assert r > 0.5, (
                f"{pop} Pearson r={r:.3f} <= 0.5 — MF and NN traces poorly correlated"
            )

    def test_comparison_stats_keys(self, mf_simulation: tuple, nn_rates: dict):
        """compare_mf_nn returns expected keys."""
        mf_rates, mf_time = mf_simulation
        stats = compare_mf_nn(mf_rates, mf_time, nn_rates)
        for key in ("rmse_FS", "rmse_RS", "corr_FS", "corr_RS"):
            assert key in stats
            assert isinstance(stats[key], float)
            assert np.isfinite(stats[key])

    def test_constant_data_zero_correlation(self):
        """When MF and NN data are constant (std≈0), corr should be 0.0 (line 202)."""
        time = np.arange(0, 1.0, 1e-3)
        mf_rates = {"FS": np.ones_like(time) * 5.0, "RS": np.ones_like(time) * 3.0}
        nn_rates = {
            "FS": (np.arange(0, 1.0, 0.1) + 0.05, np.ones(10) * 5.0),
            "RS": (np.arange(0, 1.0, 0.1) + 0.05, np.ones(10) * 3.0),
        }
        stats = compare_mf_nn(mf_rates, time, nn_rates)
        assert stats["corr_FS"] == 0.0
        assert stats["corr_RS"] == 0.0
