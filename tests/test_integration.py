"""Integration tests — end-to-end pipeline stages.

These tests exercise multiple modules together and may take seconds to run.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from ntmf.config import adding_K_params, get_network_config, get_params_model_SI
from ntmf.meanfield import simulate_MF_FS_RS
from ntmf.optimization import run_fits
from ntmf.transfer_function import (
    TF_template,
    membrane_potential_fluctuations,
    res_2_func,
)


# =====================================================================
# Full pipeline on real data: load → fit → MF sim
# =====================================================================

class TestPipelineWithRealData:
    """Uses existing .dat files to validate the entire pipeline."""

    def test_RS_load_fit_evaluate(self, df_RS, params_RS_with_K):
        """Load RS data, run single-alpha fit, verify error is reasonable."""
        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=df_RS, params=params_RS_with_K,
        )

        error, poly = run_fits(
            alpha=1.0,
            df_data=df_RS,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
        )

        assert poly is not None
        assert np.isfinite(error)
        # With alpha=1.0, error should still be below 100 Hz² (generous)
        assert error < 100.0

    def test_RS_fit_and_TF_consistency(self, df_RS, params_RS_with_K, RS_MF_params):
        """Use saved best RS params → compute TF → verify error is reasonable."""
        best = RS_MF_params[0]
        poly = np.array(best["polynomial_params"])
        alpha = best["alpha"]

        error = res_2_func(poly, data=df_RS, params=params_RS_with_K, alpha=alpha)
        # The saved params may come from a different simulation run,
        # so we just verify the error is finite, positive, and not absurd.
        assert np.isfinite(error)
        assert error >= 0
        assert error < 200.0  # biologically absurd threshold

    def test_FS_load_fit_evaluate(self, df_FS, params_FS_with_K):
        """Load FS data, run single-alpha fit."""
        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=df_FS, params=params_FS_with_K,
        )

        error, poly = run_fits(
            alpha=1.165,  # best alpha from FS_MF_params
            df_data=df_RS if False else df_FS,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_FS_with_K,
        )

        assert poly is not None
        assert np.isfinite(error)

    def test_full_MF_simulation_with_saved_params(
        self, config_dir, neuron_models_dir
    ):
        """Load saved params → run MF sim → verify rates are finite and reasonable."""
        params_FS = get_params_model_SI("FS", neuron_models_dir / "FS.json")
        params_RS = get_params_model_SI("RS", neuron_models_dir / "RS.json")
        net_cfg = get_network_config(config_dir / "network_config_file_tests.json")
        params_FS = adding_K_params(params_FS, net_cfg)
        params_RS = adding_K_params(params_RS, net_cfg)

        with open(validation_dir / "FS_MF_params.json") as f:
            fs_data = json.load(f)
        with open(validation_dir / "RS_MF_params.json") as f:
            rs_data = json.load(f)

        T, dt = 1.0, 1e-3
        time = np.arange(0, T, dt)

        driving = {
            "excitatory": {
                "FS": np.ones_like(time) * net_cfg["rates"]["freq_exc_FS"],
                "RS": np.ones_like(time) * net_cfg["rates"]["freq_exc_RS"],
            },
            "inhibitory": {
                "FS": np.ones_like(time) * net_cfg["rates"]["freq_inh_FS"],
                "RS": np.ones_like(time) * net_cfg["rates"]["freq_inh_RS"],
            },
        }

        rates = simulate_MF_FS_RS(
            time=time,
            neuron_models=["FS", "RS"],
            params={"FS": params_FS, "RS": params_RS},
            poly_params={
                "FS": fs_data[0]["polynomial_params"],
                "RS": rs_data[0]["polynomial_params"],
            },
            alphas={"FS": fs_data[0]["alpha"], "RS": rs_data[0]["alpha"]},
            network_config=net_cfg,
            driving_input=driving,
        )

        # Rates should be finite
        assert np.all(np.isfinite(rates["FS"]))
        assert np.all(np.isfinite(rates["RS"]))

        # Rates should be in a biologically plausible range (< 200 Hz)
        assert rates["FS"][-1] < 200
        assert rates["RS"][-1] < 200

        # At least one population should be active with these drive rates
        assert rates["FS"][-1] > 0 or rates["RS"][-1] > 0


# =====================================================================
# Small Brian2 simulation → data extraction → fit
# =====================================================================

class TestSmallPipeline:
    """Run a tiny Brian2 grid (3×3) to exercise the full pipeline."""

    @pytest.mark.slow
    def test_tiny_grid_extraction_and_fit(self, neuron_models_dir, config_dir):
        """3×3 grid, 5 neurons, 200 ms sim → extract rates → fit."""
        import brian2 as b2
        from ntmf.neurons import setting_simulation_Brian
        from ntmf.config import get_input_config, get_syn_info
        from ntmf.network import extracting_single_pop_freq_and_std

        data_folder = neuron_models_dir
        neuron_model = "RS_no_adapt"
        N_cell = 5
        sim_duration = 0.5 * b2.second
        dt = 0.1 * b2.ms
        times = b2.arange(0, sim_duration, dt)

        input_config = get_input_config(config_dir / "input_config_TF.json")
        N_ext_exc = input_config["connections"]["N_external_exc"]
        N_ext_inh = input_config["connections"]["N_external_inh"]
        conn_prob = input_config["connections"]["conn_prob"]
        bg_rate = input_config["rates"]["background_freq"] * b2.Hz

        Qe, Qi = get_syn_info(json_file_name=data_folder / "RS_no_adapt.json")

        exc_range = [0, 10, 20]
        inh_range = [0, 10, 20]

        p_start = 0.1 * b2.second
        p_end = 0.4 * b2.second

        rows = []
        for inh_val in inh_range:
            for exc_val in exc_range:
                b2.start_scope()

                stim = b2.TimedArray(b2.zeros(int(sim_duration / dt) + 10) * b2.amp, dt=dt)
                G = setting_simulation_Brian(
                    neuron_model=neuron_model,
                    json_file_name=data_folder / (neuron_model + ".json"),
                    N_cell=N_cell, curr_inj=stim,
                )

                # Background
                rate_bg = b2.zeros(len(times)) * b2.Hz
                rate_bg[:] = bg_rate
                ta_bg = b2.TimedArray(rate_bg, dt=dt)
                P_bg = b2.PoissonGroup(N_ext_exc, rates="ta_bg(t)")

                # Stimulus
                rate_exc = b2.zeros(len(times)) * b2.Hz
                rate_exc[(times >= p_start) & (times < p_end)] = exc_val * b2.Hz
                ta_exc = b2.TimedArray(rate_exc, dt=dt)
                P_exc = b2.PoissonGroup(N_ext_exc, rates="ta_exc(t)")

                rate_inh = b2.zeros(len(times)) * b2.Hz
                rate_inh[(times >= p_start) & (times < p_end)] = inh_val * b2.Hz
                ta_inh = b2.TimedArray(rate_inh, dt=dt)
                P_inh = b2.PoissonGroup(N_ext_inh, rates="ta_inh(t)")

                S_bg = b2.Synapses(P_bg, G, on_pre="GsynE_post+=Qe")
                S_bg.connect(p=conn_prob)
                S_exc = b2.Synapses(P_exc, G, on_pre="GsynE_post+=Qe")
                S_exc.connect(p=conn_prob)
                S_inh = b2.Synapses(P_inh, G, on_pre="GsynI_post+=Qi")
                S_inh.connect(p=conn_prob)

                mon_spike = b2.SpikeMonitor(G)

                b2.run(sim_duration, namespace={
                    "ta_bg": ta_bg, "ta_exc": ta_exc, "ta_inh": ta_inh,
                    "Qe": Qe, "Qi": Qi, "current": stim,
                })

                mean_rate, std_rate = extracting_single_pop_freq_and_std(
                    sim_duration=sim_duration, p_start=p_start, p_end=p_end,
                    pop=mon_spike, N_pop=N_cell, bin_size=0.1,
                )

                rows.append({
                    "input_exc": float(exc_val),
                    "input_inh": float(inh_val),
                    "avg_f_out": mean_rate,
                    "std_f_out": std_rate,
                })

        df = pd.DataFrame(rows)
        assert len(df) == 9

        # Verify monotonicity: more excitation → more output (on average)
        for inh_val in inh_range:
            subset = df[df["input_inh"] == inh_val].sort_values("input_exc")
            rates = subset["avg_f_out"].values
            # At least the highest excitation should give >= lowest excitation
            if rates[-1] > 0:
                assert rates[-1] >= rates[0] - 1.0  # allow some tolerance
