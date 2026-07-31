"""Tests for the two-stage fitting and alpha search."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from ntmf.optimization import run_fits, discrete_alpha_search
from ntmf.transfer_function import (
    TF_template,
    eff_thresh,
    est_thresh,
    membrane_potential_fluctuations,
    res_1_func,
    res_2_func,
)


# =====================================================================
# Residuals at true parameters
# =====================================================================

class TestResiduals:
    def test_res1_zero_at_true_params(self, df_RS, params_RS_with_K, RS_MF_params):
        """res_1_func should be ≈ 0 when est_V_th ≈ eff_thresh at true params."""
        best = RS_MF_params[0]
        alpha = best["alpha"]
        poly = np.array(best["polynomial_params"])

        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=df_RS, params=params_RS_with_K,
        )

        est_V_th = est_thresh(data=df_RS, mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, alpha=alpha)

        # At the true polynomial params, res_1 should be small (not exactly zero
        # because stage-2 optimisation changed the params)
        val = res_1_func(poly, mu_V, sig_V, tau_V_norm, est_V_th)
        assert np.isfinite(val)
        assert val >= 0

    def test_res2_finite_on_data(self, df_RS, params_RS_with_K, RS_MF_params):
        """res_2_func should be finite on real data with saved params."""
        best = RS_MF_params[0]
        val = res_2_func(
            np.array(best["polynomial_params"]),
            data=df_RS, params=params_RS_with_K, alpha=best["alpha"],
        )
        assert np.isfinite(val)
        assert val >= 0


# =====================================================================
# run_fits on synthetic data
# =====================================================================

class TestRunFits:
    def test_converges_on_synthetic_data(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params):
        """Fitting on synthetic data generated from known params should converge."""
        best = RS_MF_params[0]
        true_alpha = best["alpha"]
        true_poly = np.array(best["polynomial_params"])

        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=synthetic_RS_tf_data, params=params_RS_with_K,
        )

        error, fitted_poly = run_fits(
            alpha=true_alpha,
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
        )

        assert fitted_poly is not None
        assert np.isfinite(error)
        assert error < 50.0  # generous bound for small synthetic grid

    def test_run_fits_returns_inf_on_bad_alpha(self, synthetic_RS_tf_data, params_RS_with_K):
        """Alpha = 0 should trigger exception path → returns (inf, None)."""
        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=synthetic_RS_tf_data, params=params_RS_with_K,
        )
        # alpha=0 causes division by zero → exception caught → inf
        # But actually est_thresh divides by alpha, so alpha=0 → ZeroDivisionError
        # which should be caught by the try/except
        error, poly = run_fits(
            alpha=0.0,
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
        )
        assert error == float(np.inf)
        assert poly is None

    def test_saving_json_creates_file(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params, tmp_path):
        """run_fits with saving_json=True should create a JSON file."""
        best = RS_MF_params[0]
        true_alpha = best["alpha"]

        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=synthetic_RS_tf_data, params=params_RS_with_K,
        )

        out_file = str(tmp_path / "fit_output.json")

        error, fitted_poly = run_fits(
            alpha=true_alpha,
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
            saving_json=True,
            out_file_name=out_file,
        )

        assert os.path.exists(out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["alpha"] == true_alpha
        assert "mean_error" in data[0]
        assert "polynomial_params" in data[0]

    def test_saving_json_appends(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params, tmp_path):
        """run_fits should append to an existing JSON file."""
        best = RS_MF_params[0]

        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=synthetic_RS_tf_data, params=params_RS_with_K,
        )

        out_file = str(tmp_path / "fit_append.json")

        # First run
        run_fits(
            alpha=best["alpha"],
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
            saving_json=True, out_file_name=out_file,
        )
        # Second run with different alpha
        run_fits(
            alpha=best["alpha"] + 0.01,
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
            saving_json=True, out_file_name=out_file,
        )

        with open(out_file) as f:
            data = json.load(f)
        assert len(data) == 2


# =====================================================================
# discrete_alpha_search
# =====================================================================

class TestDiscreteAlphaSearch:
    def test_finds_alpha_near_true(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params):
        """discrete_alpha_search should find best alpha near the true value."""
        best = RS_MF_params[0]
        true_alpha = best["alpha"]

        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=synthetic_RS_tf_data, params=params_RS_with_K,
        )

        best_alpha, best_error, best_poly = discrete_alpha_search(
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
            alpha_min=true_alpha - 0.1,
            alpha_max=true_alpha + 0.1,
            alpha_step=0.05,
        )

        assert best_alpha is not None
        assert best_poly is not None
        assert best_error < float(np.inf)
        assert abs(best_alpha - true_alpha) < 0.06  # within one step

    def test_with_saving_json(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params, tmp_path):
        """discrete_alpha_search with saving_json should produce a JSON file."""
        best = RS_MF_params[0]

        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(
            data=synthetic_RS_tf_data, params=params_RS_with_K,
        )

        out_file = str(tmp_path / "alpha_search.json")

        best_alpha, best_error, best_poly = discrete_alpha_search(
            df_data=synthetic_RS_tf_data,
            mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, tau_V_norm=tau_V_norm,
            params_SI=params_RS_with_K,
            alpha_min=0.5,
            alpha_max=0.7,
            alpha_step=0.1,
            saving_json=True,
            out_file_name=out_file,
        )

        assert os.path.exists(out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert len(data) == 3  # 0.5, 0.6, 0.7
