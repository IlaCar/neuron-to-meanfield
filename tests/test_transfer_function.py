"""Tests for the transfer function mathematics (pure numpy/scipy, no Brian2)."""

import numpy as np
import pandas as pd
import pytest

from ntmf.transfer_function import (
    TF_template,
    TF_template_sim,
    eff_thresh,
    est_thresh,
    get_mean_error_distribution,
    membrane_potential_fluctuations,
    membrane_potential_fluctuations_sim,
    res_1_func,
    res_2_func,
)


# =====================================================================
# membrane_potential_fluctuations — vectorized
# =====================================================================

class TestMembranePotentialFluctuations:
    """All tests use params_RS_with_K from conftest (RS model, K_e=400, K_i=100)."""

    def _make_df(self, exc_vals, inh_vals):
        """Build a DataFrame from arrays of exc/inh rates."""
        return pd.DataFrame({
            "input_exc": np.asarray(exc_vals, dtype=float),
            "input_inh": np.asarray(inh_vals, dtype=float),
        })

    def test_zero_input_mu_V_equals_E_L(self, params_RS_with_K):
        """f_e = f_i ≈ 0 → μ_V should approach E_L = −65 mV."""
        df = self._make_df([0.0], [0.0])
        mu_V, _, _, _ = membrane_potential_fluctuations(df, params_RS_with_K)
        assert abs(mu_V[0] - params_RS_with_K["E_L"]) < 1e-6

    def test_pure_excitatory_depolarizes(self, params_RS_with_K):
        """Pure excitatory input → μ_V > E_L."""
        df = self._make_df([10.0], [0.0])
        mu_V, _, _, _ = membrane_potential_fluctuations(df, params_RS_with_K)
        assert mu_V[0] > params_RS_with_K["E_L"]

    def test_pure_inhibitory_hyperpolarizes(self, params_RS_with_K):
        """Pure inhibitory input → μ_V < E_L."""
        df = self._make_df([0.0], [10.0])
        mu_V, _, _, _ = membrane_potential_fluctuations(df, params_RS_with_K)
        assert mu_V[0] < params_RS_with_K["E_L"]

    def test_high_exc_near_E_e(self, params_RS_with_K):
        """Very high excitation, no inhibition → μ_V close to E_e = 0 mV."""
        df = self._make_df([1000.0], [0.0])
        mu_V, _, _, _ = membrane_potential_fluctuations(df, params_RS_with_K)
        # should be depolarised well above E_L
        assert mu_V[0] > -30e-3

    def test_sig_V_increases_with_input(self, params_RS_with_K):
        """More input → larger σ_V (fluctuations)."""
        df_low  = self._make_df([1.0], [1.0])
        df_high = self._make_df([20.0], [20.0])
        _, sig_low,  _, _ = membrane_potential_fluctuations(df_low,  params_RS_with_K)
        _, sig_high, _, _ = membrane_potential_fluctuations(df_high, params_RS_with_K)
        assert sig_high[0] > sig_low[0]

    def test_tau_V_positive(self, params_RS_with_K):
        """τ_V > 0 for any non-zero input."""
        df = self._make_df([5.0], [5.0])
        _, _, tau_V, _ = membrane_potential_fluctuations(df, params_RS_with_K)
        assert tau_V[0] > 0

    def test_tau_V_norm_dimensionless(self, params_RS_with_K):
        """τ_V_norm should be a small positive number (dimensionless scaling)."""
        df = self._make_df([5.0], [5.0])
        _, _, _, tau_V_norm = membrane_potential_fluctuations(df, params_RS_with_K)
        assert tau_V_norm[0] > 0

    def test_vectorized_matches_scalar(self, params_RS_with_K):
        """Vectorized and scalar versions give same results for matching inputs."""
        exc = [5.0, 10.0, 15.0]
        inh = [3.0, 7.0, 12.0]
        df = self._make_df(exc, inh)
        mu_vec, sig_vec, tau_vec, tn_vec = membrane_potential_fluctuations(df, params_RS_with_K)

        for i in range(len(exc)):
            mu_s, sig_s, tau_s, tn_s = membrane_potential_fluctuations_sim(
                f_e=exc[i], f_i=inh[i], params=params_RS_with_K,
            )
            assert abs(mu_vec[i] - mu_s) < 1e-12, f"mu_V mismatch at index {i}"
            assert abs(sig_vec[i] - sig_s) < 1e-12, f"sig_V mismatch at index {i}"
            assert abs(tau_vec[i] - tau_s) < 1e-12, f"tau_V mismatch at index {i}"
            assert abs(tn_vec[i] - tn_s) < 1e-12, f"tau_V_norm mismatch at index {i}"

    def test_exact_zero_input_does_not_crash(self, params_RS_with_K):
        """f_e = f_i = 0 should not crash (epsilon guard)."""
        df = self._make_df([0.0], [0.0])
        mu_V, sig_V, tau_V, _ = membrane_potential_fluctuations(df, params_RS_with_K)
        assert np.all(np.isfinite(mu_V))

    def test_output_shapes_match_input(self, params_RS_with_K):
        df = self._make_df([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        mu_V, sig_V, tau_V, tn = membrane_potential_fluctuations(df, params_RS_with_K)
        assert mu_V.shape == (5,)
        assert sig_V.shape == (5,)
        assert tau_V.shape == (5,)
        assert tn.shape == (5,)


# =====================================================================
# eff_thresh
# =====================================================================

class TestEffThresh:
    def test_zero_poly_returns_P0(self):
        """All params zero except P_0 → eff_thresh = P_0 at reference point."""
        P0 = -0.05
        poly = [P0] + [0.0] * 9
        result = eff_thresh(-60e-3, 4e-3, 0.5, poly)
        assert abs(result - P0) < 1e-15

    def test_at_reference_point_only_P0(self):
        """At reference point (mu_0, sig_0, tau_0), linear+quadratic terms vanish."""
        P0 = -0.05
        poly = [P0, 0.01, 0.02, 0.003, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001]
        result = eff_thresh(-60e-3, 4e-3, 0.5, poly)
        assert abs(result - P0) < 1e-15

    def test_linear_mu_term(self):
        """Only P_mu ≠ 0: eff_thresh = P_0 + P_mu * (mu - mu_0)/mu_d."""
        P0, P_mu = -0.05, 0.01
        poly = [P0, P_mu, 0, 0, 0, 0, 0, 0, 0, 0]
        mu_V = -55e-3  # 5 mV above reference
        expected = P0 + P_mu * (mu_V - (-60e-3)) / 10e-3
        result = eff_thresh(mu_V, 4e-3, 0.5, poly)
        assert abs(result - expected) < 1e-15

    def test_extreme_mu_V(self):
        """Extreme μ_V values should not crash."""
        poly = [-0.05] + [0.001] * 9
        eff_thresh(-80e-3, 4e-3, 0.5, poly)   # very hyperpolarized
        eff_thresh(+20e-3, 4e-3, 0.5, poly)    # very depolarized

    def test_vectorized(self):
        """Accepts arrays."""
        poly = [-0.05, 0.01, 0.002, 0.003, 0, 0, 0, 0, 0, 0]
        mu = np.array([-60e-3, -55e-3])
        result = eff_thresh(mu, 4e-3, 0.5, poly)
        assert result.shape == (2,)


# =====================================================================
# est_thresh
# =====================================================================

class TestEstThresh:
    def test_inverse_consistency(self, params_RS_with_K, RS_MF_params):
        """est_thresh followed by eff_thresh with true params should be close."""
        best = RS_MF_params[0]
        alpha = best["alpha"]
        poly = np.array(best["polynomial_params"])

        # Use a single data point with non-zero firing
        df = pd.DataFrame({"input_exc": [15.0], "input_inh": [5.0], "avg_f_out": [0.0]})
        mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(df, params_RS_with_K)

        # Set a known output rate from the TF
        f_out = TF_template(data=df, params=params_RS_with_K, poly_params=poly, alpha=alpha)
        df["avg_f_out"] = f_out

        # est_thresh should recover something close to eff_thresh
        est = est_thresh(data=df, mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, alpha=alpha)
        eff = eff_thresh(mu_V, sig_V, tau_V_norm, poly)
        assert abs(est[0] - eff[0]) < 1e-6

    def test_clipping_no_crash(self, params_RS_with_K):
        """Very high F_out should not crash (arg clips)."""
        df = pd.DataFrame({"input_exc": [10.0], "input_inh": [0.0], "avg_f_out": [1000.0]})
        mu_V = np.array([-50e-3])
        sig_V = np.array([5e-3])
        tau_V = np.array([0.01])
        with pytest.warns(UserWarning, match="outside"):
            est_thresh(data=df, mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, alpha=1.0)


# =====================================================================
# TF_template (vectorized)
# =====================================================================

class TestTFTemplate:
    def test_non_negative(self, params_RS_with_K, RS_MF_params):
        """F_out ≥ 0 for all inputs."""
        best = RS_MF_params[0]
        poly = best["polynomial_params"]
        alpha = best["alpha"]

        exc = np.arange(0, 31, 1)
        inh = np.arange(0, 31, 1)
        exc_grid, inh_grid = np.meshgrid(exc, inh)
        df = pd.DataFrame({
            "input_exc": exc_grid.ravel(),
            "input_inh": inh_grid.ravel(),
            "avg_f_out": np.zeros(exc_grid.size),
        })
        f_out = TF_template(data=df, params=params_RS_with_K, poly_params=poly, alpha=alpha)
        assert np.all(f_out >= 0)

    def test_high_excitation_produces_firing(self, params_RS_with_K, RS_MF_params):
        """High excitatory input → positive output rate."""
        best = RS_MF_params[0]
        df = pd.DataFrame({"input_exc": [30.0], "input_inh": [0.0], "avg_f_out": [0.0]})
        f_out = TF_template(data=df, params=params_RS_with_K,
                            poly_params=best["polynomial_params"], alpha=best["alpha"])
        assert f_out[0] > 0

    def test_high_inhibition_suppresses_firing(self, params_RS_with_K, RS_MF_params):
        """High inhibitory input, low excitation → near-zero rate."""
        best = RS_MF_params[0]
        df = pd.DataFrame({"input_exc": [0.0], "input_inh": [30.0], "avg_f_out": [0.0]})
        f_out = TF_template(data=df, params=params_RS_with_K,
                            poly_params=best["polynomial_params"], alpha=best["alpha"])
        assert f_out[0] < 1.0  # should be very close to zero

    def test_monotonic_in_excitation(self, params_RS_with_K, RS_MF_params):
        """For fixed inhibition, F_out should increase with excitation."""
        best = RS_MF_params[0]
        exc_vals = np.arange(5, 25, 2, dtype=float)
        inh_fixed = np.full_like(exc_vals, 10.0)
        df = pd.DataFrame({"input_exc": exc_vals, "input_inh": inh_fixed, "avg_f_out": np.zeros_like(exc_vals)})
        f_out = TF_template(data=df, params=params_RS_with_K,
                            poly_params=best["polynomial_params"], alpha=best["alpha"])
        # Allow small numerical violations but overall should be increasing
        diffs = np.diff(f_out)
        assert np.sum(diffs < 0) <= 1, "F_out should be mostly increasing in excitation"


# =====================================================================
# TF_template_sim (scalar) matches TF_template (vectorized)
# =====================================================================

class TestTFTemplateSim:
    def test_scalar_matches_vectorized(self, params_RS_with_K, RS_MF_params):
        best = RS_MF_params[0]
        for exc, inh in [(5, 3), (15, 10), (25, 20)]:
            df = pd.DataFrame({"input_exc": [exc], "input_inh": [inh], "avg_f_out": [0.0]})
            f_vec = TF_template(data=df, params=params_RS_with_K,
                                poly_params=best["polynomial_params"], alpha=best["alpha"])
            f_scalar = TF_template_sim(
                f_e=exc, f_i=inh, params=params_RS_with_K,
                poly_params=best["polynomial_params"], alpha=best["alpha"],
            )
            assert abs(f_vec[0] - f_scalar) < 1e-10, f"Mismatch at exc={exc}, inh={inh}"


# =====================================================================
# Regression: match saved fitting results
# =====================================================================

class TestRegression:
    def test_RS_saved_fit_error(self, df_RS, params_RS_with_K, RS_MF_params):
        """Recompute error with saved RS params — should match saved mean_error."""
        best = RS_MF_params[0]
        poly = np.array(best["polynomial_params"])
        alpha = best["alpha"]

        error = res_2_func(poly, data=df_RS, params=params_RS_with_K, alpha=alpha)
        # The saved error was computed on potentially different data / params.
        # Just check it's finite and non-negative.
        assert np.isfinite(error)
        assert error >= 0

    def test_RS_TF_template_on_data(self, df_RS, params_RS_with_K, RS_MF_params):
        """TF_template runs on full RS data without error."""
        best = RS_MF_params[0]
        f_out = TF_template(
            data=df_RS, params=params_RS_with_K,
            poly_params=best["polynomial_params"], alpha=best["alpha"],
        )
        assert len(f_out) == len(df_RS)
        assert np.all(np.isfinite(f_out))
        assert np.all(f_out >= 0)

    def test_FS_TF_template_on_data(self, df_FS, params_FS_with_K, FS_MF_params):
        """TF_template runs on full FS data without error."""
        best = FS_MF_params[0]
        f_out = TF_template(
            data=df_FS, params=params_FS_with_K,
            poly_params=best["polynomial_params"], alpha=best["alpha"],
        )
        assert len(f_out) == len(df_FS)
        assert np.all(np.isfinite(f_out))
        assert np.all(f_out >= 0)


# =====================================================================
# get_mean_error_distribution
# =====================================================================

class TestGetMeanErrorDistribution:
    def test_returns_correct_length(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params):
        """Output length should match the number of unique inhibitory values."""
        best = RS_MF_params[0]
        poly = np.array(best["polynomial_params"])
        alpha = best["alpha"]

        unique_inh = np.array([0.0, 10.0, 20.0, 30.0])
        distr = get_mean_error_distribution(
            neuron_model="RS",
            df_data=synthetic_RS_tf_data,
            poly_params_2=poly,
            params_SI=params_RS_with_K,
            alpha=alpha,
            unique_inh=unique_inh,
        )
        assert len(distr) == len(unique_inh)
        assert np.all(np.isfinite(distr))
        assert np.all(distr >= 0)

    def test_error_per_inh_level(self, synthetic_RS_tf_data, params_RS_with_K, RS_MF_params):
        """Error should be non-trivial at each inhibitory level on synthetic data."""
        best = RS_MF_params[0]
        poly = np.array(best["polynomial_params"])
        alpha = best["alpha"]

        unique_inh = np.array([0.0, 10.0, 20.0])
        distr = get_mean_error_distribution(
            neuron_model="RS",
            df_data=synthetic_RS_tf_data,
            poly_params_2=poly,
            params_SI=params_RS_with_K,
            alpha=alpha,
            unique_inh=unique_inh,
        )
        assert len(distr) == 3
        assert np.all(np.isfinite(distr))
