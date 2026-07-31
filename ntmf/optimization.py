"""Transfer function optimisation (two-stage fitting + alpha grid search)."""

from __future__ import annotations

import json
import os
from typing import Optional
import io
import contextlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ntmf.transfer_function import (
    TF_template,
    est_thresh,
    membrane_potential_fluctuations,
    res_1_func,
    res_2_func,
    fitting_Vthre_then_Freq_data_eglif_goc,
    TF_template_eglif_goc
)


# ---------------------------------------------------------------------------
# Single alpha fit
# ---------------------------------------------------------------------------

def run_fits(
    alpha: float,
    df_data: pd.DataFrame,
    mu_V: np.ndarray,
    sig_V: np.ndarray,
    tau_V: np.ndarray,
    tau_V_norm: np.ndarray,
    params_SI: dict[str, float],
    *,
    w_ad: float | np.ndarray = 0.0,
    saving_json: bool = False,
    out_file_name: Optional[str] = None,
) -> tuple[float, Optional[np.ndarray]]:
    """Run the two-stage fitting procedure for a given *alpha*.

    Stage 1: Estimate threshold from data → fit polynomial (SLSQP).
    Stage 2: Refit polynomial by minimising rate MSE (Nelder-Mead).

    Parameters
    ----------
    w_ad : float or ndarray, optional
        Adaptation current (Amps, SI). Scalar, or a per-row array aligned to
        *df_data*. Default 0 (non-adaptive fit).

        Adaptation enters stage 1 implicitly, through the *mu_V*, *sig_V*,
        *tau_V*, *tau_V_norm* arrays the caller supplies: these must have been
        computed by :func:`membrane_potential_fluctuations` with the *same*
        ``w_ad``, otherwise the two stages characterise different neurons.
        Stage 2 receives ``w_ad`` explicitly.

    Returns
    -------
    (mean_error, poly_params) — (inf, None) on failure.
    """
    try:
        # Stage 1
        # (mu_V, sig_V, tau_V already carry w_ad)
        est_V_th = est_thresh(data=df_data, mu_V=mu_V, sig_V=sig_V, tau_V=tau_V, alpha=alpha)

        poly_params_init = -np.ones(10) * 1e-3
        poly_params_init[0] = float(np.mean(est_V_th))

        fit_1 = minimize(
            res_1_func,
            poly_params_init,
            args=(mu_V, sig_V, tau_V_norm, est_V_th),
            method="SLSQP",
            options={"ftol": 1e-17, "disp": False, "maxiter": 3000},
        )
        poly_params_1 = fit_1["x"]

        # Stage 2
        fit_2 = minimize(
            res_2_func,
            poly_params_1,
            args=(df_data, params_SI, alpha, w_ad),
            method="Nelder-Mead",
            options={"disp": False, "maxiter": 10000},
        )
        poly_params_2 = fit_2["x"]

        mean_error = res_2_func(
            poly_params_2, data=df_data, params=params_SI, alpha=alpha, w_ad=w_ad,
        )

        if saving_json and out_file_name is not None:
            entry = {
                "alpha": float(alpha),
                "mean_error": float(mean_error),
                "polynomial_params": poly_params_2.tolist(),
                "adaptation": bool(np.any(np.asarray(w_ad) != 0.0)),
            }
            if os.path.exists(out_file_name):
                with open(out_file_name, "r") as fh:
                    data = json.load(fh)
            else:
                data = []
            data.append(entry)
            with open(out_file_name, "w") as fh:
                json.dump(data, fh, indent=4)

        return float(mean_error), poly_params_2

    except Exception as exc:
        print(f"Warning: alpha={alpha} caused an exception: {exc}")
        return float(np.inf), None


# ---------------------------------------------------------------------------
# Grid search over alpha
# ---------------------------------------------------------------------------

def discrete_alpha_search(
    df_data: pd.DataFrame,
    mu_V: np.ndarray,
    sig_V: np.ndarray,
    tau_V: np.ndarray,
    tau_V_norm: np.ndarray,
    params_SI: dict[str, float],
    *,
    w_ad: float | np.ndarray = 0.0,
    alpha_min: float = 0.1,
    alpha_max: float = 2.0,
    alpha_step: float = 0.001,
    saving_json: bool = False,
    out_file_name: Optional[str] = None,
) -> tuple[float, float, Optional[np.ndarray]]:
    """Search over discrete alpha values and return the best fit.

    Parameters
    ----------
    w_ad : float or ndarray, optional
        Adaptation current (Amps, SI), forwarded to :func:`run_fits`. Must be
        the same ``w_ad`` used to compute *mu_V*, *sig_V*, *tau_V*,
        *tau_V_norm*. Default 0 (non-adaptive search).

    Returns
    -------
    (best_alpha, best_error, best_poly_params)
    """
    alpha_candidates = np.round(np.arange(alpha_min, alpha_max + alpha_step, alpha_step), 3)

    best_alpha: Optional[float] = None
    best_error = float(np.inf)
    best_poly: Optional[np.ndarray] = None

    for alpha in alpha_candidates:
        mean_error, poly_params_2 = run_fits(
            alpha, df_data, mu_V, sig_V, tau_V, tau_V_norm, params_SI,
            w_ad=w_ad, saving_json=saving_json, out_file_name=out_file_name,
        )
        print(f"Tested alpha: {alpha:.3f}, mean_error: {mean_error:.6f}")

        if mean_error < best_error:
            best_error = mean_error
            best_alpha = float(alpha)
            best_poly = poly_params_2

    print(f"\nBest alpha: {best_alpha:.3f}, mean_error: {best_error:.6f}")
    return best_alpha, best_error, best_poly   # type: ignore[return-value]

# -----------------------------------------------------
def discrete_alpha_search_3_inputs_EGLIF(muGe=None,
                                   muGe_m=None,
                                   muGi=None,
                                   muG=None,
                                   muV=None,
                                   sigV=None,
                                   muGn=None,
                                   TvN=None,
                                   Freq_data=None,
                                   fe_grid=None,
                                   fe_m_grid=None,
                                   fi_grid=None,
                                   adapt=None,
                                   params_SI=None,
                                   alpha_min=0.1, alpha_max=2.0, alpha_step=0.001,
                                   maxiter=50000, xtol=1e-5,
                                   verbose=False,
                                   saving_json=False, out_file_name=None):
    """
    Three-input variant of discrete_alpha_search for GoC-type cells (two
    excitatory drives fe, fe_m and one inhibitory drive fi). Searches over alpha
    values and returns the best E-GLIF fit.
    """

    alpha_candidates = np.round(np.arange(alpha_min, alpha_max + alpha_step, alpha_step), 3)
    best_alpha = None
    best_error = np.inf
    best_poly = None

    for alpha in alpha_candidates:
        mean_error, P = run_fits_3_inputs_EGLIF(
            alpha=alpha,
            muGe=muGe, muGe_m=muGe_m, muGi=muGi, muG=muG,
            muV=muV, sigV=sigV, muGn=muGn, TvN=TvN,
            Freq_data=Freq_data,
            fe_grid=fe_grid, fe_m_grid=fe_m_grid, fi_grid=fi_grid,
            adapt=adapt,
            params_SI=params_SI,
            maxiter=maxiter, xtol=xtol,
            verbose=verbose,
            saving_json=saving_json, out_file_name=out_file_name,
        )
        print(f'Tested alpha: {alpha:.3f}, mean_error: {mean_error:.6f}')
        if mean_error < best_error:
            best_error = mean_error
            best_alpha = alpha
            best_poly = P

    if best_alpha is None:
        print('\nNo alpha produced a finite error.')
    else:
        print(f'\nBest alpha: {best_alpha:.3f}, mean_error: {best_error:.6f}')
    return best_alpha, best_error, best_poly

# -----------------------------------------------------
def run_fits_3_inputs_EGLIF(alpha=None,
                      muGe=None,
                      muGe_m=None,
                      muGi=None,
                      muG=None,
                      muV=None,
                      sigV=None,
                      muGn=None,
                      TvN=None,
                      Freq_data=None,
                      fe_grid=None,
                      fe_m_grid=None,
                      fi_grid=None,
                      adapt=None,
                      params_SI=None,
                      maxiter=50000, xtol=1e-5,
                      verbose=False,
                      saving_json=False, out_file_name=None):

    """
    Three-input variant of run_fits for GoC-type cells, which receive TWO
    excitatory drives (via granule cells 'fe', via mossy fibres 'fe_m') and ONE
    inhibitory drive ('fi').

    Runs the two-step E-GLIF fit (effective Vthre -> Freq_data) for a given alpha
    via fitting_Vthre_then_Freq_data_eglif_goc, and returns the mean squared
    error on the firing-rate data together with the fitted polynomial
    coefficients P.

    Membrane-fluctuation quantities (muGe, muGe_m, muGi, muG, muV, sigV, muGn,
    TvN) are computed ONCE outside and passed in, since they do not depend on
    alpha.
    """

    try:
        # fitting_Vthre_then_Freq_data_eglif_goc writes params_SI['P'] and prints
        # a lot at every call; work on a shallow copy and silence stdout unless
        # verbose, so a long sweep stays quiet and leaves the caller's dict clean.
        params_local = dict(params_SI)

        sink = io.StringIO()
        ctx = contextlib.redirect_stdout(sink) if not verbose else contextlib.nullcontext()
        with ctx:
            P = fitting_Vthre_then_Freq_data_eglif_goc(
                muGe=muGe,
                muGe_m=muGe_m,
                muGi=muGi,
                muG=muG,
                muV=muV,
                sigV=sigV,
                muGn=muGn,
                TvN=TvN,
                Freq_data=Freq_data,
                fe_m_grid=fe_m_grid,
                fe_grid=fe_grid,
                fi_grid=fi_grid,
                adapt=adapt,
                params_SI=params_local,
                alpha=alpha,
                maxiter=maxiter, xtol=xtol,
            )

        P = np.asarray(P, dtype=float)

        Fout = TF_template_eglif_goc(
            *P,
            fe=np.array(fe_grid, dtype=float),
            fe_m=np.array(fe_m_grid, dtype=float),
            fi=np.array(fi_grid, dtype=float),
            adapt=adapt,
            alpha=alpha,
            params_SI=params_local,
        )
        mean_error = float(np.mean((np.asarray(Freq_data) - Fout) ** 2))

        # Small alpha can push the erfcinv argument in effective_Vthre out of
        # (0, 2) -> NaN Vthre_eff -> NaN fit. Treat non-finite as a failed alpha.
        if not np.isfinite(mean_error):
            print(f"Warning: alpha={alpha} produced a non-finite error; skipped.")
            return np.inf, None

        if saving_json and out_file_name is not None:
            data_structure_to_save = {
                'alpha': float(alpha),
                'mean_error': float(mean_error),
                'polynomial_params': P.tolist(),
            }
            if os.path.exists(out_file_name):
                with open(out_file_name, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            data.append(data_structure_to_save)
            with open(out_file_name, 'w') as f:
                json.dump(data, f, indent=4)

        return mean_error, P

    except Exception as e:
        print(f"Warning: alpha={alpha} caused an exception: {e}")
        return np.inf, None