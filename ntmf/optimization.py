"""Transfer function optimisation (two-stage fitting + alpha grid search)."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ntmf.transfer_function import (
    TF_template,
    est_thresh,
    membrane_potential_fluctuations,
    res_1_func,
    res_2_func,
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
    saving_json: bool = False,
    out_file_name: Optional[str] = None,
) -> tuple[float, Optional[np.ndarray]]:
    """Run the two-stage fitting procedure for a given *alpha*.

    Stage 1: Estimate threshold from data → fit polynomial (SLSQP).
    Stage 2: Refit polynomial by minimising rate MSE (Nelder-Mead).

    Returns
    -------
    (mean_error, poly_params) — (inf, None) on failure.
    """
    try:
        # Stage 1
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
            args=(df_data, params_SI, alpha),
            method="Nelder-Mead",
            options={"disp": False, "maxiter": 10000},
        )
        poly_params_2 = fit_2["x"]

        mean_error = res_2_func(poly_params_2, data=df_data, params=params_SI, alpha=alpha)

        if saving_json and out_file_name is not None:
            entry = {
                "alpha": float(alpha),
                "mean_error": float(mean_error),
                "polynomial_params": poly_params_2.tolist(),
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
    alpha_min: float = 0.1,
    alpha_max: float = 2.0,
    alpha_step: float = 0.001,
    saving_json: bool = False,
    out_file_name: Optional[str] = None,
) -> tuple[float, float, Optional[np.ndarray]]:
    """Search over discrete alpha values and return the best fit.

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
            saving_json=saving_json, out_file_name=out_file_name,
        )
        print(f"Tested alpha: {alpha:.3f}, mean_error: {mean_error:.6f}")

        if mean_error < best_error:
            best_error = mean_error
            best_alpha = float(alpha)
            best_poly = poly_params_2

    print(f"\nBest alpha: {best_alpha:.3f}, mean_error: {best_error:.6f}")
    return best_alpha, best_error, best_poly   # type: ignore[return-value]
