"""Transfer function mathematics.

Implements the analytical transfer function mapping (ν_exc, ν_inh) → F_out
based on Zerlaut et al., 2018, with adaptation extension from Di Volo et al., 2019.

The key components are:
1. Membrane potential fluctuation statistics (μ_V, σ_V, τ_V)
2. Effective threshold polynomial (10 free parameters)
3. erfc-based output rate formula
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.special as sp_spec
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Reference / scaling constants for eff_thresh polynomial
# ---------------------------------------------------------------------------

_MU_0  = -60e-3   # V   – reference mean membrane potential
_SIG_0 =   4e-3   # V   – reference membrane potential std
_TAU_0 =   0.5    # s   – reference autocorrelation time

_MU_D  =  10e-3   # V   – scale for μ_V
_SIG_D =   6e-3   # V   – scale for σ_V
_TAU_D =   1.0    # s   – scale for τ_V


# ---------------------------------------------------------------------------
# Membrane potential fluctuations
# ---------------------------------------------------------------------------

def membrane_potential_fluctuations(
    data: pd.DataFrame,
    params: dict[str, float],
    w_ad: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (μ_V, σ_V, τ_V, τ_V_norm) for each row in *data*.

    Vectorized version that operates on DataFrames with columns
    ``input_exc`` and ``input_inh`` (rates in Hz).

    Based on Zerlaut et al., 2018, eq. 5–17.

    Parameters
    ----------
    data : DataFrame
        Must have columns ``input_exc`` and ``input_inh``.
    params : dict
        Must contain K_e, K_i, tau_syn, Q_e, Q_i, g_L, C_m, E_e, E_i, E_L.
    w_ad : float or ndarray, optional
        Adaptation current (Amps, SI). Scalar or per-row array aligned to
        *data*. Default 0 (no adaptation).

    Returns
    -------
    (mu_V, sig_V, tau_V, tau_V_norm) : each a 1-D ndarray
    """
    f_e = np.maximum(data["input_exc"].to_numpy(), 1e-9)
    f_i = np.maximum(data["input_inh"].to_numpy(), 1e-9)

    # eq. 5 – synaptic conductance statistics
    mu_Ge  = f_e * params["K_e"] * params["tau_syn"] * params["Q_e"]
    sig_Ge = np.sqrt(0.5 * f_e * params["K_e"] * params["tau_syn"]) * params["Q_e"]
    mu_Gi  = f_i * params["K_i"] * params["tau_syn"] * params["Q_i"]
    sig_Gi = np.sqrt(0.5 * f_i * params["K_i"] * params["tau_syn"]) * params["Q_i"]

    # eq. 6 – total conductance & membrane time constant
    mu_G = mu_Ge + mu_Gi + params["g_L"]
    tau_m = params["C_m"] / mu_G

    # eq. 7 – mean membrane potential (with optional adaptation)
    if np.ndim(w_ad) == 0:
        w_ad = np.full_like(f_i, w_ad)
    mu_V = (mu_Ge * params["E_e"] + mu_Gi * params["E_i"]
            + params["g_L"] * params["E_L"] - w_ad) / mu_G

    # between eq. 9–10 – unitary PSP amplitudes
    U_e = params["Q_e"] / mu_G * (params["E_e"] - mu_V)
    U_i = params["Q_i"] / mu_G * (params["E_i"] - mu_V)

    # eq. 15 – membrane potential std
    sig_V = np.sqrt(
        params["K_e"] * f_e * (U_e * params["tau_syn"]) ** 2 / (2 * (tau_m + params["tau_syn"]))
        + params["K_i"] * f_i * (U_i * params["tau_syn"]) ** 2 / (2 * (tau_m + params["tau_syn"]))
    )

    # eq. 17 – autocorrelation time
    tau_V = (
        (params["K_e"] * f_e * (U_e * params["tau_syn"]) ** 2
         + params["K_i"] * f_i * (U_i * params["tau_syn"]) ** 2)
        / (params["K_e"] * f_e * (U_e * params["tau_syn"]) ** 2 / (tau_m + params["tau_syn"])
           + params["K_i"] * f_i * (U_i * params["tau_syn"]) ** 2 / (tau_m + params["tau_syn"]))
    )
    tau_V_norm = tau_V * params["g_L"] / params["C_m"]

    return mu_V, sig_V, tau_V, tau_V_norm


def membrane_potential_fluctuations_sim(
    f_e: float,
    f_i: float,
    params: dict[str, float],
    w_ad: float = 0.0,
) -> tuple[float, float, float, float]:
    """Scalar version of :func:`membrane_potential_fluctuations` for MF integration."""
    f_e = max(f_e, 1e-9)
    f_i = max(f_i, 1e-9)

    mu_Ge  = f_e * params["K_e"] * params["tau_syn"] * params["Q_e"]
    sig_Ge = np.sqrt(0.5 * f_e * params["K_e"] * params["tau_syn"]) * params["Q_e"]
    mu_Gi  = f_i * params["K_i"] * params["tau_syn"] * params["Q_i"]
    sig_Gi = np.sqrt(0.5 * f_i * params["K_i"] * params["tau_syn"]) * params["Q_i"]

    mu_G = mu_Ge + mu_Gi + params["g_L"]
    tau_m = params["C_m"] / mu_G

    mu_V = (mu_Ge * params["E_e"] + mu_Gi * params["E_i"]
            + params["g_L"] * params["E_L"] - w_ad) / mu_G

    U_e = params["Q_e"] / mu_G * (params["E_e"] - mu_V)
    U_i = params["Q_i"] / mu_G * (params["E_i"] - mu_V)

    sig_V = np.sqrt(
        params["K_e"] * f_e * (U_e * params["tau_syn"]) ** 2 / (2 * (tau_m + params["tau_syn"]))
        + params["K_i"] * f_i * (U_i * params["tau_syn"]) ** 2 / (2 * (tau_m + params["tau_syn"]))
    )

    tau_V = (
        (params["K_e"] * f_e * (U_e * params["tau_syn"]) ** 2
         + params["K_i"] * f_i * (U_i * params["tau_syn"]) ** 2)
        / (params["K_e"] * f_e * (U_e * params["tau_syn"]) ** 2 / (tau_m + params["tau_syn"])
           + params["K_i"] * f_i * (U_i * params["tau_syn"]) ** 2 / (tau_m + params["tau_syn"]))
    )
    tau_V_norm = tau_V * params["g_L"] / params["C_m"]

    return mu_V, sig_V, tau_V, tau_V_norm


# ---------------------------------------------------------------------------
# Effective threshold
# ---------------------------------------------------------------------------

def eff_thresh(
    mu_V: np.ndarray | float,
    sig_V: np.ndarray | float,
    tau_V_norm: np.ndarray | float,
    poly_params: np.ndarray | list,
) -> np.ndarray | float:
    """Second-order polynomial effective threshold.

    ``V_th_eff = P_0 + V_1 + V_2`` where V_1 is linear in (μ, σ, τ)
    and V_2 contains quadratic and cross terms, all normalised by reference
    values.

    Parameters
    ----------
    poly_params : array-like of length 10
        [P_0, P_μ, P_σ, P_τ, P_μ², P_σ², P_τ², P_μσ, P_μτ, P_στ]
    """
    P_0, P_mu, P_sig, P_tau, P_mu2, P_sig2, P_tau2, P_mu_sig, P_mu_tau, P_sig_tau = poly_params

    dmu  = (mu_V - _MU_0) / _MU_D
    dsig = (sig_V - _SIG_0) / _SIG_D
    dtau = (tau_V_norm - _TAU_0) / _TAU_D

    V_1 = P_mu * dmu + P_sig * dsig + P_tau * dtau
    V_2 = (P_mu2 * dmu**2 + P_sig2 * dsig**2 + P_tau2 * dtau**2
           + P_mu_sig * dmu * dsig + P_mu_tau * dmu * dtau + P_sig_tau * dsig * dtau)

    return P_0 + V_1 + V_2


# ---------------------------------------------------------------------------
# Threshold estimation (inverse of TF)
# ---------------------------------------------------------------------------

def est_thresh(
    data: pd.DataFrame,
    mu_V: np.ndarray,
    sig_V: np.ndarray,
    tau_V: np.ndarray,
    alpha: float,
    clip_arg: bool = True,
) -> np.ndarray:
    """Estimate the effective threshold from data by inverting the erfc formula.

    Given F_out, μ_V, σ_V, τ_V, and α, solves:
        F_out = α · erfc(z) / (2τ_V)
    for z, then V_th = μ_V + √2 · σ_V · z.
    """
    F_out = np.asarray(data["avg_f_out"].to_numpy()) + 1e-12
    mu_V = np.asarray(mu_V)
    sig_V = np.maximum(np.asarray(sig_V), 1e-9)
    tau_V = np.maximum(np.asarray(tau_V), 1e-9)

    arg = (1.0 / alpha) * (F_out * 2.0 * tau_V)

    n_invalid = int(np.sum((arg <= 0.0) | (arg >= 2.0)))
    if n_invalid > 0:
        warnings.warn(
            f"est_thresh: {n_invalid} element(s) of 'arg' outside (0, 2). "
            "They will be clipped for erfcinv.",
            stacklevel=2,
        )

    if clip_arg:
        arg = np.clip(arg, 1e-12, 2.0 - 1e-12)

    return mu_V + np.sqrt(2.0) * sig_V * sp_spec.erfcinv(arg)


# ---------------------------------------------------------------------------
# Residuals (objective functions)
# ---------------------------------------------------------------------------

def res_1_func(
    poly_params: np.ndarray,
    mu_V: np.ndarray,
    sig_V: np.ndarray,
    tau_V_norm: np.ndarray,
    est_V_th: np.ndarray,
) -> float:
    """Stage-1 residual: MSE between estimated and polynomial threshold.

    Note
    ----
    Adaptation enters through the precomputed *mu_V* / *sig_V* / *tau_V_norm*
    (and hence *est_V_th*), so no ``w_ad`` argument is needed here.
    """
    eff_V_th = eff_thresh(mu_V, sig_V, tau_V_norm, poly_params)
    return float(np.mean((est_V_th - eff_V_th) ** 2))


def res_2_func(
    poly_params: np.ndarray,
    data: pd.DataFrame,
    params: dict[str, float],
    alpha: float,
    w_ad: float | np.ndarray = 0.0,
) -> float:
    """Stage-2 residual: MSE between data and TF prediction.

    Parameters
    ----------
    w_ad : float or ndarray, optional
        Adaptation current (Amps, SI). Scalar or per-row array aligned to
        *data* (same row order). Default 0 reproduces the non-adaptive fit.
    """
    F_out = data["avg_f_out"].to_numpy()
    F_pred = TF_template(
        data=data, params=params, poly_params=poly_params, alpha=alpha, w_ad=w_ad,
    )
    return float(np.mean((F_out - F_pred) ** 2))


# ---------------------------------------------------------------------------
# Transfer function template
# ---------------------------------------------------------------------------

def TF_template(
    data: pd.DataFrame,
    params: dict[str, float],
    poly_params: np.ndarray | list,
    alpha: float,
    w_ad: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Vectorised transfer function: (ν_e, ν_i) → F_out.

    For each row in *data*, computes:
        F_out = α · erfc(z) / (2 · τ_V)
    where z = (V_th_eff − μ_V) / (√2 · σ_V).

    ``w_ad`` may be a scalar or a per-row array aligned to *data*.
    """
    mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations(data=data, params=params, w_ad=w_ad)

    sig_V_safe = np.maximum(np.asarray(sig_V), 1e-9)
    tau_V_safe = np.maximum(np.asarray(tau_V), 1e-9)

    eff_V_th = eff_thresh(mu_V=mu_V, sig_V=sig_V_safe, tau_V_norm=tau_V_norm, poly_params=poly_params)

    z = (eff_V_th - mu_V) / (np.sqrt(2.0) * sig_V_safe)
    F_out_th = alpha * sp_spec.erfc(z) / (2.0 * tau_V_safe)

    return np.maximum(F_out_th, 0.0)


def TF_template_sim(
    f_e: float,
    f_i: float,
    params: dict[str, float],
    poly_params: np.ndarray | list,
    alpha: float,
    w_ad: float = 0.0,
) -> float:
    """Scalar transfer function for mean-field ODE integration."""
    mu_V, sig_V, tau_V, tau_V_norm = membrane_potential_fluctuations_sim(
        f_e=f_e, f_i=f_i, params=params, w_ad=w_ad,
    )

    sig_V_safe = max(sig_V, 1e-9)
    tau_V_safe = max(tau_V, 1e-9)

    eff_V_th = eff_thresh(
        mu_V=mu_V, sig_V=sig_V_safe, tau_V_norm=tau_V_norm, poly_params=poly_params,
    )

    z = (eff_V_th - mu_V) / (np.sqrt(2.0) * sig_V_safe)
    F_out_th = alpha * sp_spec.erfc(z) / (2.0 * tau_V_safe)

    return max(F_out_th, 0.0)


# ---------------------------------------------------------------------------
# Error distribution
# ---------------------------------------------------------------------------

def get_mean_error_distribution(
    *args,
    **kwargs,
) -> np.ndarray:
    """Compute mean error sliced by inhibitory input level.

    Backward-compatible with old ``utils/TF_helper.py`` signature:

        get_mean_error_distribution(neuron_model, df_data, poly_params_2,
                                     params_SI, alpha, unique_inh, alpha_idx=None)

    and the preferred new signature:

        get_mean_error_distribution(df_data, poly_params_2, params_SI,
                                     alpha, unique_inh, alpha_idx=None)

    Adaptation
    ----------
    Pass ``w_ad`` as a keyword (scalar, or per-row array aligned to
    *df_data* in its original row order) to evaluate the adapted TF. It is
    sliced consistently with each inhibition mask. Default 0.
    """
    alpha_idx: int | None = kwargs.pop("alpha_idx", None)
    w_ad = kwargs.pop("w_ad", 0.0)

    if len(args) == 5:          # new positional call
        df_data, poly_params_2, params_SI, alpha, unique_inh = args
    elif len(args) == 6:       # old positional call (neuron_model as arg 1)
        _neuron_model, df_data, poly_params_2, params_SI, alpha, unique_inh = args
    elif len(args) == 7:       # old positional with alpha_idx
        _neuron_model, df_data, poly_params_2, params_SI, alpha, unique_inh, passed_alpha_idx = args
        if alpha_idx is None:
            alpha_idx = passed_alpha_idx
    else:                       # keyword-only call
        df_data = kwargs.pop("df_data")
        poly_params_2 = kwargs.pop("poly_params_2")
        params_SI = kwargs.pop("params_SI")
        alpha = kwargs.pop("alpha")
        unique_inh = kwargs.pop("unique_inh")

    if not isinstance(df_data, pd.DataFrame):
        raise TypeError(f"Expected df_data to be a DataFrame, got {type(df_data)}")

    w_ad_arr = np.asarray(w_ad)

    distr = np.zeros(len(unique_inh))
    for idx, fixed_inh in enumerate(unique_inh):
        tol = 1e-6
        mask = np.isclose(df_data["input_inh"], fixed_inh, atol=tol)
        # slice w_ad consistently with the mask (positional alignment)
        if w_ad_arr.ndim == 0:
            w_slice: float | np.ndarray = float(w_ad_arr)
        else:
            w_slice = w_ad_arr[np.asarray(mask)]
        distr[idx] = res_2_func(
            poly_params_2, data=df_data.loc[mask].copy(), params=params_SI,
            alpha=alpha, w_ad=w_slice,
        )
    return distr

# ---------------------------------------------------------------------------
# Membrane fluctuation EGLIF - GoC
# ---------------------------------------------------------------------------
def get_membrane_fluct_eglif_goc(fi_grid = None, 
                                  fe_m_grid = None,
                                  fe_grid = None,                                  
                                  adapt = None,
                                  params_SI = None):

    Cm = params_SI['C_m'] # membrane capacitance
    Gl = params_SI['g_L'] # leak condunctance
    El = params_SI['E_L'] # leak equilibrium potential

    # excitatory synapses
    Qe_m = params_SI['Q_e_m']
    Te_m = params_SI['T_e_m']
    Ee = params_SI['E_e']
    Ke_m = params_SI['K_e_m']

    Qe = params_SI['Q_e']
    Te = params_SI['T_e']
    #Ee = params_SI['E_e'] #one Ee for both exc syn
    Ke = params_SI['K_e']    
    
    # inhibitory synapses    
    Qi = params_SI['Q_i']
    Ti = params_SI['T_i']
    Ei = params_SI['E_i']
    Ki = params_SI['K_i']
    
    fi = fi_grid
    fe_m = fe_m_grid
    fe = fe_grid


    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe, muGe_m, muGi = Qe * Ke * Te * fe, Qe_m * Ke_m * Te_m * fe_m, Qi * Ki * Ti * fi #EQUAL TO EXP
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe + muGe_m + muGi #EQUAL TO EXP
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    muV = (np.e * (muGe * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - adapt) / muG  # XX = adaptation

    muGn, Tm = muG / Gl, Cm / muG  # normalization

    Ue, Ue_m, Ui = Qe / muG * (Ee - muV), Qe_m / muG * (Ee - muV), Qi / muG * (Ei - muV) #EQUAL TO EXP



    sigVe = (2 * Tm + Te) * ((np.e * Ue * Te)/ (2 * (Te + Tm))) ** 2 * Ke * fe
    sigVe_m = (2 * Tm + Te_m) * ((np.e * Ue_m * Te_m) / (2 * (Te_m + Tm))) ** 2 * Ke_m * fe_m
    sigVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2* (Ti + Tm))) ** 2 * Ki * fi

    sigV = np.sqrt(sigVe + sigVe_m + sigVi)

    fe_m, fe, fi = fe_m + 1e-15, fe + 1e-15, fi + 1e-15  # just to insure a non zero division

    Tv_num= Ke * fe * Ue ** 2 * Te ** 2 * np.e ** 2 + \
            Ke_m * fe_m * Ue_m ** 2 * Te_m ** 2 * np.e ** 2 + \
            Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sigV+1e-20) ** 2)


    TvN = Tv * Gl / Cm  # normalization

    return muGe, muGe_m, muGi, muG, muV, sigV+1e-20, muGn, TvN, Tv

#-----------------------------------------------------------#
#---------------------- GOLGI special ----------------------#
#-----------------------------------------------------------#

def fitting_Vthre_then_Freq_data_eglif_goc(muGe = None,
                                            muGe_m = None,     
                                            muGi = None,
                                            muG = None,
                                            muV = None,
                                            sigV = None,
                                            muGn = None,
                                            TvN = None, 
                                            Freq_data = None,
                                            fe_m_grid = None, 
                                            fe_grid = None,                                             
                                            fi_grid  = None,
                                            adapt = None,
                                            params_SI = None, 
                                            alpha = None,
                                            maxiter=50000, xtol=1e-5, with_square_terms=True):
    
    Gl, Cm, El = params_SI['g_L'], params_SI['C_m'], params_SI['E_L']

    muGe, muGe_m, muGi, muG, muV, sigV, muGn, TvN, Tv = get_membrane_fluct_eglif_goc(fi_grid = fi_grid,
                                                                                       fe_m_grid = fe_m_grid,
                                                                                       fe_grid = fe_grid,
                                                                                       adapt = adapt,
                                                                                       params_SI = params_SI)

    freq_data_lim= Freq_data.max()

    i_non_zeros = np.where((Freq_data > 0.) & (Freq_data < freq_data_lim))
    print("Freq_data Limit: ", freq_data_lim)

    Vthre_eff = effective_Vthre_EGLIF(Y = Freq_data[i_non_zeros],
                                muV = muV[i_non_zeros], 
                                sigV = sigV[i_non_zeros],
                                TvN = TvN[i_non_zeros],
                                Gl = params_SI['g_L'],
                                Cm = params_SI['C_m'],
                                alpha = alpha)

    P = [-50e-3, 0, 0, 0, 0]

    print("========== OPTIMIZATION STEP 1 ========== \nVthre Optimization")
    
    def Res(poly_coeff):
        Vthre = threshold_func_EGLIF(*poly_coeff,
                               muV = muV[i_non_zeros],
                               sigV = sigV[i_non_zeros],
                               TvN = TvN[i_non_zeros],
                               muGn = muGn[i_non_zeros],
                               )
        
        return np.mean((Vthre_eff - Vthre) ** 2)

    plsq = minimize(Res, P, options={'disp': True})
    P = plsq.x

    print('========== P output from Vthreshold opt: ', P)

    print("========== OPTIMIZATION STEP 2 ========== \nFreq_data Optimization")

    def Res(poly_coeff):
        return np.mean((Freq_data -
                        TF_template_eglif_goc(*poly_coeff,
                                          fe_m = fe_m_grid,
                                          fe = fe_grid,
                                          fi = fi_grid,
                                          adapt = adapt, 
                                          alpha = alpha,
                                          params_SI = params_SI)) ** 2)


    plsq = minimize(Res, P, method='nelder-mead',
                    options={'xatol': xtol, 'disp': True, 'maxiter': maxiter})

    P = plsq.x
    print('========== P output from Freq_data opt: ', P)

    print("========== END OF OPTIMIZATION PROCESS ==========")

    params_SI['P'] = P

    return P

def get_membrane_fluct_eglif_goc(fi_grid = None, 
                                  fe_m_grid = None,
                                  fe_grid = None,                                  
                                  adapt = None,
                                  params_SI = None):

    Cm = params_SI['C_m'] # membrane capacitance
    Gl = params_SI['g_L'] # leak condunctance
    El = params_SI['E_L'] # leak equilibrium potential

    # excitatory synapses
    Qe_m = params_SI['Q_e_m']
    Te_m = params_SI['T_e_m']
    Ee = params_SI['E_e']
    Ke_m = params_SI['K_e_m']

    Qe = params_SI['Q_e']
    Te = params_SI['T_e']
    #Ee = params_SI['E_e'] #one Ee for both exc syn
    Ke = params_SI['K_e']    
    
    # inhibitory synapses    
    Qi = params_SI['Q_i']
    Ti = params_SI['T_i']
    Ei = params_SI['E_i']
    Ki = params_SI['K_i']
    
    fi = fi_grid
    fe_m = fe_m_grid
    fe = fe_grid


    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe, muGe_m, muGi = Qe * Ke * Te * fe, Qe_m * Ke_m * Te_m * fe_m, Qi * Ki * Ti * fi #EQUAL TO EXP
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe + muGe_m + muGi #EQUAL TO EXP
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    muV = (np.e * (muGe * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - adapt) / muG  # XX = adaptation

    muGn, Tm = muG / Gl, Cm / muG  # normalization

    Ue, Ue_m, Ui = Qe / muG * (Ee - muV), Qe_m / muG * (Ee - muV), Qi / muG * (Ei - muV) #EQUAL TO EXP



    sigVe = (2 * Tm + Te) * ((np.e * Ue * Te)/ (2 * (Te + Tm))) ** 2 * Ke * fe
    sigVe_m = (2 * Tm + Te_m) * ((np.e * Ue_m * Te_m) / (2 * (Te_m + Tm))) ** 2 * Ke_m * fe_m
    sigVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2* (Ti + Tm))) ** 2 * Ki * fi

    sigV = np.sqrt(sigVe + sigVe_m + sigVi)

    fe_m, fe, fi = fe_m + 1e-15, fe + 1e-15, fi + 1e-15  # just to insure a non zero division

    Tv_num= Ke * fe * Ue ** 2 * Te ** 2 * np.e ** 2 + \
            Ke_m * fe_m * Ue_m ** 2 * Te_m ** 2 * np.e ** 2 + \
            Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sigV+1e-20) ** 2)


    TvN = Tv * Gl / Cm  # normalization

    return muGe, muGe_m, muGi, muG, muV, sigV+1e-20, muGn, TvN, Tv

def pseq_params_eglif_goc(params):

    Qe, Qe_m = params['Q_e'], params['Q_e_m']
    Te, Te_m, Ee = params['T_e'], params['T_e_m'], params['E_e']
    Qi, Ti, Ei = params['Q_i'], params['T_i'], params['E_i']
    Gl, Cm , El = params['g_L'], params['C_m'] , params['E_L']

    Ke = params['K_e']
    Ke_m = params['K_e_m']
    Ki = params['K_i']

    return Qe, Qe_m, Te, Te_m, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ke_m, Ki

def TF_template_eglif_goc(P0, P1, P2, P3, P4,
                      fe = None,
                      fe_m = None,
                      fi = None,
                      adapt = None, 
                      alpha = None,
                      params_SI = None):
    
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    if hasattr(fe_m, "__len__"):
        fe_m = np.where(fe_m < 1e-8, 1e-8, fe)
    else:
        fe_m = max(fe_m, 1e-8)
    if hasattr(fe, "__len__"):
        fe = np.where(fe < 1e-8, 1e-8, fe)
    else:
        fe = max(fe, 1e-8)
    if hasattr(fi, "__len__"):
        fi = np.where(fi < 1e-8, 1e-8, fi)
    else:
        fi = max(fi, 1e-8)

    muGe, muGe_m, muGi, muG, muV, sigV, muGn, TvN, Tv = get_membrane_fluct_eglif_goc(fi_grid = fi,
                                                                     fe_m_grid = fe_m,
                                                                     fe_grid = fe,
                                                                     adapt = adapt,
                                                                     params_SI = params_SI)


    Vthre = threshold_func_EGLIF(muV = muV,
                           sigV = sigV,
                           TvN = TvN,
                           muGn = muGn,
                           P0 = P0,
                           P1 = P1,
                           P2 = P2,
                           P3 = P3,
                           P4 = P4)


    if (hasattr(muV, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        sigV[sigV < 1e-4] = 1e-4
    else:
        if (sigV < 1e-4):
            sigV = 1e-4

    Fout_th = erfc_func_EGLIF(muV = muV,
                        sigV = sigV,
                        TvN = TvN,
                        Vthre = Vthre,
                        Gl = params_SI['g_L'],
                        Cm = params_SI['C_m'],
                        alpha = alpha)

    if (hasattr(Fout_th, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        Fout_th[Fout_th < 1e-8] = 1e-8
    else:
        if (Fout_th < 1e-8):
            Fout_th = 1e-8

    return Fout_th

def effective_Vthre_EGLIF(Y = None,
                    muV = None,
                    sigV = None,
                    TvN = None,
                    Gl = None,
                    Cm = None,
                    alpha= None):

    arg = (1 / alpha) * (Y * 2 * TvN * Cm / Gl)
    print(f'min: {np.min(arg)}, max:{np.max(arg)}')
    print(np.sum((arg <= 0) | (arg >= 2)), "invalid argument values")

    Vthre_eff = muV + np.sqrt(2) * sigV * sp_spec.erfcinv((1 / alpha) * (Y * 2 * TvN * Cm / Gl))  # effective threshold

    return Vthre_eff

# Polynomial expression of V eff thre - initial condition
def threshold_func_EGLIF(P0, P1, P2, P3, P4,
                   muV = None,
                   sigV = None,
                   TvN = None,
                   muGn = None):

    # normalization factors as in Zerlaut et al., (2018)
    muV0, DmuV0 = -60e-3, 10e-3
    sigV0, DsigV0 = 4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.

    Vthre =  P0 + \
             P1 * (muV - muV0) / DmuV0 + \
             P2 * (sigV - sigV0) / DsigV0 + \
             P3 * (TvN - TvN0) / DTvN0 + \
             P4 * np.log(muGn)
    return Vthre

def TF_template_EGLIF(P0, P1, P2, P3, P4,
                      fe = None,
                      fi = None,
                      adapt = None, 
                      alpha = None,
                      params_SI = None):
    
    # here TOTAL (sum over synapses) excitatory and inhibitory input

    if hasattr(fe, "__len__"):
        fe = np.where(fe < 1e-8, 1e-8, fe)
    else:
        fe = max(fe, 1e-8)
    if hasattr(fi, "__len__"):
        fi = np.where(fi < 1e-8, 1e-8, fi)
    else:
        fi = max(fi, 1e-8)

    muGe, muGi, muG, muV, sigV, muGn, TvN = get_membrane_fluct_eglif(fi_grid = fi,
                                                                     fe_grid = fe,
                                                                     adapt = adapt,
                                                                     params_SI = params_SI)

    Vthre = threshold_func(muV = muV,
                           sigV = sigV,
                           TvN = TvN,
                           muGn = muGn,
                           P0 = P0,
                           P1 = P1,
                           P2 = P2,
                           P3 = P3,
                           P4 = P4)

    if hasattr(sigV, "__len__"):
        sigV = np.where(sigV < 1e-4, 1e-4, sigV)
    else:
        sigV = max(sigV, 1e-4)

    Fout_th = erfc_func(muV = muV,
                        sigV = sigV,
                        TvN = TvN,
                        Vthre = Vthre,
                        Gl = params_SI['g_L'],
                        Cm = params_SI['C_m'],
                        alpha = alpha)

    if hasattr(Fout_th, "__len__"):
        Fout_th = np.where(Fout_th < 1e-8, 1e-8, Fout_th)
    else:
        Fout_th = max(Fout_th, 1e-8)

    return Fout_th    

# EQUATION OF ERFC = vout = Fout = TF expression (Zerlaut et al. 2018)
def erfc_func_EGLIF(muV = None,
              sigV = None,
              TvN = None,
              Vthre = None,
              Gl = None,
              Cm= None, 
              alpha = None):
    
    return .5 / TvN * Gl / Cm * (sp_spec.erfc((Vthre - muV) / np.sqrt(2) / sigV)) * alpha
