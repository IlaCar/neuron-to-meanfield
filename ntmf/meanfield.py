"""Mean-field simulation for a two-population (FS + RS) network.

The firing rate of each population evolves as a first-order ODE:

    dν/dt = (TF(ν_eff_exc, ν_eff_inh) − ν) / τ_f

where the effective input rates combine external drive and recurrent
connections weighted by K and Q.

Two integration methods are provided:
  - :func:`simulate_MF_FS_RS` — Euler (backward-compatible)
  - :func:`simulate_MF_FS_RS_ode` — adaptive Runge-Kutta via scipy
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from ntmf.transfer_function import TF_template_sim


# ---------------------------------------------------------------------------
# MFModel — bundles all constant parameters
# ---------------------------------------------------------------------------

class MFModel:
    """Pre-computed constants for a 2-population (FS, RS) mean-field model.

    Constructing this once and calling :meth:`rhs` avoids re-computing
    network-structure constants on every time step.
    """

    def __init__(
        self,
        params: dict[str, dict[str, float]],
        poly_params: dict[str, list | np.ndarray],
        alphas: dict[str, float],
        network_config: dict[str, Any],
        tau_f: float = 0.01,
    ):
        # --- TF parameters ---
        self.params = params
        self.poly_params = {
            k: np.asarray(v, dtype=float) for k, v in poly_params.items()
        }
        self.alphas = alphas
        self.tau_f = tau_f

        # --- network structure ---
        N_FS = network_config["network_composition"]["FS_neuron"]
        N_RS = network_config["network_composition"]["RS_neuron"]

        p_ext = network_config["external_input"]["conn_prob"]
        N_exc = network_config["external_input"]["N_external_exc"]
        N_inh = network_config["external_input"]["N_external_inh"]
        self.K_ext_exc = N_exc * p_ext
        self.K_ext_inh = N_inh * p_ext

        p = network_config["network_composition"]["conn_prob"]
        self.K_RS_to_RS = int(p * N_RS)
        self.K_RS_to_FS = int(p * N_RS)
        self.K_FS_to_RS = int(p * N_FS)
        self.K_FS_to_FS = int(p * N_FS)

        self.K_ref_exc = self.K_RS_to_RS
        self.K_ref_inh = self.K_FS_to_FS

        # --- quantal conductances (SI) ---
        ext = network_config["external_input"]
        self.Qe_ext = ext["Q_e"] * 1e-9   # nS → S
        self.Qi_ext = ext["Q_i"] * 1e-9

        self.Qe_RS_RS = params["RS"]["Q_e"]
        self.Qe_RS_FS = params["FS"]["Q_e"]
        self.Qi_FS_FS = params["FS"]["Q_i"]
        self.Qi_FS_RS = params["RS"]["Q_i"]

        # --- driving-input interpolators (set later) ---
        self._interp_exc_FS: interp1d | None = None
        self._interp_exc_RS: interp1d | None = None
        self._interp_inh_FS: interp1d | None = None
        self._interp_inh_RS: interp1d | None = None

    def set_driving_input(
        self,
        time: np.ndarray,
        driving_input: dict[str, dict[str, np.ndarray]],
    ) -> None:
        """Build interpolators for the time-varying external drive.

        Uses ``kind='previous'`` so that step-function inputs are preserved.
        """
        kw = dict(kind="previous", fill_value=0.0, bounds_error=False)
        self._interp_exc_FS = interp1d(time, driving_input["excitatory"]["FS"], **kw)
        self._interp_exc_RS = interp1d(time, driving_input["excitatory"]["RS"], **kw)
        self._interp_inh_FS = interp1d(time, driving_input["inhibitory"]["FS"], **kw)
        self._interp_inh_RS = interp1d(time, driving_input["inhibitory"]["RS"], **kw)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side: ``dy/dt`` for ``y = [ν_FS, ν_RS]``."""
        nu_FS, nu_RS = y[0], y[1]

        # interpolate external drive at time t
        nu_ext_exc_FS = float(self._interp_exc_FS(t))
        nu_ext_exc_RS = float(self._interp_exc_RS(t))
        nu_ext_inh_FS = float(self._interp_inh_FS(t))
        nu_ext_inh_RS = float(self._interp_inh_RS(t))

        # --- effective input rates for RS ---
        nu_eff_exc_RS = (
            self.K_ext_exc * self.Qe_ext * nu_ext_exc_RS
            + self.K_RS_to_RS * self.Qe_RS_RS * nu_RS
        ) / (self.K_ref_exc * self.Qe_RS_RS)

        nu_eff_inh_RS = (
            self.K_ext_inh * self.Qi_ext * nu_ext_inh_RS
            + self.K_FS_to_RS * self.Qi_FS_RS * nu_FS
        ) / (self.K_ref_inh * self.Qi_FS_RS)

        # --- effective input rates for FS ---
        nu_eff_exc_FS = (
            self.K_ext_exc * self.Qe_ext * nu_ext_exc_FS
            + self.K_RS_to_FS * self.Qe_RS_FS * nu_RS
        ) / (self.K_ref_exc * self.Qe_RS_FS)

        nu_eff_inh_FS = (
            self.K_ext_inh * self.Qi_ext * nu_ext_inh_FS
            + self.K_FS_to_FS * self.Qi_FS_FS * nu_FS
        ) / (self.K_ref_inh * self.Qi_FS_FS)

        # --- evaluate transfer functions ---
        F_RS = TF_template_sim(
            f_e=nu_eff_exc_RS, f_i=nu_eff_inh_RS,
            params=self.params["RS"], poly_params=self.poly_params["RS"],
            alpha=self.alphas["RS"], w_ad=0.0,
        )
        F_FS = TF_template_sim(
            f_e=nu_eff_exc_FS, f_i=nu_eff_inh_FS,
            params=self.params["FS"], poly_params=self.poly_params["FS"],
            alpha=self.alphas["FS"], w_ad=0.0,
        )

        return np.array([
            (F_FS - nu_FS) / self.tau_f,
            (F_RS - nu_RS) / self.tau_f,
        ])


# ---------------------------------------------------------------------------
# ODE-based simulation (scipy solve_ivp)
# ---------------------------------------------------------------------------

def simulate_MF_FS_RS_ode(
    time: np.ndarray,
    neuron_models: list[str],
    params: dict[str, dict[str, float]],
    poly_params: dict[str, list | np.ndarray],
    alphas: dict[str, float],
    network_config: dict[str, Any],
    driving_input: dict[str, dict[str, np.ndarray]],
    tau_f: float = 0.01,
    method: str = "RK45",
) -> dict[str, np.ndarray]:
    """Simulate a 2-population mean-field network using adaptive ODE integration.

    Parameters are identical to :func:`simulate_MF_FS_RS`, plus:

    Parameters
    ----------
    method : str
        Integration method passed to ``scipy.integrate.solve_ivp``
        (default ``'RK45'``).

    Returns
    -------
    dict[str, ndarray]
        ``{"FS": rates, "RS": rates}`` — firing rate traces in Hz,
        evaluated at the time points in *time*.
    """
    model = MFModel(
        params=params,
        poly_params=poly_params,
        alphas=alphas,
        network_config=network_config,
        tau_f=tau_f,
    )
    model.set_driving_input(time, driving_input)

    y0 = np.array([0.0, 0.0])

    sol = solve_ivp(
        model.rhs,
        t_span=(time[0], time[-1]),
        y0=y0,
        method=method,
        t_eval=time,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    pops = neuron_models
    # y[0] = FS, y[1] = RS  (order matches neuron_models)
    return {
        pops[0]: sol.y[0],
        pops[1]: sol.y[1],
    }


# ---------------------------------------------------------------------------
# Euler-based simulation (backward-compatible)
# ---------------------------------------------------------------------------

def simulate_MF_FS_RS(
    time: np.ndarray,
    neuron_models: list[str],
    params: dict[str, dict[str, float]],
    poly_params: dict[str, list | np.ndarray],
    alphas: dict[str, float],
    network_config: dict[str, Any],
    driving_input: dict[str, dict[str, np.ndarray]],
    tau_f: float = 0.01,
) -> dict[str, np.ndarray]:
    """Simulate a 2-population mean-field network using Euler integration.

    Parameters
    ----------
    time : ndarray
        1-D time vector in seconds.
    neuron_models : list[str]
        Population names, e.g. ``["FS", "RS"]``.
    params : dict
        ``{"FS": params_SI_FS, "RS": params_SI_RS}`` (with K_e/K_i already added).
    poly_params : dict
        ``{"FS": [...], "RS": [...]}`` — 10 polynomial coefficients per pop.
    alphas : dict
        ``{"FS": float, "RS": float}``.
    network_config : dict
        Network configuration (must include ``network_composition``,
        ``external_input`` with ``Q_e``, ``Q_i``).
    driving_input : dict
        ``{"excitatory": {"FS": ndarray, "RS": ndarray},
            "inhibitory": {"FS": ndarray, "RS": ndarray}}``
        Each array has the same length as *time*.
    tau_f : float
        Population rate time constant in seconds (default 10 ms).

    Returns
    -------
    dict[str, ndarray]
        ``{"FS": rates, "RS": rates}`` — firing rate traces in Hz.
    """
    model = MFModel(
        params=params,
        poly_params=poly_params,
        alphas=alphas,
        network_config=network_config,
        tau_f=tau_f,
    )
    model.set_driving_input(time, driving_input)

    dt = time[1] - time[0]
    n_steps = len(time)
    pops = neuron_models

    rates = {pop: np.zeros(n_steps) for pop in pops}

    for t_idx in range(1, n_steps):
        t = time[t_idx]
        y = np.array([rates[pops[0]][t_idx - 1], rates[pops[1]][t_idx - 1]])
        dydt = model.rhs(t, y)

        rates[pops[0]][t_idx] = rates[pops[0]][t_idx - 1] + dt * dydt[0]
        rates[pops[1]][t_idx] = rates[pops[1]][t_idx - 1] + dt * dydt[1]

    return rates
