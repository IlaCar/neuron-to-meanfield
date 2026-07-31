"""Mean-field simulation for a two-population (FS + RS) network.

The firing rate of each population evolves as a first-order ODE:

    dν/dt = (TF(ν_eff_exc, ν_eff_inh) − ν) / τ_f

where the effective input rates combine external drive and recurrent
connections weighted by K and Q.

When ``adaptation=True`` the RS (excitatory) population additionally carries a
mean adaptation current ``W`` (Amps, SI) as a third state variable, following
Di Volo et al., 2019:

    τ_w dW/dt = −W + b·τ_w·ν_RS + a·(μ_V_RS − E_L)

``W`` is fed into the RS transfer function as ``w_ad``, so it hyperpolarises
μ_V exactly as in the single-neuron characterisation. The FS population is
non-adaptive. With ``adaptation=False`` (default) the model reduces to the
original 2-variable form with ``w_ad=0`` (the "baked-in" approach).

Two integration methods are provided:
  - :func:`simulate_MF_FS_RS` — Euler (backward-compatible)
  - :func:`simulate_MF_FS_RS_ode` — adaptive Runge-Kutta via scipy
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from ntmf.transfer_function import (
    TF_template_sim,
    membrane_potential_fluctuations_sim,
)


# ---------------------------------------------------------------------------
# helper: defensive adaptation-parameter lookup
# ---------------------------------------------------------------------------

def _lookup(d: dict[str, float], names: list[str], what: str) -> float:
    for n in names:
        if n in d:
            return float(d[n])
    raise KeyError(
        f"adaptation parameter '{what}' not found in RS params; tried {names}. "
        f"Available keys: {sorted(d)}"
    )


# ---------------------------------------------------------------------------
# MFModel — bundles all constant parameters
# ---------------------------------------------------------------------------

class MFModel:
    """Pre-computed constants for a 2-population (FS, RS) mean-field model.

    Constructing this once and calling :meth:`rhs` avoids re-computing
    network-structure constants on every time step.

    Parameters
    ----------
    adaptation : bool
        If True, add the RS adaptation current ``W`` as a third state
        variable and feed it into the RS transfer function. Requires
        ``a``, ``b``, ``tau_w`` (SI) in ``params["RS"]``. Default False.
    """

    def __init__(
        self,
        params: dict[str, dict[str, float]],
        poly_params: dict[str, list | np.ndarray],
        alphas: dict[str, float],
        network_config: dict[str, Any],
        tau_f: float = 0.01,
        adaptation: bool = False,
    ):
        # --- TF parameters ---
        self.params = params
        self.poly_params = {
            k: np.asarray(v, dtype=float) for k, v in poly_params.items()
        }
        self.alphas = alphas
        self.tau_f = tau_f
        self.adaptation = adaptation

        # --- adaptation parameters (RS only) ---
        if self.adaptation:
            rs = params["RS"]
            self.a_RS = _lookup(rs, ["a", "a_w", "a_adapt"], "a")
            self.b_RS = _lookup(rs, ["b", "b_w", "b_adapt"], "b")
            self.tau_w_RS = _lookup(rs, ["tau_w", "tau_adapt", "tauw"], "tau_w")
            self.E_L_RS = float(rs["E_L"])
        else:
            self.a_RS = self.b_RS = self.tau_w_RS = self.E_L_RS = 0.0

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

    @property
    def n_state(self) -> int:
        """Number of state variables: 3 with adaptation (ν_FS, ν_RS, W), else 2."""
        return 3 if self.adaptation else 2

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
        """ODE right-hand side.

        State is ``y = [ν_FS, ν_RS]`` (no adaptation) or
        ``y = [ν_FS, ν_RS, W]`` (with adaptation).
        """
        if self.adaptation:
            nu_FS, nu_RS, W = y[0], y[1], y[2]
        else:
            nu_FS, nu_RS = y[0], y[1]
            W = 0.0

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
        # RS sees the current adaptation W; FS is non-adaptive.
        F_RS = TF_template_sim(
            f_e=nu_eff_exc_RS, f_i=nu_eff_inh_RS,
            params=self.params["RS"], poly_params=self.poly_params["RS"],
            alpha=self.alphas["RS"], w_ad=W,
        )
        F_FS = TF_template_sim(
            f_e=nu_eff_exc_FS, f_i=nu_eff_inh_FS,
            params=self.params["FS"], poly_params=self.poly_params["FS"],
            alpha=self.alphas["FS"], w_ad=0.0,
        )

        dnu_FS = (F_FS - nu_FS) / self.tau_f
        dnu_RS = (F_RS - nu_RS) / self.tau_f

        if not self.adaptation:
            return np.array([dnu_FS, dnu_RS])

        # --- RS adaptation current (Di Volo et al., 2019) ---
        # spike-triggered term (b) always present; subthreshold term (a)
        # only if a != 0, and it needs μ_V evaluated at the current W.
        if self.a_RS != 0.0:
            mu_V_RS, _, _, _ = membrane_potential_fluctuations_sim(
                f_e=nu_eff_exc_RS, f_i=nu_eff_inh_RS,
                params=self.params["RS"], w_ad=W,
            )
            sub = self.a_RS * (mu_V_RS - self.E_L_RS)
        else:
            sub = 0.0

        dW = (-W + self.b_RS * self.tau_w_RS * nu_RS + sub) / self.tau_w_RS

        return np.array([dnu_FS, dnu_RS, dW])


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
    adaptation: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate a 2-population mean-field network using adaptive ODE integration.

    Parameters are identical to :func:`simulate_MF_FS_RS`, plus:

    Parameters
    ----------
    method : str
        Integration method passed to ``scipy.integrate.solve_ivp``
        (default ``'RK45'``).
    adaptation : bool
        If True, integrate the RS adaptation current ``W`` explicitly and
        return it under the ``"W"`` key. Default False.

    Returns
    -------
    dict[str, ndarray]
        ``{"FS": rates, "RS": rates}`` — firing rate traces in Hz.
        If ``adaptation=True`` also includes ``"W"`` (Amps, SI).
    """
    model = MFModel(
        params=params,
        poly_params=poly_params,
        alphas=alphas,
        network_config=network_config,
        tau_f=tau_f,
        adaptation=adaptation,
    )
    model.set_driving_input(time, driving_input)

    y0 = np.zeros(model.n_state)

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
    # y[0] = FS, y[1] = RS, (y[2] = W)  (order matches neuron_models)
    out = {
        pops[0]: sol.y[0],
        pops[1]: sol.y[1],
    }
    if model.adaptation:
        out["W"] = sol.y[2]
    return out


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
    adaptation: bool = False,
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
        With ``adaptation=True``, ``params["RS"]`` must also contain
        ``a``, ``b``, ``tau_w`` (SI).
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
    adaptation : bool
        If True, integrate the RS adaptation current ``W`` explicitly and
        return it under the ``"W"`` key. Default False.

    Returns
    -------
    dict[str, ndarray]
        ``{"FS": rates, "RS": rates}`` — firing rate traces in Hz.
        If ``adaptation=True`` also includes ``"W"`` (Amps, SI).
    """
    model = MFModel(
        params=params,
        poly_params=poly_params,
        alphas=alphas,
        network_config=network_config,
        tau_f=tau_f,
        adaptation=adaptation,
    )
    model.set_driving_input(time, driving_input)

    dt = time[1] - time[0]
    n_steps = len(time)
    pops = neuron_models

    rates = {pop: np.zeros(n_steps) for pop in pops}
    W = np.zeros(n_steps) if model.adaptation else None

    for t_idx in range(1, n_steps):
        t = time[t_idx]
        if model.adaptation:
            y = np.array([
                rates[pops[0]][t_idx - 1],
                rates[pops[1]][t_idx - 1],
                W[t_idx - 1],
            ])
        else:
            y = np.array([rates[pops[0]][t_idx - 1], rates[pops[1]][t_idx - 1]])

        dydt = model.rhs(t, y)

        rates[pops[0]][t_idx] = rates[pops[0]][t_idx - 1] + dt * dydt[0]
        rates[pops[1]][t_idx] = rates[pops[1]][t_idx - 1] + dt * dydt[1]
        if model.adaptation:
            W[t_idx] = W[t_idx - 1] + dt * dydt[2]

    if model.adaptation:
        rates["W"] = W
    return rates
