"""Mean field of a single population with N synaptic channels.

The population rate follows the first-order equation used for the RS/FS
network (``ntmf.meanfield``):

    tau_f · dν/dt = TF({ν_s(t)}) − ν

where ν_s(t) is the input rate on synaptic channel s and TF is the N-channel
transfer function fitted with ``TF_fitting_neuron_<N>_inputs.ipynb``
(``ntmf.transfer_function.TF_template_rates``). Each channel keeps its own
E_s, Q_s, tau_s and K_s, so inputs need no rescaling onto a reference channel,
unlike the e/i model.

Adaptation is "baked in": the TF was fitted on stationary rates with
w_ad = 0, so the mean adaptation current is absorbed by the fit and there is
no W variable. Rates are therefore expected to match the spiking population
at steady state; adaptation transients after input steps are not reproduced.

The population is disconnected (no recurrent channel), so TF does not depend
on ν. Recurrent or network inputs can later be passed as further time-varying
channel rates.

Also provided, for validation against spiking neurons:
  - :func:`simulate_NN_single_pop` — Brian2 simulation of N disconnected
    AdEx neurons driven by the same protocol,
  - :func:`compare_MF_NN_single_pop` — RMSE and Pearson r between the traces.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from ntmf.transfer_function import TF_template_rates, membrane_potential_fluctuations_rates


# ---------------------------------------------------------------------------
# Loading a TF fit
# ---------------------------------------------------------------------------

def load_TF_fit(json_file_name: str | Path, rank: int = 0) -> tuple[float, np.ndarray, dict]:
    """(alpha, poly_params, entry) from a TF-fitting results file.

    Works with ``results_<model>.json``, ``10_best_params_TF_<model>.json`` and
    ``alpha_results_<folder>.json`` (lists of entries with ``alpha``,
    ``mean_error``, ``polynomial_params``). Entries are sorted by mean error;
    *rank* = 0 is the best one.
    """
    with open(json_file_name, "r") as fh:
        entries = json.load(fh)
    if isinstance(entries, dict):
        entries = [entries]
    entries = sorted(entries, key=lambda e: e["mean_error"])
    e = entries[rank]
    return float(e["alpha"]), np.asarray(e["polynomial_params"], dtype=float), e


# ---------------------------------------------------------------------------
# Driving input
# ---------------------------------------------------------------------------

def driving_input_from_protocol(
    input_config: dict[str, Any],
    time: np.ndarray,
    params: dict | None = None,
    include_background: bool = True,
) -> dict[str, np.ndarray]:
    """Per-channel input rate traces (Hz) from a stimulation-protocol config.

    The config has the layout of ``config/input_config_<N>_channels.json``::

        "connections": {"N_external_k": ..., "conn_prob_k": ...},
        "channels": {"k": {"background_freq": ..., "stimulations":
                           [{"t_start": s, "t_end": s, "freq": Hz}, ...]}}

    Each channel's rate is its background plus the stimulation rate inside
    each [t_start, t_end) window (overlapping windows add).

    If *params* (with K per channel, from ``adding_K_params_general``) is
    given and the protocol's K_k = N_external_k · conn_prob_k differs from the
    K the TF was fitted with, the rate is rescaled by K_protocol / K_fit, so
    that the mean input conductance K·ν·τ·Q is preserved. This is exact for
    μ_V and σ_V, since each channel enters the TF only through K_k·ν_k.

    Returns
    -------
    {channel: ndarray}, same length as *time*.
    """
    conn = input_config["connections"]
    drive: dict[str, np.ndarray] = {}
    for k, ch in input_config["channels"].items():
        nu = np.full(len(time), float(ch.get("background_freq", 0.0)) if include_background else 0.0)
        for stim in ch.get("stimulations", []):
            mask = (time >= stim["t_start"]) & (time < stim["t_end"])
            nu[mask] += float(stim["freq"])

        if params is not None:
            if k not in params["channels"]:
                raise ValueError(f"Protocol channel '{k}' is not a channel of the fitted TF.")
            K_protocol = float(conn[f"N_external_{k}"]) * float(conn[f"conn_prob_{k}"])
            K_fit = params["channels"][k]["K"]
            if not np.isclose(K_protocol, K_fit):
                print(f"channel {k}: protocol K = {K_protocol:g}, fitted K = {K_fit:g} "
                      f"-> rates rescaled by {K_protocol / K_fit:.3g}")
                nu = nu * K_protocol / K_fit
        drive[k] = nu
    return drive


# ---------------------------------------------------------------------------
# Mean-field simulation
# ---------------------------------------------------------------------------

def simulate_MF_single_pop(
    time: np.ndarray,
    driving_input: dict[str, np.ndarray],
    params: dict,
    poly_params: np.ndarray | list,
    alpha: float,
    tau_f: float = 0.01,
    nu_0: float = 0.0,
    method: str = "euler",
    background_in_input: bool = True,
) -> dict[str, np.ndarray]:
    """Integrate tau_f · dν/dt = TF({ν_s(t)}) − ν for one population.

    Parameters
    ----------
    time : ndarray
        Time vector (s), uniform step for ``method='euler'``.
    driving_input : dict
        {channel: rate trace in Hz}, one entry per TF channel, same length as
        *time* (e.g. from :func:`driving_input_from_protocol`).
    params : dict
        SI parameters with a ``channels`` block (``get_params_model_SI_general``
        + ``adding_K_params_general``).
    poly_params, alpha
        TF fit (e.g. from :func:`load_TF_fit`).
    tau_f : float
        Population time constant (s), default 10 ms as in ``ntmf.meanfield``.
    nu_0 : float
        Initial rate (Hz).
    method : str
        ``'euler'`` (explicit, step = time step) or any ``solve_ivp`` method
        (e.g. ``'RK45'``); inputs are held piecewise constant between samples.
    background_in_input : bool
        True (default) if *driving_input* already contains the background
        rates (as returned by :func:`driving_input_from_protocol`); the
        ``nu_bg`` stored in *params* is then set to 0 to avoid counting the
        background twice.

    Returns
    -------
    dict with
        ``t``     : time (s)
        ``nu``    : population rate (Hz)
        ``TF``    : instantaneous TF of the inputs (Hz), the stationary rate
        ``mu_V``, ``sig_V``, ``tau_V`` : membrane statistics of the inputs (SI)
    """
    missing = [k for k in params["channels"] if k not in driving_input]
    if missing:
        raise ValueError(f"No driving input for channel(s) {missing}.")

    p = copy.deepcopy(params)
    if background_in_input:
        for c in p["channels"].values():
            c["nu_bg"] = 0.0

    time = np.asarray(time, dtype=float)
    rates = {k: np.asarray(driving_input[k], dtype=float) for k in p["channels"]}

    # TF and membrane statistics along the input trace (no recurrence: independent of ν)
    TF_t = np.asarray(TF_template_rates(rates, p, poly_params, alpha), dtype=float)
    mu_V, sig_V, tau_V, _ = membrane_potential_fluctuations_rates(rates, p)

    if method == "euler":
        nu = np.empty(len(time))
        nu[0] = nu_0
        dt = np.diff(time)
        for i in range(1, len(time)):
            nu[i] = nu[i - 1] + dt[i - 1] / tau_f * (TF_t[i - 1] - nu[i - 1])
    else:
        TF_interp = interp1d(time, TF_t, kind="previous", bounds_error=False, fill_value=(TF_t[0], TF_t[-1]))
        sol = solve_ivp(lambda t, y: (TF_interp(t) - y) / tau_f, (time[0], time[-1]), [nu_0],
                        method=method, t_eval=time, rtol=1e-8, atol=1e-10, max_step=np.min(np.diff(time)))
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        nu = sol.y[0]

    return {"t": time, "nu": nu, "TF": TF_t, "mu_V": mu_V, "sig_V": sig_V, "tau_V": tau_V}


# ---------------------------------------------------------------------------
# Spiking reference: N disconnected neurons driven by the same protocol
# ---------------------------------------------------------------------------

def simulate_NN_single_pop(
    neuron_model: str,
    neuron_json: str | Path,
    input_config: dict[str, Any],
    sim_duration: float,
    N_cell: int = 50,
    dt: float = 1e-4,
    bin_size: float = 0.05,
    seed: int | None = 12345,
    idx: int = 0,
) -> dict[str, np.ndarray]:
    """Brian2 simulation of *N_cell* disconnected neurons under a protocol.

    Same input scheme as the TF extraction: for every channel k, a background
    Poisson group (if ``background_freq`` > 0) and a stimulation Poisson group
    with ``N_external_k`` sources, each connected with probability
    ``conn_prob_k`` (``Gsyn{k} += Q_k``).

    Parameters
    ----------
    sim_duration, dt, bin_size : float
        In seconds.

    Returns
    -------
    dict with ``t`` (bin centres, s), ``rate`` (population rate, Hz),
    ``spike_t`` (s), ``spike_i``.
    """
    import brian2 as b2
    from ntmf.config import get_syn_info_general
    from ntmf.neurons import create_AdEx_neuron

    b2.start_scope()
    if seed is not None:
        b2.seed(seed)
    b2.defaultclock.dt = dt * b2.second
    T = sim_duration * b2.second
    times = b2.arange(0, T, dt * b2.second)

    Q = get_syn_info_general(neuron_json, idx=idx)
    zero_current = b2.TimedArray(np.zeros(int(sim_duration * 1e3) + 2) * b2.amp, dt=1. * b2.ms)
    G = create_AdEx_neuron(neuron_model=neuron_model, json_file_name=neuron_json, idx=idx,
                           N_cell=N_cell, curr_inj=zero_current)

    conn = input_config["connections"]
    objects = [G]
    namespace = {"current": zero_current}
    for k, ch in input_config["channels"].items():
        if k not in Q:
            raise ValueError(f"Protocol channel '{k}' is not a channel of {neuron_model}.")
        N_ext = int(conn[f"N_external_{k}"])
        p_conn = float(conn[f"conn_prob_{k}"])
        namespace[f"Q_{k}"] = Q[k]

        bg = float(ch.get("background_freq", 0.0))
        if bg > 0:
            P_bg = b2.PoissonGroup(N_ext, rates=bg * b2.Hz)
            S_bg = b2.Synapses(P_bg, G, on_pre=f"Gsyn{k}_post += Q_{k}")
            S_bg.connect(p=p_conn)
            objects += [P_bg, S_bg]

        stims = ch.get("stimulations", [])
        if stims:
            rate = np.zeros(len(times))
            t_s = np.asarray(times / b2.second)
            for st in stims:
                rate[(t_s >= st["t_start"]) & (t_s < st["t_end"])] += st["freq"]
            namespace[f"rate_timed_array_{k}"] = b2.TimedArray(rate * b2.Hz, dt=dt * b2.second)
            P_st = b2.PoissonGroup(N_ext, rates=f"rate_timed_array_{k}(t)")
            S_st = b2.Synapses(P_st, G, on_pre=f"Gsyn{k}_post += Q_{k}")
            S_st.connect(p=p_conn)
            objects += [P_st, S_st]

    mon = b2.SpikeMonitor(G)
    objects.append(mon)
    net = b2.Network(*objects)
    net.run(T, namespace=namespace)

    spike_t = np.asarray(mon.t / b2.second)
    edges = np.arange(0, sim_duration + bin_size / 2, bin_size)
    counts, _ = np.histogram(spike_t, bins=edges)
    return {"t": 0.5 * (edges[:-1] + edges[1:]),
            "rate": counts / (N_cell * bin_size),
            "spike_t": spike_t,
            "spike_i": np.asarray(mon.i)}


def compare_MF_NN_single_pop(
    mf: dict[str, np.ndarray],
    nn: dict[str, np.ndarray],
    t_window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """RMSE (Hz) and Pearson r between the MF rate and the binned spiking rate.

    The MF trace is averaged over each spiking bin (bins of equal width
    centred on ``nn['t']``), so both traces have the same time resolution.
    *t_window* = (t0, t1) restricts the comparison (e.g. to skip the initial
    transient).
    """
    t_nn, r_nn = np.asarray(nn["t"]), np.asarray(nn["rate"])
    half = 0.5 * (t_nn[1] - t_nn[0])
    mf_binned = np.array([mf["nu"][(mf["t"] >= c - half) & (mf["t"] < c + half)].mean() for c in t_nn])

    sel = np.ones(len(t_nn), dtype=bool)
    if t_window is not None:
        sel = (t_nn >= t_window[0]) & (t_nn <= t_window[1])
    a, b = mf_binned[sel], r_nn[sel]
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    corr = 0.0 if (np.std(a) < 1e-12 or np.std(b) < 1e-12) else float(pearsonr(a, b)[0])
    return {"rmse": rmse, "corr": corr, "t": t_nn, "mf_binned": mf_binned}
