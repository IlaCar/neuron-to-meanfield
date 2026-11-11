"""Configuration and parameter loading.

Handles JSON config files for neuron models, network topology, and input protocols.
All parameter values are returned in SI units unless noted.
"""

import json
from pathlib import Path
from typing import Any

import brian2 as b2
import h5py
import numpy as np
import pandas as pd

import os

AdEx_IMPLEMENTED_NEURON_MODELS = ["FS", "RS", "RS_no_adapt"]
IMPLEMENTED_NEURON_MODELS = ["FS", "RS", "RS_no_adapt", "GoC", "neuron_3_inputs", "neuron_4_inputs"]


# ---------------------------------------------------------------------------
# Neuron model parameters
# ---------------------------------------------------------------------------

def get_params_model_SI(
    neuron_model: str,
    json_file_name: str | Path,
) -> dict[str, float]:
    """Load AdEx neuron model parameters and convert to SI units.

    Parameters
    ----------
    neuron_model : str
        One of IMPLEMENTED_NEURON_MODELS.
    json_file_name : str or Path
        Path to the neuron model JSON file (e.g. ``FS.json``).

    Returns
    -------
    dict
        Model parameters in SI units (F, S, V, s, A).

    Raises
    ------
    ValueError
        If *neuron_model* is not recognised or *json_file_name* is None.
    """
    if neuron_model is None:
        raise ValueError("Please specify the neuron_model you wish to simulate.")
    if neuron_model not in IMPLEMENTED_NEURON_MODELS:
        raise ValueError(
            f"neuron_model must be one of {IMPLEMENTED_NEURON_MODELS}, "
            f"but got '{neuron_model}'."
        )
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")

    with open(json_file_name, "r") as fh:
        data = json.load(fh)

    m = data[0][0]["model"]

    return {
        "C_m":     m["C_m"]     * 1e-9,   # nF -> F
        "g_L":     m["g_L"]     * 1e-6,   # uS -> S
        "E_L":     m["E_L"]     * 1e-3,   # mV -> V
        "a":       m["a"]       * 1e-9,   # nS -> S
        "b":       m["b"]       * 1e-9,   # nA -> A
        "tau_w":   m["tau_w"]   * 1e-3,   # ms -> s
        "V_th":    m["V_th"]    * 1e-3,
        "Delta_T": m["Delta_T"] * 1e-3,
        "V_reset": m["V_reset"] * 1e-3,
        "V_peak":  m["V_peak"]  * 1e-3,
        "t_ref":   m["t_ref"]   * 1e-3,
        "E_e":     m["E_e"]     * 1e-3,
        "Q_e":     m["Q_e"]     * 1e-9,
        "E_i":     m["E_i"]     * 1e-3,
        "Q_i":     m["Q_i"]     * 1e-9,
        "tau_syn": m["tau_syn"] * 1e-3,
    }

def get_params_model_SI_general(
    json_file_name: str | Path,
    idx: int = 0,
) -> dict[str, Any]:
    """Load AdEx parameters with N synaptic channels and convert to SI units.

    N-channel counterpart of :func:`get_params_model_SI`. Membrane and
    adaptation parameters are returned at the top level, as in the e/i
    version; the synaptic channels go in a ``channels`` block::

        params["channels"] = {"1": {"E": ..., "Q": ..., "tau": ...}, "2": {...}, ...}

    The presence of ``channels`` is what routes
    ``membrane_potential_fluctuations`` and ``TF_template`` to their
    N-channel versions. The number of inputs per channel (``K``) and the
    background rate (``nu_bg``) are added by :func:`adding_K_params_general`.

    Works for classic e/i JSONs too (channels ``"E"`` and ``"I"``, both with
    ``tau_syn``), which is useful for checking that the two code paths agree.
    """
    from ntmf.neurons import synaptic_channels_from_model  # local import: neurons imports config

    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")
    with open(json_file_name, "r") as fh:
        m = json.load(fh)[0][idx]["model"]

    params: dict[str, Any] = {
        "C_m":     m["C_m"]     * 1e-9,   # nF -> F
        "g_L":     m["g_L"]     * 1e-6,   # uS -> S
        "E_L":     m["E_L"]     * 1e-3,   # mV -> V
        "a":       m["a"]       * 1e-9,   # nS -> S
        "b":       m["b"]       * 1e-9,   # nA -> A
        "tau_w":   m["tau_w"]   * 1e-3,   # ms -> s
        "V_th":    m["V_th"]    * 1e-3,
        "Delta_T": m["Delta_T"] * 1e-3,
        "V_reset": m["V_reset"] * 1e-3,
        "V_peak":  m["V_peak"]  * 1e-3,
        "t_ref":   m["t_ref"]   * 1e-3,
        "channels": {},
    }
    for c in synaptic_channels_from_model(m):
        params["channels"][c["name"]] = {
            "E":   m[c["E_key"]]   * 1e-3,   # mV -> V
            "Q":   m[c["Q_key"]]   * 1e-9,   # nS -> S
            "tau": m[c["tau_key"]] * 1e-3,   # ms -> s
        }
    return params

def adding_K_params_general(
    neuron_params: dict[str, Any],
    tf_meta: dict[str, Any] | str | Path,
    include_background: bool = True,
) -> dict[str, Any]:
    """Add K (inputs per neuron) and the background rate to every channel (in-place).

    N-channel counterpart of :func:`adding_K_params`. Values are taken from
    the ``_meta.json`` written by the TF extraction script, so the fit uses
    exactly the connectivity the data were simulated with:
    ``K_k = N_external_k * conn_prob_k``.

    Parameters
    ----------
    tf_meta : dict or path
        Meta dict (or path to the ``_meta.json``).
    include_background : bool
        If True (default), ``nu_bg`` is set to the channel's background rate,
        and the TF adds it to the swept input rate. The background in the
        extraction script is an independent Poisson group with the same
        ``N_external`` and ``conn_prob``, so the two rates add. If False,
        ``nu_bg`` = 0, i.e. the background is ignored as in the e/i RS fit.
    """
    if not isinstance(tf_meta, dict):
        with open(tf_meta, "r") as fh:
            tf_meta = json.load(fh)

    missing = [k for k in neuron_params["channels"] if k not in tf_meta["channels"]]
    if missing:
        raise ValueError(f"Channels {missing} not found in the TF meta file.")

    for k, ch in neuron_params["channels"].items():
        m = tf_meta["channels"][k]
        ch["K"] = float(m["N_external"]) * float(m["conn_prob"])
        ch["nu_bg"] = float(m.get("background_freq", 0.0)) if include_background else 0.0
    return neuron_params
    
#-------------------------------------------------------------------    
def get_params_EGLIF_SI(p):
    params_SI={}
    params_SI['C_m'] = p['C_m'] * 10**-12
    params_SI['g_L'] = p['g_L'] * 10**-9
    params_SI['E_L'] = p['E_L'] * 10**-3
    params_SI['V_th'] = p['V_th'] * 10**-3
    params_SI['V_reset'] = p['V_reset'] * 10**-3
    params_SI['V_spike'] = p['V_spike'] * 10**-3
    params_SI['delta_V'] = p['delta_V'] * 10**-3
    params_SI['tau_V'] = p['tau_V'] * 10**-3
    params_SI['lambda_0'] = p['lambda_0']
    params_SI['t_ref'] = p['t_ref'] * 10**-3
    params_SI['k_a'] = p['k_a']
    params_SI['k_2'] = p['k_2'] * 10**3
    params_SI['k_1'] = p['k_1'] * 10**3
    params_SI['A_2'] = p['A_2'] * 10**-12
    params_SI['A_1'] = p['A_1'] * 10**-12
    params_SI['I_e'] = p['I_e'] * 10**-9
    params_SI['E_e'] = p['E_e'] * 10**-3
    params_SI['K_e'] = p['K_e']
    params_SI['T_e'] = p['T_e'] * 10**-3
    params_SI['Q_e'] = p['Q_e'] * 10**-9
    params_SI['E_i'] = p['E_i'] * 10**-3
    params_SI['K_i'] = p['K_i']
    params_SI['T_i'] = p['T_i'] * 10**-3
    params_SI['Q_i'] = p['Q_i'] * 10**-9

    if 'Q_e_m' in p.keys():
        params_SI['K_e_m'] = p['K_e_m']
        params_SI['T_e_m'] = p['T_e_m'] * 10**-3
        params_SI['Q_e_m'] = p['Q_e_m'] * 10**-9           
    
    return params_SI

def get_syn_info(
    json_file_name: str | Path,
    idx: int = 0,
) -> tuple[Any, Any]:
    """Extract quantal conductances (Qe, Qi) as Brian2 quantities.

    Returns
    -------
    tuple[brian2.Quantity, brian2.Quantity]
        (Qe, Qi) in nanosiemens.
    """
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")
    with open(json_file_name, "r") as fh:
        data = json.load(fh)

    Qe = data[0][idx]["model"]["Q_e"] * b2.nS
    Qi = data[0][idx]["model"]["Q_i"] * b2.nS
    return Qe, Qi

def get_syn_info_general(
    json_file_name: str | Path,
    idx: int = 0,
) -> dict[str, Any]:
    """Quantal conductances for every synaptic channel, as Brian2 quantities.

    Works for both channel schemes (see ``ntmf.neurons.synaptic_channels_from_model``):
    numbered JSONs return ``{"1": Q_1, "2": Q_2, ...}``, classic e/i JSONs
    return ``{"E": Q_e, "I": Q_i}``. Values are in nanosiemens.
    """
    # local import: ntmf.neurons imports this module at load time
    from ntmf.neurons import synaptic_channels_from_model

    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")
    with open(json_file_name, "r") as fh:
        m = json.load(fh)[0][idx]["model"]

    return {c["name"]: m[c["Q_key"]] * b2.nS for c in synaptic_channels_from_model(m)}

# ---------------------------------------------------------------------------
# Network / input configuration
# ---------------------------------------------------------------------------

def get_input_config(
    json_file_name: str | Path,
) -> dict[str, Any]:
    """Load input configuration (connections, rates, units)."""
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name.")
    with open(json_file_name, "r") as fh:
        return json.load(fh)[0]


def get_network_config(
    json_file_name: str | Path,
) -> dict[str, Any]:
    """Load network configuration (composition, external_input, rates)."""
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name.")
    with open(json_file_name, "r") as fh:
        return json.load(fh)[0]

def get_TF_sweep_config(
    json_file_name: str | Path,
    channel_names: list[str],
) -> dict[str, Any]:
    """Load an N-channel TF sweep config and build the input samples.

    Expected JSON layout (one entry per synaptic channel ``k``)::

        "connections": {"N_external_k": ..., "conn_prob_k": ...},
        "channels":    {"k": {"background_freq": ..., "sweep": {"min": ..., "max": ..., "step": ...}}},
        "sampling":    {"mode": "grid"}  or  {"mode": "lhs", "n_samples": ..., "seed": ...}

    ``grid`` takes the Cartesian product of the per-channel ranges (last
    channel varies fastest, as in the 2-channel e/i scripts). ``lhs`` draws
    ``n_samples`` Latin-hypercube points in the per-channel [min, max] box and
    ignores ``step``.

    Parameters
    ----------
    json_file_name : str or Path
        Path to the sweep config.
    channel_names : list of str
        Channel names of the neuron model, in order (e.g. ``["1", "2", "3", "4"]``).
        The config must define exactly these channels.

    Returns
    -------
    dict with keys
        ``channels`` : {name: {"N_external", "conn_prob", "background_freq", "min", "max", "step"}}
        ``samples``  : ndarray, shape (n_samples, n_channels), rates in Hz
        ``sampling`` : the sampling block of the config
    """
    import itertools

    cfg = get_input_config(json_file_name)
    conn = cfg["connections"]
    ch_cfg = cfg["channels"]

    missing = [k for k in channel_names if k not in ch_cfg]
    extra = [k for k in ch_cfg if k not in channel_names]
    if missing or extra:
        raise ValueError(
            f"Sweep config channels {sorted(ch_cfg)} do not match the neuron channels "
            f"{list(channel_names)} (missing: {missing}, extra: {extra})."
        )

    channels: dict[str, dict[str, float]] = {}
    for k in channel_names:
        sweep = ch_cfg[k]["sweep"]
        channels[k] = {
            "N_external": int(conn[f"N_external_{k}"]),
            "conn_prob": float(conn[f"conn_prob_{k}"]),
            "background_freq": float(ch_cfg[k].get("background_freq", 0.0)),
            "min": float(sweep["min"]),
            "max": float(sweep["max"]),
            "step": float(sweep.get("step", 0.0)),
        }

    sampling = cfg.get("sampling", {"mode": "grid"})
    mode = sampling.get("mode", "grid")

    if mode == "grid":
        axes = []
        for k in channel_names:
            c = channels[k]
            if c["step"] <= 0 or c["max"] == c["min"]:
                axes.append(np.array([c["min"]]))
            else:
                # half-step margin so that max is included despite float rounding
                axes.append(np.round(np.arange(c["min"], c["max"] + 0.5 * c["step"], c["step"]), 10))
        samples = np.array(list(itertools.product(*axes)), dtype=float)
    elif mode == "lhs":
        from scipy.stats import qmc

        n_samples = int(sampling["n_samples"])
        seed = sampling.get("seed")
        try:  # scipy >= 1.15 uses `rng`; older versions only accept `seed`
            sampler = qmc.LatinHypercube(d=len(channel_names), rng=seed)
        except TypeError:
            sampler = qmc.LatinHypercube(d=len(channel_names), seed=seed)
        lo = np.array([channels[k]["min"] for k in channel_names])
        hi = np.array([channels[k]["max"] for k in channel_names])
        # manual scaling (qmc.scale rejects channels held fixed with min == max)
        samples = lo + sampler.random(n_samples) * (hi - lo)
    else:
        raise ValueError(f"Unknown sampling mode '{mode}'; use 'grid' or 'lhs'.")

    return {"channels": channels, "samples": samples, "sampling": sampling}

# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------

def adding_K_params(
    neuron_params: dict[str, float],
    network_config: dict[str, Any],
) -> dict[str, float]:
    """Add effective number of inputs K_e, K_i to *neuron_params* (in-place).

    K_e = N_external_exc * conn_prob
    K_i = N_external_inh * conn_prob
    """
    neuron_params["K_e"] = (
        network_config["external_input"]["N_external_exc"]
        * network_config["external_input"]["conn_prob"]
    )
    neuron_params["K_i"] = (
        network_config["external_input"]["N_external_inh"]
        * network_config["external_input"]["conn_prob"]
    )
    return neuron_params


# ---------------------------------------------------------------------------
# HDF5 spike data
# ---------------------------------------------------------------------------

def load_spike_data(fname: str | Path) -> dict[str, Any]:
    """Load simulation data from an HDF5 file.

    Returns dict with keys: network_composition, external_input, spikes, sim_duration.
    """
    data: dict[str, Any] = {}

    with h5py.File(fname, "r") as fh:
        # network composition
        data["network_composition"] = {
            k: fh["network_composition"][k][()] for k in fh["network_composition"]
        }

        # external input
        data["external_input"] = {
            k: fh["external_input"][k][()] for k in fh["external_input"]
        }

        # spikes
        data["sim_duration"] = fh["spikes"]["sim_duration"][()]
        spikes: dict[str, Any] = {}
        for pop in ("FS", "RS"):
            grp = fh["spikes"][pop]
            spikes[pop] = {"i": grp["i"][()], "t": grp["t"][()]}
        data["spikes"] = spikes

    return data

# -------------------- #
def load_sim_data(folder_path, model_type="NN"):
    """
    Parses h5 files in a folder and returns a list of results.
    model_type: "NN" (collects avg and std) or "MF" (collects avg)
    """
    results = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5") and "exc_" in filename:
            # Parse exc and inh values from filename: sim_data_exc_10_inh_5.h5
            parts = filename.replace('.h5', '').split('_')
            exc = float(parts[3])
            inh = float(parts[5])
            
            with h5py.File(os.path.join(folder_path, filename), "r") as f:
                # Extract stats for both populations
                res = {
                    'exc': exc, 
                    'inh': inh, 
                    'fs_avg': f['stats']['FS_avg_freq'][()],
                    'rs_avg': f['stats']['RS_avg_freq'][()]
                }
                
                # Extract Standard Deviations only if it's the NN model
                if model_type == "NN":
                    res['fs_std'] = f['stats']['FS_std_freq'][()]
                    res['rs_std'] = f['stats']['RS_std_freq'][()]
                
                results.append(res)
                
    return pd.DataFrame(results)