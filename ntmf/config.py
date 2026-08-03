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
IMPLEMENTED_NEURON_MODELS = ["FS", "RS", "RS_no_adapt", "GoC"]


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