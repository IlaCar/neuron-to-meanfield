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
