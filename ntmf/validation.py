"""MF-vs-NN validation: compare mean-field ODE simulations against neural-network spike data.

Provides utilities to:
1. Reconstruct time-varying driving inputs from saved HDF5 simulation metadata.
2. Bin neural-network spike data into population firing-rate traces.
3. Quantitatively compare MF and NN rate traces (RMSE, Pearson correlation).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from ntmf.config import adding_K_params, get_network_config, get_params_model_SI


# ---------------------------------------------------------------------------
# Driving-input reconstruction from HDF5
# ---------------------------------------------------------------------------

def build_driving_input_from_hdf5(
    hdf5_path: str,
    time: np.ndarray,
    network_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Reconstruct driving_input arrays from saved HDF5 simulation metadata.

    Reads the external-input stimulus definitions stored in the HDF5 file
    and builds the time-varying ``driving_input`` dictionary expected by
    :func:`~ntmf.meanfield.simulate_MF_FS_RS_ode`.

    The HDF5 stores *stimulus-specific* step inputs as separate datasets.
    A persistent background excitatory rate (from the network config's
    ``rates.background_freq``) is added on top.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 spike-data file.
    time : ndarray
        1-D time vector (seconds) at which to evaluate the driving input.
    network_config : dict, optional
        If provided, ``network_config["rates"]["background_freq"]`` is used
        for the persistent background excitation.  Falls back to 0.3 Hz.

    Returns
    -------
    dict
        ``{"excitatory": {"FS": ndarray, "RS": ndarray},
            "inhibitory": {"FS": ndarray, "RS": ndarray}}``
    """
    # Default background rate from the NN simulation protocol
    bg_freq = 0.3  # Hz
    if network_config is not None:
        bg_freq = network_config.get("rates", {}).get("background_freq", bg_freq)

    n = len(time)

    # --- read HDF5 metadata ---
    with h5py.File(hdf5_path, "r") as fh:
        ext = fh["external_input"]

        exc_FS_time = ext["exc_input_FS_time_s"][()]
        exc_FS_freq = float(ext["exc_input_FS_freq_Hz"][()])
        exc_RS_time = ext["exc_input_RS_time_s"][()]
        exc_RS_freq = float(ext["exc_input_RS_freq_Hz"][()])

        inh_FS_time = ext["inh_input_FS_time_s"][()]
        inh_FS_freq = float(ext["inh_input_FS_freq_Hz"][()])
        inh_RS_time = ext["inh_input_RS_time_s"][()]
        inh_RS_freq = float(ext["inh_input_RS_freq_Hz"][()])

    # --- build per-population, per-type step arrays ---
    exc_FS = np.full(n, bg_freq, dtype=float)
    exc_RS = np.full(n, bg_freq, dtype=float)
    inh_FS = np.zeros(n, dtype=float)
    inh_RS = np.zeros(n, dtype=float)

    # Excitatory stimulus to FS during [exc_FS_time[0], exc_FS_time[1]]
    mask = (time >= exc_FS_time[0]) & (time < exc_FS_time[1])
    exc_FS[mask] += exc_FS_freq

    # Excitatory stimulus to RS during [exc_RS_time[0], exc_RS_time[1]]
    mask = (time >= exc_RS_time[0]) & (time < exc_RS_time[1])
    exc_RS[mask] += exc_RS_freq

    # Inhibitory stimulus to FS during [inh_FS_time[0], inh_FS_time[1]]
    mask = (time >= inh_FS_time[0]) & (time < inh_FS_time[1])
    inh_FS[mask] += inh_FS_freq

    # Inhibitory stimulus to RS during [inh_RS_time[0], inh_RS_time[1]]
    mask = (time >= inh_RS_time[0]) & (time < inh_RS_time[1])
    inh_RS[mask] += inh_RS_freq

    return {
        "excitatory": {"FS": exc_FS, "RS": exc_RS},
        "inhibitory": {"FS": inh_FS, "RS": inh_RS},
    }


# ---------------------------------------------------------------------------
# NN population rate extraction
# ---------------------------------------------------------------------------

def compute_nn_population_rates(
    spike_data: dict[str, Any],
    N_FS: int,
    N_RS: int,
    sim_duration: float,
    bin_size: float = 0.1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bin NN spikes into population firing-rate traces (Hz).

    Parameters
    ----------
    spike_data : dict
        As returned by :func:`~ntmf.config.load_spike_data`.  Must contain
        ``spike_data["spikes"]["FS"]["i"]``, ``["t"]``, and the same for RS.
    N_FS : int
        Number of FS neurons.
    N_RS : int
        Number of RS neurons.
    sim_duration : float
        Total simulation duration (seconds).
    bin_size : float
        Width of each time bin (seconds).  Default 100 ms.

    Returns
    -------
    dict
        ``{"FS": (bin_centers, rates_Hz), "RS": (bin_centers, rates_Hz)}``
        where *bin_centers* are 1-D arrays of bin midpoints and *rates_Hz*
        are the average firing rates within each bin.
    """
    edges = np.arange(0, sim_duration + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for pop, N in [("FS", N_FS), ("RS", N_RS)]:
        counts, _ = np.histogram(
            spike_data["spikes"][pop]["t"], bins=edges,
        )
        # Rate = spikes / (N_neurons * bin_width)
        rates = counts / (N * bin_size)
        result[pop] = (centers, rates)

    return result


# ---------------------------------------------------------------------------
# MF-vs-NN quantitative comparison
# ---------------------------------------------------------------------------

def compare_mf_nn(
    mf_rates: dict[str, np.ndarray],
    mf_time: np.ndarray,
    nn_rates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Quantitative comparison between MF and NN rate traces.

    Interpolates MF rates at NN bin centres for direct comparison.

    Parameters
    ----------
    mf_rates : dict
        ``{"FS": ndarray, "RS": ndarray}`` — MF firing-rate traces (Hz).
    mf_time : ndarray
        Time vector corresponding to *mf_rates*.
    nn_rates : dict
        ``{"FS": (bin_centers, rates_Hz), "RS": (bin_centers, rates_Hz)}``
        as returned by :func:`compute_nn_population_rates`.

    Returns
    -------
    dict
        Keys: ``rmse_FS``, ``rmse_RS``, ``corr_FS``, ``corr_RS``,
        each a float.  Correlation is Pearson *r*.
    """
    stats: dict[str, Any] = {}

    for pop in ("FS", "RS"):
        nn_bins, nn_vals = nn_rates[pop]
        mf_interp = interp1d(
            mf_time, mf_rates[pop], kind="linear",
            bounds_error=False, fill_value=0.0,
        )
        mf_at_nn = mf_interp(nn_bins)

        # RMSE
        rmse = float(np.sqrt(np.mean((mf_at_nn - nn_vals) ** 2)))
        stats[f"rmse_{pop}"] = rmse

        # Pearson correlation
        if np.std(mf_at_nn) < 1e-12 or np.std(nn_vals) < 1e-12:
            stats[f"corr_{pop}"] = 0.0
        else:
            stats[f"corr_{pop}"] = float(pearsonr(mf_at_nn, nn_vals)[0])

    return stats


# ---------------------------------------------------------------------------
# Helper: build MF network config with Q_e / Q_i from HDF5 simulation
# ---------------------------------------------------------------------------

def build_mf_network_config(
    network_config_path: str,
    Q_e_nS: float = 1.5,
    Q_i_nS: float = 5.0,
) -> dict[str, Any]:
    """Load a network config JSON and augment it with Q_e / Q_i.

    The original ``network_config_file_v0.json`` does not contain quantal
    conductances for external inputs.  This function adds them so that the
    MF model can use them.

    Parameters
    ----------
    network_config_path : str
        Path to the JSON file (e.g. ``config/network_config_file_v0.json``).
    Q_e_nS : float
        Excitatory quantal conductance in nS.
    Q_i_nS : float
        Inhibitory quantal conductance in nS.

    Returns
    -------
    dict
        A deep copy of the loaded config with ``Q_e`` and ``Q_i`` added
        to ``external_input``.
    """
    cfg = get_network_config(network_config_path)
    cfg = deepcopy(cfg)
    cfg["external_input"]["Q_e"] = Q_e_nS
    cfg["external_input"]["Q_i"] = Q_i_nS
    return cfg
