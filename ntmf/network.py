"""Network creation and spike-rate extraction utilities.

Provides functions to wire two-population (FS + RS) networks in Brian2
and to extract population firing rates from SpikeMonitor objects.
"""

from typing import Any, Optional

import brian2 as b2
import numpy as np


# ---------------------------------------------------------------------------
# Network creation
# ---------------------------------------------------------------------------

def network_creation(
    conn_prob: float,
    pop_1: b2.NeuronGroup,
    pop_2: b2.NeuronGroup,
    Qe_FS: b2.Quantity,
    Qi_FS: b2.Quantity,
    Qe_RS: b2.Quantity,
    Qi_RS: b2.Quantity,
    seed: Optional[int] = None,
) -> tuple[b2.Synapses, b2.Synapses, b2.Synapses, b2.Synapses]:
    """Create recurrent synapses for a two-population (FS, RS) network.

    Connectivity:
        S_11: pop_1 (FS) → pop_1 (FS), inhibitory, no self-connections
        S_12: pop_1 (FS) → pop_2 (RS), inhibitory
        S_21: pop_2 (RS) → pop_1 (FS), excitatory
        S_22: pop_2 (RS) → pop_2 (RS), excitatory, no self-connections

    Parameters
    ----------
    conn_prob : float
        Connection probability (0–1).
    pop_1, pop_2 : brian2.NeuronGroup
        The two populations (typically FS and RS).
    Qe_FS, Qi_FS, Qe_RS, Qi_RS : brian2.Quantity
        Quantal conductance increments for each population.
    seed : int or None
        Random seed for reproducible connectivity.

    Returns
    -------
    tuple of (S_11, S_12, S_21, S_22)
    """
    if seed is not None:
        b2.seed(seed)

    S_11 = b2.Synapses(pop_1, pop_1, on_pre="GsynI_post+=Qi_FS", name="S_11")
    S_11.connect("i!=j", p=conn_prob)

    S_12 = b2.Synapses(pop_1, pop_2, on_pre="GsynI_post+=Qi_RS", name="S_12")
    S_12.connect(p=conn_prob)

    S_21 = b2.Synapses(pop_2, pop_1, on_pre="GsynE_post+=Qe_FS", name="S_21")
    S_21.connect(p=conn_prob)

    S_22 = b2.Synapses(pop_2, pop_2, on_pre="GsynE_post+=Qe_RS", name="S_22")
    S_22.connect("i!=j", p=conn_prob)

    return S_11, S_12, S_21, S_22


# ---------------------------------------------------------------------------
# Rate extraction
# ---------------------------------------------------------------------------

def extracting_pop_freq_and_std(
    sim_duration: b2.Quantity,
    p_start: b2.Quantity,
    p_end: b2.Quantity,
    pop1: b2.SpikeMonitor,
    pop2: b2.SpikeMonitor,
    N_pop1: int,
    N_pop2: int,
    bin_size: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Extract mean and std firing rates for a two-population network.

    Parameters
    ----------
    bin_size : float
        Bin width in seconds.

    Returns
    -------
    (mean_rates, std_rates) : each is [FS_value, RS_value] in Hz.
    """
    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]

    spike_matrix_FS = _build_spike_matrix(pop1, N_pop1, time_bins, bin_size)
    spike_matrix_RS = _build_spike_matrix(pop2, N_pop2, time_bins, bin_size)

    mean_rate_FS = np.mean(spike_matrix_FS, axis=0)
    std_rate_FS = np.std(spike_matrix_FS, axis=0)
    mean_rate_RS = np.mean(spike_matrix_RS, axis=0)
    std_rate_RS = np.std(spike_matrix_RS, axis=0)

    # Select stimulation window
    if p_end == sim_duration - p_start:
        left_bound = int((p_start / bin_size).item())
        right_bound = -left_bound + 1
    else:
        left_bound = int((p_start / bin_size).item())
        right_bound = int((p_end / bin_size).item())

    mean_rates = [
        float(np.mean(mean_rate_FS[left_bound:right_bound])),
        float(np.mean(mean_rate_RS[left_bound:right_bound])),
    ]
    std_rates = [
        float(np.std(std_rate_FS[left_bound:right_bound])),
        float(np.std(std_rate_RS[left_bound:right_bound])),
    ]

    return mean_rates, std_rates


def extracting_single_pop_freq_and_std(
    sim_duration: b2.Quantity,
    p_start: b2.Quantity,
    p_end: b2.Quantity,
    pop: b2.SpikeMonitor,
    N_pop: int,
    bin_size: float = 0.1,
    delay: Optional[b2.Quantity] = None,
) -> tuple[float, float]:
    """Extract mean and std firing rate for a single population.

    Parameters
    ----------
    delay : brian2.Quantity or None
        Delay after p_start before counting spikes.

    Returns
    -------
    (mean_rate, std_rate) in Hz.
    """
    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]

    spike_matrix = _build_spike_matrix(pop, N_pop, time_bins, bin_size)
    mean_rate = np.mean(spike_matrix, axis=0)
    std_rate = np.std(spike_matrix, axis=0)

    if delay is not None:
        left_bound = int(((p_start + delay) / bin_size).item())
    else:
        left_bound = int((p_start / bin_size).item())
    right_bound = left_bound + int(((p_end - p_start) / bin_size).item()) + 1

    return float(np.mean(mean_rate[left_bound:right_bound])), float(np.std(std_rate[left_bound:right_bound]))


def _build_spike_matrix(
    pop: b2.SpikeMonitor,
    N_pop: int,
    time_bins: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    """Build a (n_neurons × n_bins) spike-count matrix, converted to Hz."""
    n_bins = len(time_bins)
    matrix = np.zeros((N_pop, n_bins))
    for i, t in zip(pop.i, pop.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < n_bins:
            matrix[i, bin_idx] += 1
    matrix /= bin_size  # counts → rate in Hz
    return matrix
