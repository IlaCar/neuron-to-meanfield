"""Network creation and spike-rate extraction utilities.

Provides functions to wire two-population (FS + RS) networks in Brian2
and to extract population firing rates from SpikeMonitor objects.

Networks with any number of populations and synaptic channels are built
from a network config + protocol config (see the last section of the file).

"""
import json
from pathlib import Path
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

def extracting_single_pop_freq_and_std_window(
    sim_duration: b2.Quantity,
    t_start: b2.Quantity,
    t_end: b2.Quantity,
    pop: b2.SpikeMonitor,
    N_pop: int,
    bin_size: float = 0.1,
) -> tuple[float, float]:
    """Mean and std firing rate of a single population inside [t_start, t_end).

    Same statistics as ``extracting_single_pop_freq_and_std`` (mean over time
    of the across-neuron mean rate; std over time of the across-neuron std),
    but only bins lying entirely inside the window are used. To discard a
    transient, pass ``t_start = p_start + delay``.

    Parameters
    ----------
    t_start, t_end : brian2.Quantity
        Averaging window.
    bin_size : float
        Bin width in seconds.

    Returns
    -------
    (mean_rate, std_rate) in Hz.
    """
    t0 = float(t_start / b2.second)
    t1 = float(t_end / b2.second)
    if t1 <= t0:
        raise ValueError(f"Empty averaging window [{t0}, {t1}) s.")

    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]

    spike_matrix = _build_spike_matrix(pop, N_pop, time_bins, bin_size)
    mean_rate = np.mean(spike_matrix, axis=0)
    std_rate = np.std(spike_matrix, axis=0)

    eps = 1e-9  # guard against float rounding at bin edges
    left_bound = int(np.ceil(t0 / bin_size - eps))
    right_bound = int(np.floor(t1 / bin_size + eps))
    if right_bound <= left_bound:
        raise ValueError(f"No complete {bin_size} s bin inside [{t0}, {t1}) s.")

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

# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def generate_ou_process(time, dt, mu, tau, sigma, x0=None):
    x = np.zeros_like(time)
    x[0] = x0 if x0 is not None else mu
    np.random.seed(0)
    
    # Pre-generate Gaussian noise for efficiency
    noise = np.random.normal(0, 1, len(time))
    
    for i in range(1, len(time)):
        dx = ((mu - x[i-1]) / tau) * dt + sigma * np.sqrt(dt) * noise[i]
        x[i] = x[i-1] + dx
        
    # Rectify to prevent negative firing rates
    return np.maximum(0, x)


# ---------------------------------------------------------------------------
# Config-driven network with any number of populations and synaptic channels
# ---------------------------------------------------------------------------
#
# Network config (config/network_config_<name>.json), first list element:
#
#     "populations":          {name: {"neuron_model": str, "N": int}}
#     "external_populations": {name: {"N": int}}                (Poisson sources)
#     "internal_connections": {"source->target": {"ch": k, "conn_prob": p}}
#     "external_connections": {"source->target": {"ch": k, "conn_prob": p}}
#
# Protocol config (config/input_config_<name>.json), first list element:
#
#     "external_populations": {name: {"background_freq": Hz,
#                                      "stimulations": [{"t_start": s, "t_end": s, "freq": Hz}]}}
#     "simulation":           {"sim_duration": s, "dt": ms}
#
# Channel numbers are local to the target: E_k, Q_k and tau_k are read from the
# target population's neuron JSON. Several connections may share a channel.

def _split_connection_key(key: str) -> tuple[str, str]:
    """'source->target' -> ('source', 'target')."""
    parts = [x.strip() for x in key.split("->")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Connection key '{key}' must have the form 'source->target'.")
    return parts[0], parts[1]


def all_connections(network_config: dict[str, Any]) -> dict[str, dict]:
    """Internal and external connections in one dict, keyed 'source->target'."""
    internal = network_config.get("internal_connections", {})
    external = network_config.get("external_connections", {})
    shared = set(internal) & set(external)
    if shared:
        raise ValueError(f"Connections listed both as internal and external: {sorted(shared)}")
    return {**internal, **external}


def load_neuron_model(
    neuron_model: str,
    model_folder: str | Path,
    idx: int = 0,
) -> dict[str, Any]:
    """Entry ``idx`` of ``<model_folder>/<neuron_model>.json`` (metainfo, model, init, units)."""
    with open(Path(model_folder) / f"{neuron_model}.json", "r") as fh:
        return json.load(fh)[0][idx]


def check_network_config(
    network_config: dict[str, Any],
    model_folder: str | Path,
    input_config: Optional[dict[str, Any]] = None,
    idx: int = 0,
) -> None:
    """Raise ``ValueError`` if the network (and protocol) configs are inconsistent.

    Checks that every connection names known populations, that internal
    sources are populations and external sources are external populations,
    that the channel exists in the target's neuron JSON, that 0 <= conn_prob <= 1,
    that population names are distinct from each other and from the neuron
    model names of other populations (Brian2 names the NeuronGroup after its
    neuron model), and, if *input_config* is given, that every external
    population has a protocol entry and vice versa.
    """
    from ntmf.neurons import synaptic_channels_from_model  # local import, as in config.py

    pops = network_config["populations"]
    ext = network_config.get("external_populations", {})
    errors = []

    if set(pops) & set(ext):
        errors.append(f"Names used both as population and external population: {sorted(set(pops) & set(ext))}")
    models = [p["neuron_model"] for p in pops.values()]
    if len(set(models)) != len(models):
        errors.append("Two populations use the same neuron_model; Brian2 group names would clash.")

    channels = {}
    for name, p in pops.items():
        m = load_neuron_model(p["neuron_model"], model_folder, idx)["model"]
        channels[name] = [c["name"] for c in synaptic_channels_from_model(m)]

    for kind, conns, src_pool in (
        ("internal_connections", network_config.get("internal_connections", {}), pops),
        ("external_connections", network_config.get("external_connections", {}), ext),
    ):
        for key, c in conns.items():
            src, tgt = _split_connection_key(key)
            if src not in src_pool:
                errors.append(f"{kind}['{key}']: unknown source '{src}'.")
            if tgt not in pops:
                errors.append(f"{kind}['{key}']: unknown target population '{tgt}'.")
                continue
            if str(c["ch"]) not in channels[tgt]:
                errors.append(f"{kind}['{key}']: channel {c['ch']} not in {pops[tgt]['neuron_model']} "
                              f"(channels {channels[tgt]}).")
            if not 0.0 <= float(c["conn_prob"]) <= 1.0:
                errors.append(f"{kind}['{key}']: conn_prob {c['conn_prob']} outside [0, 1].")

    if input_config is not None:
        proto = input_config.get("external_populations", {})
        if set(proto) != set(ext):
            errors.append(f"Protocol / network external populations differ: "
                          f"only in protocol {sorted(set(proto) - set(ext))}, "
                          f"only in network {sorted(set(ext) - set(proto))}.")

    if errors:
        raise ValueError("Network config problems:\n  " + "\n  ".join(errors))


def channel_in_degrees(
    network_config: dict[str, Any],
    by_source: bool = False,
) -> dict[str, dict]:
    """Expected in-degree per target population and channel.

    K_ch = sum over the sources projecting onto that channel of N_src * conn_prob.

    Returns
    -------
    ``{target: {ch: K}}``, or with ``by_source=True``
    ``{target: {ch: {source: K_src}}}``. Channel keys are strings ("1", "2", ...).
    Channels of a population that receive no connection do not appear.
    """
    sizes = {**{k: v["N"] for k, v in network_config.get("external_populations", {}).items()},
             **{k: v["N"] for k, v in network_config["populations"].items()}}
    out: dict[str, dict] = {}
    for key, c in all_connections(network_config).items():
        src, tgt = _split_connection_key(key)
        ch = str(c["ch"])
        out.setdefault(tgt, {}).setdefault(ch, {})
        out[tgt][ch][src] = out[tgt][ch].get(src, 0.0) + sizes[src] * float(c["conn_prob"])
    if by_source:
        return out
    return {t: {ch: float(sum(d.values())) for ch, d in chs.items()} for t, chs in out.items()}


def external_rate_trace(
    ext_spec: dict[str, Any],
    times_s: np.ndarray,
) -> np.ndarray:
    """Rate (Hz) of one external population on *times_s* (seconds).

    Background plus the ``freq`` of every stimulation window [t_start, t_end)
    containing t; overlapping windows add (same rule as ``ntmf.meanfield_nchannels``).
    """
    rate = np.full(len(times_s), float(ext_spec.get("background_freq", 0.0)))
    for stim in ext_spec.get("stimulations", []):
        mask = (times_s >= stim["t_start"]) & (times_s < stim["t_end"])
        rate[mask] += float(stim["freq"])
    return rate


def protocol_epochs(input_config: dict[str, Any]) -> list[tuple[float, float]]:
    """Intervals (s) between consecutive stimulation on/offsets, covering the whole simulation."""
    T = float(input_config["simulation"]["sim_duration"])
    edges = {0.0, T}
    for spec in input_config["external_populations"].values():
        for stim in spec.get("stimulations", []):
            edges.update(x for x in (float(stim["t_start"]), float(stim["t_end"])) if 0.0 < x < T)
    edges = sorted(edges)
    return list(zip(edges[:-1], edges[1:]))


def build_network_from_config(
    network_config: dict[str, Any],
    input_config: dict[str, Any],
    model_folder: str | Path,
    curr_inj: Any = None,
    seed: Optional[int] = None,
    idx: int = 0,
    rate_dt: b2.Quantity = 1 * b2.ms,
) -> dict[str, Any]:
    """Build a Brian2 network from a network config and a protocol config.

    Every population is created with ``ntmf.neurons.create_AdEx_neuron`` from
    ``<model_folder>/<neuron_model>.json``. Each external population is a
    ``PoissonGroup`` whose rate follows its protocol entry (``external_rate_trace``).
    Each connection 'source->target' becomes one ``Synapses`` object (Brian2
    name 'source_to_target') adding the target's Q_k to its channel-k
    conductance ``Gsyn<k>``; a population projecting onto itself gets no
    self-connections (as in ``network_creation``).

    All objects are collected in an explicit ``b2.Network``: Brian2's default
    ``b2.run()`` only finds objects held in plain variables, not in dicts, and
    would otherwise silently simulate nothing. Add monitors with
    ``net.add(...)`` and run with ``net.run(duration, namespace=built["namespace"])``.

    Parameters
    ----------
    curr_inj : brian2.TimedArray or None
        Injected current shared by all populations (``Is = current(t)``).
        None gives zero current.
    seed : int or None
        Passed to ``b2.seed`` before connecting and creating the Poisson inputs.
    rate_dt : brian2.Quantity
        Time step of the external rate TimedArrays.

    Returns
    -------
    dict with keys ``network`` (b2.Network), ``populations`` {name: NeuronGroup},
    ``external`` {name: PoissonGroup}, ``synapses`` {'source->target': Synapses},
    ``namespace`` (to pass to ``net.run``), ``sim_duration`` and ``dt`` (Quantities).
    """
    from ntmf.neurons import create_AdEx_neuron  # local import: neurons imports config

    check_network_config(network_config, model_folder, input_config, idx)

    sim_duration = float(input_config["simulation"]["sim_duration"]) * b2.second
    dt = float(input_config["simulation"]["dt"]) * b2.ms
    b2.defaultclock.dt = dt
    if seed is not None:
        b2.seed(seed)

    if curr_inj is None:
        curr_inj = b2.TimedArray(np.zeros(2) * b2.amp, dt=sim_duration)
    namespace: dict[str, Any] = {"current": curr_inj}

    populations = {}
    for name, p in network_config["populations"].items():
        populations[name] = create_AdEx_neuron(
            p["neuron_model"], Path(model_folder) / f"{p['neuron_model']}.json", idx, int(p["N"]), curr_inj
        )

    times_s = np.arange(0.0, float(sim_duration / b2.second), float(rate_dt / b2.second))
    external = {}
    for name, e in network_config.get("external_populations", {}).items():
        rate = b2.TimedArray(external_rate_trace(input_config["external_populations"][name], times_s) * b2.Hz,
                             dt=rate_dt)
        namespace[f"rate_{name}"] = rate
        external[name] = b2.PoissonGroup(int(e["N"]), rates=f"rate_{name}(t)",
                                         namespace={f"rate_{name}": rate}, name=f"P_{name}")

    synapses = {}
    for key, c in all_connections(network_config).items():
        src, tgt = _split_connection_key(key)
        ch = str(c["ch"])
        m = load_neuron_model(network_config["populations"][tgt]["neuron_model"], model_folder, idx)["model"]
        source = populations[src] if src in populations else external[src]
        S = b2.Synapses(source, populations[tgt], on_pre=f"Gsyn{ch}_post += Q_syn",
                        namespace={"Q_syn": m[f"Q_{ch}"] * b2.nS}, name=f"{src}_to_{tgt}")
        if src == tgt:
            S.connect("i != j", p=float(c["conn_prob"]))
        else:
            S.connect(p=float(c["conn_prob"]))
        synapses[key] = S

    net = b2.Network(*populations.values(), *external.values(), *synapses.values())
    return {"network": net, "populations": populations, "external": external, "synapses": synapses,
            "namespace": namespace, "sim_duration": sim_duration, "dt": dt}


def realized_in_degrees(built: dict[str, Any]) -> dict[str, np.ndarray]:
    """Actual number of afferents of every target neuron, per connection.

    Returns ``{'source->target': array of length N_target}``.
    """
    out = {}
    for key, S in built["synapses"].items():
        out[key] = np.bincount(np.asarray(S.j[:]), minlength=len(S.target))
    return out


def population_rates(
    spike_mon: b2.SpikeMonitor,
    N_pop: int,
    sim_duration: b2.Quantity,
    bin_size: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Population-averaged rate (Hz) in bins of *bin_size* seconds.

    Returns (bin centres in s, rate in Hz).
    """
    edges = np.arange(0.0, float(sim_duration / b2.second) + bin_size / 2, bin_size)
    counts, _ = np.histogram(np.asarray(spike_mon.t / b2.second), bins=edges)
    return edges[:-1] + bin_size / 2, counts / (N_pop * bin_size)


def epoch_mean_rates(
    spike_mons: dict[str, b2.SpikeMonitor],
    sizes: dict[str, int],
    epochs: list[tuple[float, float]],
    skip: float = 0.2,
) -> dict[str, list[float]]:
    """Mean rate (Hz) of each population in every epoch, ignoring the first *skip* s.

    Returns ``{population: [rate in epoch 0, rate in epoch 1, ...]}``.
    """
    out = {}
    for name, mon in spike_mons.items():
        t = np.asarray(mon.t / b2.second)
        rates = []
        for t0, t1 in epochs:
            a = t0 + skip
            if a >= t1:
                raise ValueError(f"skip = {skip} s leaves nothing of epoch [{t0}, {t1}) s.")
            rates.append(float(np.sum((t >= a) & (t < t1)) / (sizes[name] * (t1 - a))))
        out[name] = rates
    return out

# ---------------------------------------------------------------------------
# TF sweep for one population of a config-driven network
# ---------------------------------------------------------------------------
#
# Used by transfer_function/data_extraction/TF_data_extraction_network.py.
# Sweep config (config/input_config_TF_<name>.json), first list element:
#
#     "network_config": file name, relative to the sweep config
#     "protocol":       {"sim_duration": s, "p_start": s, "p_end": s, "bin_size": s,
#                        "dt": ms, "N_cell": int, "seed": int}
#     "populations":    {name: {"delay": s, "channels": {k: {"min", "max", "step", "power"}},
#                               "sampling": {"mode": "grid"} or {"mode": "lhs", "n_samples", "seed"}}}

def get_TF_sweep_config_network(
    json_file_name: str | Path,
    population: str,
    channel_names: list[str],
    network_config: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Load the TF sweep of one population of a network and build the input samples.

    Network counterpart of :func:`get_TF_sweep_config`. The number of inputs
    per channel is not set in the sweep config: it is taken from the network
    config, so the TF is extracted with the in-degrees the population has in
    the network (:func:`channel_in_degrees`). Each channel is swept at the merged rate
    ``nu_eff = sum_src K_src * nu_src / K_ch``.

    Expected JSON layout::

        "network_config": "network_config_4_pops.json",   (relative to this file)
        "protocol": {"sim_duration": 7, "p_start": 2, "p_end": 5, "bin_size": 0.25,
                     "dt": 0.1, "N_cell": 5, "seed": 12345},
        "populations": {
            "<pop>": {
                "delay": 1.0,
                "channels": {"k": {"min": ..., "max": ..., "step": ..., "power": 1.0}, ...},
                "sampling": {"mode": "grid"}  or  {"mode": "lhs", "n_samples": ..., "seed": ...}
            }, ...
        }

    ``grid`` takes the Cartesian product of the per-channel ranges (last
    channel varies fastest). ``lhs`` draws ``n_samples`` Latin-hypercube points
    and ignores ``step``; the optional per-channel ``power`` (default 1) maps
    the unit coordinate u to ``min + (max - min) * u**power``, so ``power`` > 1
    puts more samples at low rates.

    Parameters
    ----------
    json_file_name : str or Path
        Path to the sweep config.
    population : str
        Population name, as in the network config.
    channel_names : list of str
        Channel names of the population's neuron model, in order. Channels
        that receive no connection in the network are not swept (rate 0).
    network_config : dict, str or Path, optional
        Network config (first list element) or its path; overrides the
        ``network_config`` entry of the sweep config.

    Returns
    -------
    dict with keys
        ``channels`` : {name: {"K", "sources", "background_freq", "min", "max", "step", "power"}},
                   ``sources`` = [{"source", "N", "conn_prob", "K"}, ...] (from ``channel_in_degrees``),
                       swept channels only
        ``swept``    : list of swept channel names (columns of ``samples``)
        ``unused``   : model channels without inputs in the network
        ``samples``  : ndarray, shape (n_samples, len(swept)), nu_eff in Hz
        ``sampling``, ``protocol``, ``delay``,
        ``network_config_file`` (None if a dict was passed)
    """
    import itertools

    json_file_name = Path(json_file_name)
    with open(json_file_name, "r") as fh:
        cfg = json.load(fh)[0]
    network_config_file = None
    if network_config is None:
        network_config = json_file_name.parent / cfg["network_config"]
    if not isinstance(network_config, dict):
        network_config_file = str(network_config)
        with open(network_config, "r") as fh:
            network_config = json.load(fh)[0]
    net_cfg = network_config

    if population not in cfg["populations"]:
        raise ValueError(f"Population '{population}' not in the sweep config ({list(cfg['populations'])}).")
    pop_cfg = cfg["populations"][population]
    ch_cfg = {str(k): v for k, v in pop_cfg["channels"].items()}

    if population not in net_cfg["populations"]:
        raise ValueError(f"Population '{population}' not in the network config ({list(net_cfg['populations'])}).")
    sizes = {**{k: v["N"] for k, v in net_cfg.get("external_populations", {}).items()},
             **{k: v["N"] for k, v in net_cfg["populations"].items()}}
    K_src = channel_in_degrees(net_cfg, by_source=True).get(population, {})
    if not K_src:
        raise ValueError(f"No connections target population '{population}'.")
    # per source: N and the conn_prob that gives its K (conn_prob of the config, summed if listed twice)
    sources = {ch: [{"source": s, "N": int(sizes[s]), "conn_prob": K / sizes[s], "K": K}
                    for s, K in d.items()]
               for ch, d in K_src.items()}
    not_in_model = [k for k in sources if k not in channel_names]
    if not_in_model:
        raise ValueError(f"Network connects to channels {not_in_model} of '{population}', "
                         f"but its neuron model only has channels {list(channel_names)}.")
    swept = [k for k in channel_names if k in sources]
    unused = [k for k in channel_names if k not in sources]
    if sorted(ch_cfg, key=int) != swept:
        raise ValueError(f"Sweep config channels {sorted(ch_cfg, key=int)} of '{population}' do not match "
                         f"the channels it receives in the network {swept}.")

    channels: dict[str, dict[str, Any]] = {}
    for k in swept:
        c = ch_cfg[k]
        channels[k] = {
            "K": float(sum(s["K"] for s in sources[k])),
            "sources": sources[k],
            "background_freq": 0.0,   # every input of the network is in nu_eff
            "min": float(c["min"]),
            "max": float(c["max"]),
            "step": float(c.get("step", 0.0)),
            "power": float(c.get("power", 1.0)),
        }
        if channels[k]["min"] < 0 or channels[k]["max"] < channels[k]["min"]:
            raise ValueError(f"Channel {k} of '{population}': need 0 <= min <= max.")

    sampling = pop_cfg.get("sampling", {"mode": "grid"})
    mode = sampling.get("mode", "grid")
    lo = np.array([channels[k]["min"] for k in swept])
    hi = np.array([channels[k]["max"] for k in swept])

    if mode == "grid":
        axes = []
        for k in swept:
            c = channels[k]
            if c["step"] <= 0 or c["max"] == c["min"]:
                axes.append(np.array([c["min"]]))
            else:
                axes.append(np.round(np.arange(c["min"], c["max"] + 0.5 * c["step"], c["step"]), 10))
        samples = np.array(list(itertools.product(*axes)), dtype=float)
    elif mode == "lhs":
        from scipy.stats import qmc

        n_samples = int(sampling["n_samples"])
        seed = sampling.get("seed")
        try:  # scipy >= 1.15 uses `rng`; older versions only accept `seed`
            sampler = qmc.LatinHypercube(d=len(swept), rng=seed)
        except TypeError:
            sampler = qmc.LatinHypercube(d=len(swept), seed=seed)
        power = np.array([channels[k]["power"] for k in swept])
        samples = lo + sampler.random(n_samples) ** power * (hi - lo)
        samples = np.round(samples, 4)   # exact in the %.6g .dat rows below 100 Hz; keeps --resume consistent
    else:
        raise ValueError(f"Unknown sampling mode '{mode}'; use 'grid' or 'lhs'.")

    return {"channels": channels, "swept": swept, "unused": unused, "samples": samples,
            "sampling": sampling, "protocol": cfg["protocol"],
            "delay": float(pop_cfg.get("delay", 0.0)),
            "network_config_file": network_config_file}