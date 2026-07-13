"""Quasi-random AdEx parameter search.

Bottom-up screening of AdEx (adaptive exponential integrate-and-fire)
single-neuron parameters against experimental spike features. The pipeline is:

1. Define searched vs. fixed parameters and their bounds.
2. Draw a low-discrepancy (Sobol) sample from the searched parameter space.
3. Build one AdEx parameter set per sample and simulate a current-injection
   protocol (see :mod:`AdEx_search_helper`).
4. Score each candidate by a weighted relative error over spike features.
5. Rank candidates, plot the best, and save models/errors to JSON.

This is a sampling-based *search*, not a local optimisation: candidates are
drawn independently and scored, with no gradient or iterative refinement.
The transfer-function fitting in :mod:`ntmf.optimization` is a separate stage.

Units follow the PyNN convention (see :mod:`AdEx_search_helper`).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np
from scipy.stats import qmc

from AdEx_search_helper import extract_spike_features


# ---------------------------------------------------------------------------
# Default parameter bounds
# ---------------------------------------------------------------------------
# A tuple ``(lo, hi)`` marks a parameter to search; a scalar fixes it.
# Units are the PyNN standard.

DEFAULT_BOUNDS: dict[str, Any] = {
    "cm_bound":         (0.1, 0.2),    # nF - membrane capacitance
    "gl_bound":         (0.02, 0.05),  # uS - leak conductance
    "v_rest_bound":     -60,           # mV - leak reversal potential
    "i_offset_bound":   0,             # nA - constant external input current
    "a_bound":          (0, 10),       # nS - subthreshold adaptation
    "b_bound":          (0, 0.3),      # nA - spike-triggered adaptation
    "tau_w_bound":      (0, 200),      # ms - adaptation time constant
    "v_thresh_bound":   (-45, -43),    # mV - spike initiation threshold
    "delta_T_bound":    (2, 8),        # mV - slope factor
    "v_reset_bound":    (-53, -50),    # mV - reset potential after a spike
    "v_spike_bound":    17,            # mV - spike detection threshold
    "tau_refrac_bound": (1, 3),        # ms - refractory period
    "v_0_bound":        -60,           # mV - initial membrane potential
    "w_0_bound":        0,             # nA - initial adaptation current
}


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

def parse_param_bounds(
    bounds: dict[str, Any],
) -> tuple[list[str], list[tuple[float, float]], list[str], list[float]]:
    """Split parameter specifications into searched and fixed sets.

    Each key is expected to end in ``_bound`` (the suffix is stripped to
    recover the parameter name). A tuple value ``(lo, hi)`` marks a parameter
    to search; any scalar value fixes it.

    Parameters
    ----------
    bounds : dict
        Mapping ``{"<name>_bound": (lo, hi) | value}``.

    Returns
    -------
    searched_params : list of str
        Names of the parameters to search over.
    intervals : list of (float, float)
        Search interval for each entry in *searched_params*.
    fixed_names : list of str
        Names of the fixed parameters.
    fixed_values : list of float
        Value of each fixed parameter.
    """
    searched_params: list[str] = []
    intervals: list[tuple[float, float]] = []
    fixed_names: list[str] = []
    fixed_values: list[float] = []

    for arg, value in bounds.items():
        name = arg.removesuffix("_bound")
        if isinstance(value, tuple):
            intervals.append(value)
            searched_params.append(name)
        else:
            fixed_names.append(name)
            fixed_values.append(value)

    return searched_params, intervals, fixed_names, fixed_values


# ---------------------------------------------------------------------------
# AdEx parameter construction
# ---------------------------------------------------------------------------

def build_adex_params(
    cm: Optional[float] = None,
    gl: Optional[float] = None,
    v_rest: Optional[float] = None,
    i_offset: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    tau_w: Optional[float] = None,
    v_thresh: Optional[float] = None,
    delta_T: Optional[float] = None,
    v_reset: Optional[float] = None,
    v_spike: Optional[float] = None,
    tau_refrac: Optional[float] = None,
    v_0: Optional[float] = None,
    w_0: Optional[float] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Assemble an AdEx parameter set and its initial values.

    The subthreshold adaptation ``a`` is converted from nS to uS
    (``a * 1e-3``) to match the units expected by :mod:`AdEx_search_helper`; all other
    quantities are passed through unchanged.

    Returns
    -------
    params : dict
        AdEx model parameters keyed as in :mod:`AdEx_search_helper`.
    initial_values : dict
        ``{"v": v_0, "w": w_0}``.
    """
    params = {
        "C_m":     cm,                        # nF - membrane capacitance
        "g_L":     gl,                        # uS - leak conductance
        "E_L":     v_rest,                    # mV - leak reversal potential
        "I_e":     i_offset,                  # nA - constant external input
        "a":       np.round(a * 1e-3, 3),     # nS -> uS - subthresh. adaptation
        "b":       b,                         # nA - spike-triggered adaptation
        "tau_w":   tau_w,                     # ms - adaptation time constant
        "V_th":    v_thresh,                  # mV - spike threshold
        "Delta_T": delta_T,                   # mV - slope factor
        "V_reset": v_reset,                   # mV - reset potential
        "V_peak":  v_spike,                   # mV - spike detection threshold
        "t_ref":   tau_refrac,                # ms - refractory period
    }
    initial_values = {"v": v_0, "w": w_0}
    return params, initial_values


# ---------------------------------------------------------------------------
# Quasi-random sampling
# ---------------------------------------------------------------------------

def sample_parameter_space(
    intervals: list[tuple[float, float]],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Draw a scrambled Sobol sample and scale it to *intervals*.

    Parameters
    ----------
    intervals : list of (float, float)
        Search interval per dimension.
    n_samples : int
        Number of samples to draw (a power of two is recommended for Sobol
        balance properties).
    seed : int
        Seed for the underlying random generator.

    Returns
    -------
    ndarray of shape (n_samples, len(intervals))
        Samples scaled to the intervals and rounded to 3 decimals.
    """
    rng = np.random.default_rng(seed)
    dim = len(intervals)

    engine = qmc.Sobol(d=dim, scramble=True, seed=rng)
    sobol_samples = engine.random(n=n_samples)

    scaled_samples = np.zeros_like(sobol_samples)
    for i, (lo, hi) in enumerate(intervals):
        scaled_samples[:, i] = np.round(lo + sobol_samples[:, i] * (hi - lo), 3)

    return scaled_samples


def build_model_set(
    scaled_samples: np.ndarray,
    searched_params: list[str],
    fixed_params: dict[str, float],
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Build one ``(params, initial_values)`` model per sample.

    Parameters
    ----------
    scaled_samples : ndarray
        Output of :func:`sample_parameter_space`.
    searched_params : list of str
        Column names for *scaled_samples*.
    fixed_params : dict
        Fixed parameter name -> value.

    Returns
    -------
    list of (params, initial_values)
    """
    models = []
    for row in scaled_samples:
        searched = dict(zip(searched_params, row))
        models.append(build_adex_params(**{**searched, **fixed_params}))
    return models


# ---------------------------------------------------------------------------
# Current-injection protocol
# ---------------------------------------------------------------------------

def build_current_protocols(
    amps: list[float],
    time: np.ndarray,
    stim_delay: float,
    stim_duration: float,
    holding_current: float,
) -> list[np.ndarray]:
    """Build one step-current protocol (pA) per amplitude.

    Each protocol holds at *holding_current* before and after a step to *amp*
    that starts at *stim_delay* steps and lasts *stim_duration* steps.

    Returns
    -------
    list of ndarray
        One protocol vector (length ``len(time)``) per amplitude.
    """
    protocols = []
    start = int(stim_delay)
    stop = int(stim_delay + stim_duration)
    for amp in amps:
        protocol = np.zeros(len(time))
        protocol[:start] = holding_current
        protocol[start:stop] = amp
        protocol[stop:] = holding_current
        protocols.append(protocol)
    return protocols


# ---------------------------------------------------------------------------
# Feature-error scoring
# ---------------------------------------------------------------------------

def compute_feature_error(
    data_current: list[float],
    data_freq: list[float],
    model_freq: np.ndarray,
    data_inv_first_ISI: list[float],
    model_inv_first_ISI: np.ndarray,
    data_inv_last_ISI: list[float],
    model_inv_last_ISI: np.ndarray,
    data_time_to_first_spike: list[float],
    model_time_to_first_spike: np.ndarray,
    data_time_to_second_spike: list[float],
    model_time_to_second_spike: np.ndarray,
    data_time_to_third_spike: list[float],
    model_time_to_third_spike: np.ndarray,
    data_time_to_last_spike: list[float],
    model_time_to_last_spike: np.ndarray,
    *,
    weights: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted relative error between model and target spike features.

    For each model and current step the error accumulates the relative
    mismatch of the firing rate and, progressively (gated by the target rate),
    the spike latencies and inverse ISIs. A missing model feature (``NaN``)
    contributes a fixed penalty of 3.

    Parameters
    ----------
    data_* : list of float
        Target feature values, one per current step.
    model_* : ndarray of shape (n_models, n_currents)
        Model feature values.
    weights : dict, optional
        Per-feature weights. Missing keys default to 1. Recognised keys:
        ``freq, first_spike, second_spike, third_spike, last_spike,
        inv_first_ISI, inv_last_ISI``.

    Returns
    -------
    err : ndarray of shape (n_models, n_currents)
        Per-current error contributions.
    err_tot : ndarray of shape (n_models,)
        Error summed over current steps.
    """
    w = {
        "freq": 1.0, "first_spike": 1.0, "second_spike": 1.0,
        "third_spike": 1.0, "last_spike": 1.0,
        "inv_first_ISI": 1.0, "inv_last_ISI": 1.0,
    }
    if weights:
        w.update(weights)

    n_samples = model_freq.shape[0]
    n_currents = len(data_current)

    err = np.zeros((n_samples, n_currents))
    err_tot = np.zeros(n_samples)

    for i in range(n_samples):
        for j in range(n_currents):
            if abs(data_freq[j]) != 0:
                # Relative error on firing rate.
                err[i, j] = (abs((abs(data_freq[j]) - abs(model_freq[i, j]))
                                 / abs(data_freq[j]))) * w["freq"]

                if math.isnan(model_time_to_first_spike[i, j]):
                    err[i, j] += 3
                else:
                    err[i, j] += (abs(data_time_to_first_spike[j]
                                      - model_time_to_first_spike[i, j])
                                  / data_time_to_first_spike[j]) * w["first_spike"]

                if abs(data_freq[j]) > 1:
                    if math.isnan(model_inv_first_ISI[i, j]):
                        err[i, j] += 3
                    else:
                        err[i, j] += (abs(abs(data_inv_first_ISI[j])
                                          - abs(model_inv_first_ISI[i, j]))
                                      / abs(data_inv_first_ISI[j])) * w["inv_first_ISI"]

                    if math.isnan(model_time_to_second_spike[i, j]):
                        err[i, j] += 3
                    else:
                        err[i, j] += (abs(data_time_to_second_spike[j]
                                          - model_time_to_second_spike[i, j])
                                      / data_time_to_second_spike[j]) * w["second_spike"]

                if abs(data_freq[j]) > 2:
                    if math.isnan(model_inv_last_ISI[i, j]):
                        err[i, j] += 3
                    else:
                        err[i, j] += (abs(abs(data_inv_last_ISI[j])
                                          - abs(model_inv_last_ISI[i, j]))
                                      / abs(data_inv_last_ISI[j])) * w["inv_last_ISI"]

                    if math.isnan(model_time_to_third_spike[i, j]):
                        err[i, j] += 3
                    else:
                        err[i, j] += (abs(data_time_to_third_spike[j]
                                          - model_time_to_third_spike[i, j])
                                      / data_time_to_third_spike[j]) * w["third_spike"]

                    if math.isnan(model_time_to_last_spike[i, j]):
                        err[i, j] += 3
                    else:
                        err[i, j] += (abs(data_time_to_last_spike[j]
                                          - model_time_to_last_spike[i, j])
                                      / data_time_to_last_spike[j]) * w["last_spike"]
            else:
                # Absolute rate error when the target rate is zero.
                err[i, j] = abs(abs(data_freq[j]) - abs(model_freq[i, j]))

        err_tot[i] = np.sum(err[i])

    return err, err_tot


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def append_to_json(file_path: str, data: Any) -> None:
    """Append *data* to a JSON list on disk, creating the file if needed."""
    if os.path.exists(file_path):
        with open(file_path, "r") as fh:
            existing_data = json.load(fh)
    else:
        existing_data = []

    existing_data.append(data)

    with open(file_path, "w") as fh:
        json.dump(existing_data, fh, indent=4)


def load_target_features(
    data_file: str,
    indices: list[int],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Load experimental spike features and the stimulation protocol.

    Parameters
    ----------
    data_file : str
        Path to the extracted-features JSON.
    indices : list of int
        Which current steps to keep as search targets.

    Returns
    -------
    targets : dict
        Selected feature lists (``current``, ``freq``, ``inv_first_ISI``, ...).
    protocol : dict
        Stimulation-protocol fields (``Time``, ``delay``, ``duration``,
        ``holding_current``) in ms / pA.
    """
    with open(data_file, "r") as fh:
        exp_data = json.load(fh)

    keys = [
        ("current", "current"),
        ("freq", "mean_frequency"),
        ("inv_first_ISI", "inv_first_ISI"),
        ("inv_last_ISI", "inv_last_ISI"),
        ("time_to_first_spike", "time_to_first_spike"),
        ("time_to_second_spike", "time_to_second_spike"),
        ("time_to_third_spike", "time_to_third_spike"),
        ("time_to_last_spike", "time_to_last_spike"),
    ]
    targets = {
        short: [exp_data[full][i] for i in indices] for short, full in keys
    }

    stim = exp_data["stimulation_protocol"]
    delay = stim[0][1]
    protocol = {
        "Time": stim[0][3],                  # ms - total protocol duration
        "delay": delay,                      # ms - stimulus onset
        "duration": stim[0][2] - delay,      # ms - stimulus duration
        "holding_current": stim[1][0],       # pA - holding current
    }
    return targets, protocol


# ---------------------------------------------------------------------------
# Simulation sweep
# ---------------------------------------------------------------------------

def simulate_models(
    models: list[tuple[dict[str, float], dict[str, float]]],
    current_protocols: list[np.ndarray],
    dt: float,
    time: np.ndarray,
    stim_delay: float,
    stim_duration: float,
    *,
    parallel: bool = True,
    num_threads: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Run the feature extraction for every model, serially or in parallel.

    Returns
    -------
    dict
        Feature name -> ndarray of shape ``(n_models, n_currents)``. Keys:
        ``current, volt_stimend, freq, inv_first_ISI, inv_last_ISI,
        time_to_first_spike, time_to_second_spike, time_to_third_spike,
        time_to_last_spike``.
    """
    n_models = len(models)
    n_currents = len(current_protocols)
    fields = [
        "current", "volt_stimend", "freq", "inv_first_ISI", "inv_last_ISI",
        "time_to_first_spike", "time_to_second_spike", "time_to_third_spike",
        "time_to_last_spike",
    ]
    out = {name: np.zeros((n_models, n_currents)) for name in fields}

    def _store(results: list[tuple]) -> None:
        for (idx, cc, current, volt_stimend, freq, inv_first_ISI, inv_last_ISI,
             t1, t2, t3, tlast) in results:
            out["current"][idx, cc] = current
            out["volt_stimend"][idx, cc] = volt_stimend
            out["freq"][idx, cc] = freq
            out["inv_first_ISI"][idx, cc] = inv_first_ISI
            out["inv_last_ISI"][idx, cc] = inv_last_ISI
            out["time_to_first_spike"][idx, cc] = t1
            out["time_to_second_spike"][idx, cc] = t2
            out["time_to_third_spike"][idx, cc] = t3
            out["time_to_last_spike"][idx, cc] = tlast

    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(extract_spike_features, idx, model,
                                current_protocols, dt, time,
                                stim_delay, stim_duration)
                for idx, model in enumerate(models)
            ]
            for future in as_completed(futures):
                _store(future.result())
    else:
        for idx, model in enumerate(models):
            _store(extract_spike_features(
                idx, model, current_protocols, dt, time,
                stim_delay, stim_duration))

    return out


# ---------------------------------------------------------------------------
# Ranking and plotting
# ---------------------------------------------------------------------------

def select_best_models(
    err_tot: np.ndarray,
    num_best: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and errors of the *num_best* lowest-error models.

    Returns
    -------
    best_idx : ndarray
        Indices of the best models (ascending error).
    best_err : ndarray
        Their total errors.
    """
    order = np.argsort(err_tot)
    best_idx = order[:num_best]
    return best_idx, err_tot[best_idx]


def plot_best_models(
    best_idx: np.ndarray,
    model_current: np.ndarray,
    model_freq: np.ndarray,
    model_inv_first_ISI: np.ndarray,
    data_current: list[float],
    data_freq: list[float],
    data_inv_first_ISI: list[float],
    num_samples: int,
    out_path: str,
) -> None:
    """Plot the best models against the target data and save to *out_path*."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 5),
                            gridspec_kw={"width_ratios": [1, 1, 0.2]})

    for idx in best_idx:
        axs[0].plot(model_current[idx, :], model_freq[idx, :],
                    "s-", alpha=0.5, label=idx)
        axs[1].plot(model_current[idx, :], model_inv_first_ISI[idx, :],
                    "s-", alpha=0.5, label=idx)

    axs[0].plot(data_current, data_freq, "o-", color="k", label="data")
    axs[1].plot(data_current, data_inv_first_ISI, "o-", color="k", label="data")

    plt.suptitle(f"{len(best_idx)} best models out of {num_samples}")
    axs[0].set_xlabel("current (pA)")
    axs[0].set_ylabel("frequency (Hz)")
    axs[0].set_title("Suprathreshold response")
    axs[1].set_xlabel("current (pA)")
    axs[1].set_ylabel("frequency (Hz)")
    axs[1].set_title("Inv first ISI")

    handles, labels = axs[0].get_legend_handles_labels()
    axs[2].legend(handles, labels, loc="center")
    axs[2].axis("off")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_search(
    *,
    fitting_folder_name: str = "test_cortex_control_v0",
    data_file: str = os.path.join("../", "extracting_features",
                                  "extracting_features_test.json"),
    bounds: Optional[dict[str, Any]] = None,
    indices_sup: Optional[list[int]] = None,
    num_samples: int = 2 ** 12,
    num_best_model: int = 2 ** 3,
    seed: int = 12345,
    dt: float = 0.01,
    parallel_running: bool = True,
    num_threads: Optional[int] = 4,
) -> None:
    """Run the full quasi-random AdEx parameter search.

    Draws *num_samples* parameter sets, simulates the suprathreshold current
    steps selected by *indices_sup*, scores each candidate, then plots and
    saves the *num_best_model* best fits under *fitting_folder_name*.

    Parameters
    ----------
    fitting_folder_name : str
        Output directory (created if absent).
    data_file : str
        Path to the extracted-features JSON.
    bounds : dict, optional
        Parameter bounds (see :data:`DEFAULT_BOUNDS`). Defaults to
        :data:`DEFAULT_BOUNDS`.
    indices_sup : list of int, optional
        Current-step indices to use as targets. Defaults to ``[0]``.
    num_samples : int
        Number of candidate parameter sets.
    num_best_model : int
        Number of best models to plot and save.
    seed : int
        Seed for reproducible sampling.
    dt : float
        Integration time step (ms).
    parallel_running : bool
        Whether to run the sweep with a ``ProcessPoolExecutor``.
    num_threads : int, optional
        Worker count when *parallel_running* is True.
    """
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    indices_sup = [0] if indices_sup is None else indices_sup

    # --- 0. Output folder -------------------------------------------------
    os.makedirs(fitting_folder_name, exist_ok=True)
    print(f"Output folder: '{fitting_folder_name}'")

    # --- 1. Bounds --------------------------------------------------------
    searched_params, intervals, fixed_names, fixed_values = parse_param_bounds(bounds)
    fixed_params = dict(zip(fixed_names, fixed_values))
    print(f"Searched parameters: {searched_params}, intervals: {intervals}")
    print(f"Fixed parameters: {fixed_names}, values: {fixed_values}")

    # --- 2. Sampling ------------------------------------------------------
    scaled_samples = sample_parameter_space(intervals, num_samples, seed)
    print(f"Drew {num_samples} Sobol samples (seed={seed}).")

    models = build_model_set(scaled_samples, searched_params, fixed_params)

    json_file_model_name = os.path.join(
        fitting_folder_name, f"AdEx_models_testing_{num_samples}.json")
    append_to_json(json_file_model_name, models)

    # --- 3. Target features ----------------------------------------------
    targets, protocol = load_target_features(data_file, indices_sup)
    data_current = targets["current"]

    # --- 4./5. Current-injection protocol --------------------------------
    time = np.arange(0, protocol["Time"], dt)
    stim_delay = protocol["delay"] / dt
    stim_duration = protocol["duration"] / dt
    current_protocols = build_current_protocols(
        data_current, time, stim_delay, stim_duration,
        protocol["holding_current"])

    # --- 6. Simulation sweep ---------------------------------------------
    feat = simulate_models(
        models, current_protocols, dt, time, stim_delay, stim_duration,
        parallel=parallel_running, num_threads=num_threads)

    # --- 7. Error --------------------------------------------------------
    err, err_tot = compute_feature_error(
        data_current,
        targets["freq"], feat["freq"],
        targets["inv_first_ISI"], feat["inv_first_ISI"],
        targets["inv_last_ISI"], feat["inv_last_ISI"],
        targets["time_to_first_spike"], feat["time_to_first_spike"],
        targets["time_to_second_spike"], feat["time_to_second_spike"],
        targets["time_to_third_spike"], feat["time_to_third_spike"],
        targets["time_to_last_spike"], feat["time_to_last_spike"],
    )

    json_file_error_name = json_file_model_name[:-5] + "_err.json"
    err_records = [
        {"err_tot": err_tot[i].tolist(), "err_mismatch": err[i].tolist()}
        for i in range(len(models))
    ]
    append_to_json(json_file_error_name, err_records)

    # --- 8. Plot best models ---------------------------------------------
    best_idx, _best_err = select_best_models(err_tot, num_best_model)
    plot_best_models(
        best_idx, feat["current"], feat["freq"], feat["inv_first_ISI"],
        data_current, targets["freq"], targets["inv_first_ISI"],
        num_samples, os.path.join(fitting_folder_name, f"summary_{num_samples}.pdf"))

    # --- 9. Save best models ---------------------------------------------
    best_records = [
        {
            "model": models[idx][0],
            "init": models[idx][1],
            "err_tot": err_tot[idx].tolist(),
            "err_mismatch": err[idx].tolist(),
        }
        for idx in best_idx
    ]
    append_to_json(os.path.join(fitting_folder_name, "best_models.json"),
                   best_records)


if __name__ == "__main__":
    run_search()
