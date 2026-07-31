"""AdEx single-neuron integration and spike-feature extraction.

Forward-Euler integrator and feature-extraction helpers used by the
parameter search in :mod:`AdEx_search`. Given a candidate AdEx (adaptive
exponential integrate-and-fire) parameter set and a current-injection
protocol, this module integrates the membrane trace and reduces the
resulting spike train to a small set of scalar features (firing rate,
inverse first/last ISI, spike latencies).

The integrator is intentionally independent of Brian2. The search evaluates
thousands of candidate parameter sets, so it relies on a lightweight
NumPy/Numba forward-Euler kernel rather than the Brian2 neuron models used
elsewhere in the project (see :mod:`ntmf.neurons`).

Model
-----
The AdEx neuron is a 2-D system in the membrane potential ``V`` (mV) and the
adaptation current ``w`` (nA)::

    C_m dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_th)/Delta_T) - w + I_e
    tau_w dw/dt = a (V - E_L) - w

with a spike-and-reset rule: when ``V`` crosses ``V_peak`` the sample is
marked at ``V_peak``, ``V`` is clamped to ``V_reset`` for the refractory
window, and ``w`` is incremented by ``b``.

Units
-----
PyNN convention throughout::

    C_m [nF], g_L [uS], E_L/V_th/Delta_T/V_reset/V_peak [mV],
    a [uS], b [nA], tau_w/t_ref [ms], I_e [nA].

Current protocols are supplied in **pA** and converted to nA internally.

Notes
-----
The integration is sequential (each Euler step depends on the previous one),
so it cannot be vectorised across time. The inner loop is JIT-compiled with
Numba (:func:`_adex_euler_kernel`). If Numba is not installed, ``njit``
degrades to a no-op decorator and the identical kernel runs as plain Python,
bit-for-bit equivalent, without the speedup.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------

try:
    from numba import njit

    _HAVE_NUMBA = True
    print("Using Numba")
except Exception:  # pragma: no cover - numba is optional
    _HAVE_NUMBA = False
    print("Numba not found; running the Euler kernel as plain Python "
          "(pip install numba for the JIT speedup).")

    def njit(*args, **kwargs):
        """No-op fallback so the kernel runs as plain Python without Numba."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Right-hand sides
# ---------------------------------------------------------------------------

def dVdt(V: float, w: float, p: dict[str, float]) -> float:
    """Membrane-potential derivative dV/dt of the AdEx model.

    Parameters
    ----------
    V, w : float
        Current membrane potential (mV) and adaptation current (nA).
    p : dict
        Parameter dictionary with keys ``g_L, E_L, Delta_T, V_th, I_e, C_m``.

    Returns
    -------
    float
        dV/dt in mV/ms.

    Notes
    -----
    Retained for readability and for callers that integrate the equations
    directly. The hot path in :func:`integrate_current_protocol` inlines this
    expression inside the compiled kernel for speed.
    """
    return (-p["g_L"] * (V - p["E_L"])
            + p["g_L"] * p["Delta_T"] * np.exp((V - p["V_th"]) / p["Delta_T"])
            - w + p["I_e"]) / p["C_m"]


def dwdt(V: float, w: float, p: dict[str, float]) -> float:
    """Adaptation-current derivative dw/dt of the AdEx model.

    Parameters
    ----------
    V, w : float
        Current membrane potential (mV) and adaptation current (nA).
    p : dict
        Parameter dictionary with keys ``a, E_L, tau_w``.

    Returns
    -------
    float
        dw/dt in nA/ms.
    """
    return (p["a"] * (V - p["E_L"]) - w) / p["tau_w"]


# ---------------------------------------------------------------------------
# Compiled integration kernel
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _adex_euler_kernel(I_nA, dt, v0, w0, C_m, g_L, E_L, Delta_T,
                       V_th, a, tau_w, b, V_peak, V_reset, t_ref):
    """Numba-compiled forward-Euler integration of one AdEx trace.

    All parameters are passed as scalars (not a dict) so the loop is free of
    Python object access and can be compiled to machine code. The algorithm,
    including spike marking and refractory bookkeeping, matches the reference
    implementation to floating-point rounding.

    Parameters
    ----------
    I_nA : ndarray of float64
        Injected current per time step, already converted to nA.
    dt : float
        Integration time step (ms).
    v0, w0 : float
        Initial membrane potential (mV) and adaptation current (nA).
    C_m, g_L, E_L, Delta_T, V_th, a, tau_w, b, V_peak, V_reset, t_ref : float
        AdEx parameters (see module docstring for units).

    Returns
    -------
    V, w : ndarray of float64
        Membrane-potential and adaptation traces. Spikes are marked by samples
        exactly equal to ``V_peak`` (the convention used downstream to detect
        spike times via ``np.where(V == V_peak)``).
    """
    n = I_nA.shape[0]
    V = np.zeros(n)
    w = np.zeros(n)
    V[0] = v0
    w[0] = w0

    refractory_steps = int(t_ref / dt)  # refractory window in time steps

    i = 1
    while i < n:
        Vp = V[i - 1]
        wp = w[i - 1]
        # Euler step using the current sample's injected current.
        V[i] = Vp + dt * ((-g_L * (Vp - E_L)
                           + g_L * Delta_T * np.exp((Vp - V_th) / Delta_T)
                           - wp + I_nA[i]) / C_m)
        w[i] = wp + dt * ((a * (Vp - E_L) - wp) / tau_w)

        # Spike-and-reset discontinuity.
        if V[i] >= V_peak:
            V[i - 1] = V_peak      # mark the spike sample at the crossing
            V[i] = V_reset         # reset membrane potential
            w[i] = w[i] + b        # spike-triggered adaptation
            wi = w[i]
            # Hold V at reset and w frozen for the refractory window.
            for j in range(refractory_steps):
                if i + j < n:
                    V[i + j] = V_reset
                    if j == 0:
                        w[i + j] = wi
                    else:
                        w[i + j] = w[i + j - 1]
            i += refractory_steps - 1

        i += 1

    return V, w


def integrate_current_protocol(
    params: dict[str, float],
    initial_values: dict[str, float],
    dt: float,
    time: np.ndarray,
    current_protocol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one AdEx trace over a current-injection protocol.

    Parameters
    ----------
    params : dict
        AdEx parameter dictionary (keys as produced by
        :func:`search.build_adex_params`). Not mutated.
    initial_values : dict
        Initial values ``{"v": V0, "w": w0}``.
    dt : float
        Integration time step (ms).
    time : ndarray
        Time vector; only its length is used to size the output arrays.
    current_protocol : ndarray
        Injected current per time step in **pA**.

    Returns
    -------
    V, w : ndarray
        Membrane-potential (mV) and adaptation (nA) traces.
    """
    # Convert the whole protocol pA -> nA once.
    I_nA = np.asarray(current_protocol, dtype=np.float64) * 1e-3
    # Guard the length against a possibly longer `time` array.
    if I_nA.shape[0] != len(time):
        buf = np.zeros(len(time))
        buf[:I_nA.shape[0]] = I_nA[:len(time)]
        I_nA = buf

    return _adex_euler_kernel(
        I_nA, float(dt), float(initial_values["v"]), float(initial_values["w"]),
        float(params["C_m"]), float(params["g_L"]), float(params["E_L"]),
        float(params["Delta_T"]), float(params["V_th"]), float(params["a"]),
        float(params["tau_w"]), float(params["b"]), float(params["V_peak"]),
        float(params["V_reset"]), float(params["t_ref"]),
    )


# ---------------------------------------------------------------------------
# Spike-feature extraction
# ---------------------------------------------------------------------------

def extract_spike_features(
    idx: int,
    model: tuple[dict[str, float], dict[str, float]],
    current_protocols: list[np.ndarray],
    dt: float,
    time: np.ndarray,
    stim_delay: float,
    stim_duration: float,
) -> list[tuple]:
    """Simulate every current step for one model and extract spike features.

    For each protocol in *current_protocols* the AdEx model is integrated and
    a set of scalar features is computed from the spike train: firing rate,
    inverse first/last inter-spike interval, and the latencies to the first
    four and the last spike.

    Parameters
    ----------
    idx : int
        Index of this model in the sample set (passed through to the results).
    model : tuple
        ``(params, initial_values)`` as stored by the driver.
    current_protocols : list of ndarray
        One current-vs-time protocol (pA) per current amplitude.
    dt : float
        Integration time step (ms).
    time : ndarray
        Time vector (ms).
    stim_delay, stim_duration : float
        Stimulus onset delay and duration expressed in **time steps**
        (i.e. ``delay / dt`` and ``duration / dt``).

    Returns
    -------
    list of tuple
        One tuple per protocol::

            (idx, cc, current, volt_stimend, freq, inv_first_ISI, inv_last_ISI,
             time_to_first_spike, time_to_second_spike, time_to_third_spike,
             time_to_last_spike)
    """
    results = []
    params, initial_values = model

    for cc, current_protocol in enumerate(current_protocols):
        v_Euler, _w_Euler_nA = integrate_current_protocol(
            params, initial_values, dt, time, current_protocol)

        # Feature defaults (stay None when the spike count is too low).
        volt_stimend = None
        freq = None
        inv_first_ISI = None
        inv_last_ISI = None
        time_to_first_spike = None
        time_to_second_spike = None
        time_to_third_spike = None
        time_to_last_spike = None

        # Representative injected current: value one third into the protocol,
        # i.e. during the stimulus window.
        idx_curr = int(len(current_protocol) / 3)
        current = current_protocol[idx_curr]

        # Spikes are the samples marked exactly at V_peak by the kernel.
        peak_index = np.where(v_Euler == params["V_peak"])[0]

        # Firing rate in Hz. stim_duration is in steps, so the stimulus window
        # in seconds is stim_duration * dt / 1000.
        stim_window_s = stim_duration * dt / 1000.0
        freq = len(peak_index) / stim_window_s

        if len(peak_index) == 1:
            time_to_first_spike = time[peak_index[0]] - stim_delay * dt  # ms
        if len(peak_index) > 1:
            inv_first_ISI = 1000 / (time[peak_index[1]] - time[peak_index[0]])  # Hz
            time_to_first_spike = time[peak_index[0]] - stim_delay * dt  # ms
            time_to_second_spike = time[peak_index[1]] - stim_delay * dt  # ms
        if len(peak_index) > 2:
            time_to_third_spike = time[peak_index[2]] - stim_delay * dt  # ms
            inv_last_ISI = 1000 / (time[peak_index[-1]] - time[peak_index[-2]])  # Hz
        if len(peak_index) > 3:
            time_to_last_spike = time[peak_index[-1]] - stim_delay * dt  # ms

        results.append((idx, cc, current, volt_stimend, freq, inv_first_ISI,
                        inv_last_ISI, time_to_first_spike, time_to_second_spike,
                        time_to_third_spike, time_to_last_spike))

    return results
