"""Brian2 neuron model creation and voltage-clamp protocol.

This module wraps the AdEx and E-GLIF model equations and provides factory functions
for creating Brian2 ``NeuronGroup`` objects and voltage-clamp synapse models.
"""

import json
from pathlib import Path
from typing import Any

import brian2 as b2
import numpy as np

from ntmf.config import IMPLEMENTED_NEURON_MODELS


# ---------------------------------------------------------------------------
# AdEx model equations
# ---------------------------------------------------------------------------

AdEx_eqs = """
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Itot = (GsynI+GsynE)*v : ampere
Is = current(t) : ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
"""

syn_eqs = """
dgE/dt = -gE/tau_syn_e : siemens
dgI/dt = -gI/tau_syn_i : siemens
IE = -gE*(V_hold - Ee) : ampere
II = -gI*(V_hold - Ei) : ampere
Itot = IE + II : ampere
tau_syn_e : second
tau_syn_i : second
Ee : volt
Ei : volt
V_hold : volt
"""


# ---------------------------------------------------------------------------
# E-GLIF model equations
# ---------------------------------------------------------------------------
EGLIF_eqs = '''
dV/dt = (gl*(V - El) - Ia + Id + Ie + Is + gE*(Ee - V) + gI*(Ei - V)) / Cm : volt (unless refractory)
dIa/dt = kadap*(V - El) - k2*Ia : amp
dId/dt = -k1*Id : amp
dgE/dt = xE - gE/Te : siemens
dxE/dt = -xE/Te : siemens/second
dgI/dt = xI - gI/Ti : siemens
dxI/dt = -xI/Ti : siemens/second
Is = external_current(t) : amp
Ie : amp
gl : siemens
Cm : farad
El : volt
Vmin: volt
kadap : 1/henry
k2 : 1/second
k1 : 1/second
Ee : volt
Ei : volt
Te : second
Ti : second
'''


# ---------------------------------------------------------------------------
# Neuron creation
# ---------------------------------------------------------------------------

def setting_simulation_Brian(
    neuron_model: str,
    json_file_name: str | Path,
    idx: int = 0,
    N_cell: int = 1,
    curr_inj: Any = None,
) -> b2.NeuronGroup:
    """Create a Brian2 ``NeuronGroup`` for the requested AdEx neuron type.

    Parameters
    ----------
    neuron_model : str
        One of ``IMPLEMENTED_NEURON_MODELS`` (FS, RS, RS_no_adapt).
    json_file_name : str or Path
        Path to the neuron-model JSON file.
    idx : int
        Parameter-set index inside the JSON (default 0).
    N_cell : int
        Number of neurons in the group (default 1).
    curr_inj : Brian2 TimedArray or None
        External current injection waveform.

    Returns
    -------
    brian2.NeuronGroup
    """
    if neuron_model not in IMPLEMENTED_NEURON_MODELS:
        raise ValueError(
            f"neuron_model must be one of {IMPLEMENTED_NEURON_MODELS}, "
            f"but got '{neuron_model}'."
        )
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")

    with open(json_file_name, "r") as fh:
        data = json.load(fh)

    entry = data[0][idx]
    m = entry["model"]
    init = entry["init"]

    V_th_value = m["V_peak_detect"]
    V_reset_value = m["V_reset"]
    t_ref_value = m["t_ref"]
    b_value = m["b"]

    G = b2.NeuronGroup(
        N_cell,
        AdEx_eqs,
        threshold=f"v > {V_th_value} * mV",
        reset=f"v = {V_reset_value} * mV; w += {b_value} * nA",
        refractory=f"{t_ref_value} * ms",
        method="heun",
        name=neuron_model,
    )

    # initial conditions
    G.v = init["v"] * b2.mV
    G.w = init["w"] * b2.nA
    G.GsynI = init["g_I"] * b2.nS
    G.GsynE = init["g_E"] * b2.nS

    # model parameters
    G.Cm = m["C_m"] * b2.nF
    G.gl = m["g_L"] * b2.uS
    G.El = m["E_L"] * b2.mV
    G.a = m["a"] * b2.nS
    G.tau_w = m["tau_w"] * b2.ms
    G.Vt = m["V_th"] * b2.mV
    G.Dt = m["Delta_T"] * b2.mV
    G.Ee = m["E_e"] * b2.mV
    G.Ei = m["E_i"] * b2.mV
    G.Tsyn = m["tau_syn"] * b2.ms

    if m["I_e"] != 0:
        print(f"!!! Attention!!! I_e = {m['I_e']} nA. Set the current accordingly.")

    return G


def setting_EGLIF_simulation_Brian_Vmin_membrane_fluc(
    neuron_model: str,
    json_file_name: str | Path,
    idx: int = 0,
    N_cell: int = 1,
    curr_inj: Any = None,
) -> b2.NeuronGroup:
    """Create a Brian2 ``NeuronGroup`` for the requested E-GLIF neuron type.

    Parameters
    ----------
    neuron_model : str
        One of ``IMPLEMENTED_NEURON_MODELS``.
    json_file_name : str or Path
        Path to the neuron-model JSON file.
    idx : int
        Parameter-set index inside the JSON (default 0).
    N_cell : int
        Number of neurons in the group (default 1).
    curr_inj : Brian2 TimedArray or None
        External current injection waveform.

    Returns
    -------
    brian2.NeuronGroup
    """
    if neuron_model not in IMPLEMENTED_NEURON_MODELS:
        raise ValueError(
            f"neuron_model must be one of {IMPLEMENTED_NEURON_MODELS}, "
            f"but got '{neuron_model}'."
        )
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")

    with open(json_file_name, "r") as fh:
        data = json.load(fh)

    print(f'neuron model: {neuron_model}')
    V_th_value = data[0][idx]['model']['V_th']
    V_reset_value = data[0][idx]['model']['V_reset']
    lambda_0_value = data[0][idx]['model']['lambda_0']
    tau_V_value = data[0][idx]['model']['tau_V']
    t_ref_value = data[0][idx]['model']['t_ref']
    A2_value = data[0][idx]['model']['A_2']
    A1_value = data[0][idx]['model']['A_1']

    G_neuron = b2.NeuronGroup(
        N_cell,
        model=EGLIF_eqs,
        threshold=f'rand() < dt * {lambda_0_value} / ms * exp((V - {V_th_value} * mV) / ({tau_V_value} * mV))',
        reset=f'''
            V = {V_reset_value} * mV
            Ia += {A2_value} * pA
            Id = {A1_value} * pA
        ''',
        refractory=f'{t_ref_value} * ms',
        method='rk4',
        events={'vmin_event': 'V < Vmin'}
    )

    # Randomly initialize membrane potential
    EL = data[0][idx]['model']['E_L']
    Vreset = data[0][idx]['model']['V_reset']
    Vth = data[0][idx]['model']['V_th']

    lower = EL + (Vreset - EL)
    upper = EL + (Vth - EL) / 2

    G_neuron.V = (lower + (upper - lower) * b2.rand(N_cell)) * b2.mV

    G_neuron.Ia = data[0][idx]['init']['I_a'] * b2.nA
    G_neuron.Id = data[0][idx]['init']['I_d'] * b2.nA
    G_neuron.gE = 0 * b2.nS
    G_neuron.gI = 0 * b2.nS
    G_neuron.xE = 0 * b2.nS / b2.ms
    G_neuron.xI = 0 * b2.nS / b2.ms

    G_neuron.Cm = data[0][idx]['model']['C_m'] * b2.pF
    G_neuron.gl = data[0][idx]['model']['g_L'] * b2.nS
    G_neuron.El = EL * b2.mV
    G_neuron.kadap = data[0][idx]['model']['k_a'] * (b2.amp / (b2.volt * b2.second))
    G_neuron.k1 = data[0][idx]['model']['k_1'] / b2.ms
    G_neuron.k2 = data[0][idx]['model']['k_2'] / b2.ms
    G_neuron.Ie = data[0][idx]['model']['I_e'] * b2.nA

    G_neuron.Ee = data[0][idx]['model']['E_e'] * b2.mV
    G_neuron.Ei = data[0][idx]['model']['E_i'] * b2.mV
    G_neuron.Te = data[0][idx]['model']['T_e'] * b2.ms
    G_neuron.Ti = data[0][idx]['model']['T_i'] * b2.ms

    if 'V_min' in data[0][idx]['model'].keys():
        G_neuron.Vmin = data[0][idx]['model']['V_min'] * b2.mV

    return G_neuron, data

# ---------------------------------------------------------------------------
# Voltage clamp
# ---------------------------------------------------------------------------

def voltage_clamp_synapse(
    json_file_name: str | Path,
    V_hold: float = -60.0,
    idx: int = 0,
    dt: float = 0.1,
):
    """Run a voltage-clamp protocol to measure synaptic currents.

    Returns a Brian2 StateMonitor.
    """
    with open(json_file_name, "r") as fh:
        data = json.load(fh)

    m = data[0][idx]["model"]
    s = data[0][idx]["simulation"]
    init = data[0][idx]["init"]

    b2.start_scope()
    b2.defaultclock.dt = dt * b2.ms

    G = b2.NeuronGroup(1, syn_eqs, method="exact")
    G.tau_syn_e = m["tau_e"] * b2.ms
    G.tau_syn_i = m["tau_i"] * b2.ms
    G.Ee = m["E_e"] * b2.mV
    G.Ei = m["E_i"] * b2.mV
    G.V_hold = V_hold * b2.mV
    G.gE = init["g_E"] * b2.nS
    G.gI = init["g_I"] * b2.nS

    sim_time = s["sim_duration"] * b2.ms
    t_event = s["t_pulse"] * b2.ms

    @b2.network_operation(dt=dt * b2.ms)
    def inject_event():
        if abs(b2.defaultclock.t - t_event) < 0.5 * dt * b2.ms:
            G.gE[0] += m["Q_e"] * b2.nS
            G.gI[0] += m["Q_i"] * b2.nS

    M = b2.StateMonitor(G, ["gE", "gI", "IE", "II", "Itot"], record=True)

    net = b2.Network(G, inject_event, M)
    net.run(sim_time)
    return M
