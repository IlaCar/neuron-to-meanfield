"""Brian2 neuron model creation and voltage-clamp protocol.

This module wraps the AdEx and E-GLIF model equations and provides factory functions
for creating Brian2 ``NeuronGroup`` objects and voltage-clamp synapse models.
"""

import json
from pathlib import Path
from typing import Any

import brian2 as b2
import numpy as np
import re

from ntmf.config import IMPLEMENTED_NEURON_MODELS, AdEx_IMPLEMENTED_NEURON_MODELS


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
# E-GLIF model equations with three inputs -- GoC
# ---------------------------------------------------------------------------
EGLIF_eqs_GoC = '''
dV/dt = (gl*(V - El) - Ia + Id + Ie + Is + gE*(Ee - V)+ gE_m*(Ee_m - V) + gI*(Ei - V)) / Cm : volt (unless refractory)
dIa/dt = kadap*(V - El) - k2*Ia : amp
dId/dt = -k1*Id : amp
dgE/dt = xE - gE/Te : siemens
dxE/dt = -xE/Te : siemens/second
dgE_m/dt = xE_m - gE_m/Te_m : siemens
dxE_m/dt = -xE_m/Te_m : siemens/second
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
Ee_m : volt
Ei : volt
Te : second
Te_m : second
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

def setting_EGLIF_simulation_Brian_Vmin_membrane_fluc_GoC(idx=None, N_cell=None, neuron_model=None, json_file_name=None, curr_inj=None):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Please, specify the neuron_model you wish to simulate")
    if neuron_model not in IMPLEMENTED_NEURON_MODELS:
        raise ValueError(f"neuron_model must be one of {IMPLEMENTED_NEURON_MODELS}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Please, specify the json_file_name containing the model parameters")

    with open(json_file_name, 'r') as file:
        data = json.load(file)
    print(f'Imported data: {json_file_name}')

    if neuron_model in IMPLEMENTED_NEURON_MODELS:
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
            model=EGLIF_eqs_GoC,
            threshold=f'rand() < dt * {lambda_0_value} / ms * exp((V - {V_th_value} * mV) / ({tau_V_value} * mV))',
            reset=f'''
                V = {V_reset_value} * mV
                Ia += {A2_value} * pA
                Id = {A1_value} * pA
            ''',
            refractory=f'{t_ref_value} * ms',
            method='rk4',
            events={'vmin_event': 'V < Vmin'}  # define extra event
        )

        # init variables:
        # G_neuron.V = data[0][idx]['init']['V']*b2.mV

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

        # parameter values:
        G_neuron.Cm = data[0][idx]['model']['C_m'] * b2.pF
        G_neuron.gl = data[0][idx]['model']['g_L'] * b2.nS
        G_neuron.El = data[0][idx]['model']['E_L'] * b2.mV
        G_neuron.kadap = data[0][idx]['model']['k_a'] * (b2.amp / (b2.volt * b2.second))
        G_neuron.k1 = data[0][idx]['model']['k_1'] / b2.ms
        G_neuron.k2 = data[0][idx]['model']['k_2'] / b2.ms
        G_neuron.Ie = data[0][idx]['model']['I_e'] * b2.nA

        G_neuron.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_neuron.Ee_m = data[0][idx]['model']['E_e_m'] * b2.mV
        G_neuron.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_neuron.Te = data[0][idx]['model']['T_e'] * b2.ms
        G_neuron.Te_m = data[0][idx]['model']['T_e_m'] * b2.ms
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

# ---------------------------------------------------------------------------
# Channel detection
# ---------------------------------------------------------------------------

def synaptic_channels_from_model(model: dict) -> list[dict]:
    """Normalize a model dict into an ordered list of synaptic-channel specs.

    Supports two naming schemes:

    * classic 2-channel: ``E_e``/``Q_e`` and ``E_i``/``Q_i``, sharing ``tau_syn``;
    * generalized N-channel: ``E_1``/``Q_1``/``tau_1``, ``E_2``/... , any N.

    Each returned spec carries the Brian2 variable stem (``name``) plus the JSON
    keys needed to look up the reversal potential, quantal conductance, decay
    time constant and (optional) initial conductance.
    """
    numbered = sorted(
        int(match.group(1))
        for key in model
        if (match := re.fullmatch(r"E_(\d+)", key)) is not None
    )
    if numbered:
        return [
            {
                "name": str(n),
                "E_key": f"E_{n}",
                "Q_key": f"Q_{n}",
                "tau_key": f"tau_{n}",
                "init_key": f"g_{n}",
            }
            for n in numbered
        ]

    if "E_e" in model and "E_i" in model:
        return [
            {"name": "E", "E_key": "E_e", "Q_key": "Q_e", "tau_key": "tau_syn", "init_key": "g_E"},
            {"name": "I", "E_key": "E_i", "Q_key": "Q_i", "tau_key": "tau_syn", "init_key": "g_I"},
        ]

    raise ValueError(
        "Could not detect synaptic channels: expected either E_e/E_i or E_1, E_2, ..."
    )

def is_generalized_model(model: dict) -> bool:
    """True if the model dict uses the numbered E_1, E_2, ... channel scheme."""
    return any(re.fullmatch(r"E_\d+", key) for key in model)

# ---------------------------------------------------------------------------
# Equation builder
# ---------------------------------------------------------------------------

def build_AdEx_eqs(channels: list[dict]) -> str:
    """Assemble the AdEx equation string for an arbitrary set of channels.

    For each channel ``k`` this creates a conductance state variable ``Gsyn{k}``
    with its own decay ``Tsyn{k}`` and reversal ``E{k}``. With the classic e/i
    channel list this reproduces the original two-channel dynamics (with the
    single ``tau_syn`` duplicated into ``TsynE`` and ``TsynI``).
    """
    stems = [c["name"] for c in channels]

    syn_drive = " ".join(f"- Gsyn{s}*(v-E{s})" for s in stems)
    syn_odes = "\n".join(f"dGsyn{s}/dt = -Gsyn{s}/Tsyn{s} : siemens" for s in stems)
    E_decls = "\n".join(f"E{s} : volt" for s in stems)
    tau_decls = "\n".join(f"Tsyn{s} : second" for s in stems)
    Itot_terms = "+".join(f"Gsyn{s}" for s in stems)

    return f"""
dv/dt = ({syn_drive} - gl*(v-El) + gl*Dt*exp((v-Vt)/Dt) - w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w : ampere
{syn_odes}
Itot = ({Itot_terms})*v : ampere
Is = current(t) : ampere
Cm : farad
gl : siemens
El : volt
a : siemens
tau_w : second
Dt : volt
Vt : volt
{E_decls}
{tau_decls}
"""

# ---------------------------------------------------------------------------
# Generalized neuron creation
# ---------------------------------------------------------------------------

def setting_simulation_Brian_general(
    neuron_model: str,
    json_file_name: str | Path,
    idx: int = 0,
    N_cell: int = 1,
    curr_inj: Any = None,
) -> b2.NeuronGroup:
    """Create an AdEx ``NeuronGroup`` for a cell with any number of channels.

    Mirrors ``setting_simulation_Brian`` but reads the synaptic channels from the
    JSON rather than hard-coding two. Membrane / adaptation handling is identical.
    """
    if neuron_model not in IMPLEMENTED_NEURON_MODELS:
        raise ValueError(
            f"neuron_model must be one of {IMPLEMENTED_NEURON_MODELS}, but got '{neuron_model}'."
        )
    if json_file_name is None:
        raise ValueError("Please specify the json_file_name containing the model parameters.")

    with open(json_file_name, "r") as fh:
        data = json.load(fh)

    entry = data[0][idx]
    m = entry["model"]
    init = entry["init"]

    channels = synaptic_channels_from_model(m)
    eqs = build_AdEx_eqs(channels)

    G = b2.NeuronGroup(
        N_cell,
        eqs,
        threshold=f"v > {m['V_peak_detect']} * mV",
        reset=f"v = {m['V_reset']} * mV; w += {m['b']} * nA",
        refractory=f"{m['t_ref']} * ms",
        method="heun",
        name=neuron_model,
    )

    # membrane initial conditions
    G.v = init["v"] * b2.mV
    G.w = init["w"] * b2.nA

    # membrane / adaptation parameters
    G.Cm = m["C_m"] * b2.nF
    G.gl = m["g_L"] * b2.uS
    G.El = m["E_L"] * b2.mV
    G.a = m["a"] * b2.nS
    G.tau_w = m["tau_w"] * b2.ms
    G.Vt = m["V_th"] * b2.mV
    G.Dt = m["Delta_T"] * b2.mV

    # per-channel reversal, decay and (optional) initial conductance
    for c in channels:
        s = c["name"]
        setattr(G, f"E{s}", m[c["E_key"]] * b2.mV)
        setattr(G, f"Tsyn{s}", m[c["tau_key"]] * b2.ms)
        setattr(G, f"Gsyn{s}", init.get(c["init_key"], 0.0) * b2.nS)

    if m.get("I_e", 0) != 0:
        print(f"!!! Attention!!! I_e = {m['I_e']} nA. Set the current accordingly.")

    return G

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def create_AdEx_neuron(
    neuron_model: str,
    json_file_name: str | Path,
    idx: int = 0,
    N_cell: int = 1,
    curr_inj: Any = None,
    setting_simulation_Brian=None,
) -> b2.NeuronGroup:
    """Select the correct AdEx setup based on the cell JSON.

    Numbered channels (E_1, E_2, ...) -> ``setting_simulation_Brian_general``.
    Classic e/i                       -> the original ``setting_simulation_Brian``.

    Pass the original function in as ``setting_simulation_Brian`` (or rely on it
    being importable in ``neurons.py``).
    """
    with open(json_file_name, "r") as fh:
        model = json.load(fh)[0][idx]["model"]

    if is_generalized_model(model):
        return setting_simulation_Brian_general(neuron_model, json_file_name, idx, N_cell, curr_inj)
    return setting_simulation_Brian(neuron_model, json_file_name, idx, N_cell, curr_inj)