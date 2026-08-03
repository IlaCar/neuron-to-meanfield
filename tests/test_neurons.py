"""Tests for Brian2 neuron model creation.

These tests require Brian2 but use minimal simulations (< 1 s each).
"""

import json
import os

import numpy as np
import pytest
import brian2 as b2

from ntmf.neurons import (
    AdEx_eqs,
    AdEx_IMPLEMENTED_NEURON_MODELS,
    setting_simulation_Brian,
    voltage_clamp_synapse,
)
from ntmf.config import AdEx_IMPLEMENTED_NEURON_MODELS as CONFIG_MODELS


# =====================================================================
# Neuron creation
# =====================================================================

class TestNeuronCreation:
    def test_create_single_FS(self, neuron_models_dir):
        b2.start_scope()
        stim = _zero_current(1.0)
        G = setting_simulation_Brian("FS", neuron_models_dir / "FS.json", curr_inj=stim)
        assert G.N == 1
        assert G.name == "FS"

    def test_create_single_RS(self, neuron_models_dir):
        b2.start_scope()
        stim = _zero_current(1.0)
        G = setting_simulation_Brian("RS", neuron_models_dir / "RS.json", curr_inj=stim)
        assert G.N == 1
        assert G.name == "RS"

    def test_create_single_RS_no_adapt(self, neuron_models_dir):
        b2.start_scope()
        stim = _zero_current(1.0)
        G = setting_simulation_Brian("RS_no_adapt", neuron_models_dir / "RS_no_adapt.json", curr_inj=stim)
        assert G.N == 1
        assert G.name == "RS_no_adapt"

    def test_create_N_neurons(self, neuron_models_dir):
        b2.start_scope()
        stim = _zero_current(1.0)
        N = 10
        G = setting_simulation_Brian("FS", neuron_models_dir / "FS.json", N_cell=N, curr_inj=stim)
        assert G.N == N

    def test_invalid_model_raises(self, neuron_models_dir):
        with pytest.raises(ValueError, match="neuron_model must be one of"):
            setting_simulation_Brian("INVALID", neuron_models_dir / "FS.json")

    def test_none_model_raises(self, neuron_models_dir):
        with pytest.raises(ValueError):
            setting_simulation_Brian(None, neuron_models_dir / "FS.json")

    def test_none_filename_raises(self):
        with pytest.raises(ValueError):
            setting_simulation_Brian("FS", None)

    def test_adex_equations_parse(self):
        """The AdEx_eqs string should be usable in a Brian2 NeuronGroup."""
        b2.start_scope()
        G = b2.NeuronGroup(1, AdEx_eqs, method="heun")
        assert G.N == 1

    def test_neuron_spikes_under_current(self, neuron_models_dir):
        """A neuron receiving step current should fire at least once."""
        b2.start_scope()
        duration = 500 * b2.ms
        dt = 0.1 * b2.ms

        # Build step current: 0.5 nA from 100ms to 400ms
        n_steps = int(duration / dt)
        curr_vals = b2.zeros(n_steps) * b2.amp
        start = int(100 * b2.ms / dt)
        end = int(400 * b2.ms / dt)
        curr_vals[start:end] = 0.5 * b2.nA
        current_timed_array = b2.TimedArray(curr_vals, dt=dt)

        G = setting_simulation_Brian("RS", neuron_models_dir / "RS.json", curr_inj=current_timed_array)
        spike_mon = b2.SpikeMonitor(G)

        b2.run(duration, namespace={"current": current_timed_array})
        assert spike_mon.num_spikes > 0


# =====================================================================
# I_e != 0 warning path (line 133)
# =====================================================================

class TestNeuronNonZeroIe:
    def test_non_zero_ie_prints_warning(self, neuron_models_dir, tmp_path):
        """A model with I_e != 0 should trigger the print warning (line 133)."""
        import builtins
        printed = []
        original_print = builtins.print
        builtins.print = lambda *a, **kw: (printed.append(" ".join(str(x) for x in a)), original_print(*a, **kw))
        try:
            # Load real RS params and modify I_e
            with open(neuron_models_dir / "RS.json") as f:
                data = json.load(f)
            data[0][0]["model"]["I_e"] = 0.5
            tmp_json = tmp_path / "RS_ie.json"
            with open(tmp_json, "w") as f:
                json.dump(data, f)

            b2.start_scope()
            stim = _zero_current(0.5)
            G = setting_simulation_Brian("RS", tmp_json, curr_inj=stim)
            assert G.N == 1
            assert any("Attention" in p and "0.5" in p for p in printed)
        finally:
            builtins.print = original_print


# =====================================================================
# Voltage clamp (lines 152-184) — slow, requires Brian2 sim
# =====================================================================

class TestVoltageClampSynapse:
    @pytest.fixture(scope="class")
    def voltage_clamp_json(self, neuron_models_dir, tmp_path_factory):
        """Create a JSON file with the extra keys voltage_clamp_synapse needs."""
        with open(neuron_models_dir / "RS.json") as f:
            data = json.load(f)
        m = data[0][0]["model"]
        # Add keys required by voltage_clamp_synapse
        m["tau_e"] = 5.0   # ms
        m["tau_i"] = 10.0  # ms
        # simulation section
        data[0][0]["simulation"] = {
            "sim_duration": 50.0,  # ms
            "t_pulse": 20.0,       # ms
        }
        tmp_json = tmp_path_factory.mktemp("vc") / "vc_RS.json"
        with open(tmp_json, "w") as f:
            json.dump(data, f)
        return tmp_json

    @pytest.mark.slow
    def test_returns_state_monitor(self, voltage_clamp_json):
        """voltage_clamp_synapse should return a Brian2 StateMonitor."""
        M = voltage_clamp_synapse(voltage_clamp_json, V_hold=-60.0, dt=0.1)
        assert hasattr(M, "gE")
        assert hasattr(M, "gI")
        assert hasattr(M, "IE")
        assert hasattr(M, "II")
        assert hasattr(M, "Itot")

    @pytest.mark.slow
    def test_conductance_pulse_visible(self, voltage_clamp_json):
        """The conductance should jump at the pulse time."""
        M = voltage_clamp_synapse(voltage_clamp_json, V_hold=-60.0, dt=0.1)
        gE = np.array(M.gE[0] / b2.nS)
        # gE should be higher after the pulse than before
        idx_pulse = int(20.0 / 0.1)  # pulse at t=20ms
        assert np.mean(gE[idx_pulse:]) > np.mean(gE[:idx_pulse])

    @pytest.mark.slow
    def test_different_V_hold(self, voltage_clamp_json):
        """Different V_hold should produce different currents."""
        M1 = voltage_clamp_synapse(voltage_clamp_json, V_hold=-60.0, dt=0.1)
        M2 = voltage_clamp_synapse(voltage_clamp_json, V_hold=-40.0, dt=0.1)
        # Different V_hold → different driving force → different Itot after pulse
        itot1 = np.asarray(M1.Itot[0])
        itot2 = np.asarray(M2.Itot[0])
        # Values at the pulse peak (around index 210) should differ
        v1, v2 = float(itot1[210]), float(itot2[210])
        assert v1 != v2, f"Itot should differ: {v1} vs {v2}"


# =====================================================================
# Helper
# =====================================================================

def _zero_current(duration_s: float) -> b2.TimedArray:
    """Create a zero-current TimedArray for the given duration."""
    dt = 1.0 * b2.ms
    n = int(duration_s * b2.second / dt)
    vals = b2.zeros(n) * b2.amp
    return b2.TimedArray(vals, dt=dt)
