"""Tests for configuration and parameter loading."""

import json
from pathlib import Path

import pytest

from ntmf.config import (
    IMPLEMENTED_NEURON_MODELS,
    adding_K_params,
    get_input_config,
    get_network_config,
    get_params_model_SI,
    get_syn_info,
)


# =====================================================================
# get_params_model_SI
# =====================================================================

class TestGetParamsModelSI:
    def test_all_AdEx_models_load(self, neuron_models_dir):
        """Every implemented model loads without error."""
        for model in IMPLEMENTED_NEURON_MODELS:
            p = get_params_model_SI(model, neuron_models_dir / "AdEx"/ f"{model}.json")
            assert isinstance(p, dict)
            assert len(p) > 0

    def test_SI_unit_conversion(self, params_RS):
        """Check key conversions: RS.json has C_m=0.2 nF → 2e-10 F."""
        assert abs(params_RS["C_m"] - 0.2e-9) < 1e-20
        assert abs(params_RS["g_L"] - 0.01e-6) < 1e-16
        assert abs(params_RS["E_L"] - (-65e-3)) < 1e-16
        assert abs(params_RS["tau_syn"] - 5e-3) < 1e-16
        assert abs(params_RS["Q_e"] - 1.5e-9) < 1e-16
        assert abs(params_RS["Q_i"] - 5.0e-9) < 1e-16

    def test_FS_has_no_adaptation(self, params_FS):
        """FS model: b=0, a=0."""
        assert params_FS["b"] == 0.0
        assert params_FS["a"] == 0.0

    def test_RS_has_adaptation(self, params_RS):
        """RS model: b=0.1 nA."""
        assert abs(params_RS["b"] - 0.1e-9) < 1e-20

    def test_RS_no_adapt_has_zero_b(self, params_RS_na):
        assert params_RS_na["b"] == 0.0

    def test_all_keys_present(self, params_FS):
        expected = {"C_m", "g_L", "E_L", "a", "b", "tau_w",
                    "V_th", "Delta_T", "V_reset", "V_peak", "t_ref",
                    "E_e", "Q_e", "E_i", "Q_i", "tau_syn"}
        assert expected.issubset(params_FS.keys())

    def test_missing_model_raises(self, neuron_models_dir):
        with pytest.raises(ValueError, match="neuron_model must be one of"):
            get_params_model_SI("UNKNOWN", neuron_models_dir / "FS.json")

    def test_none_model_raises(self, neuron_models_dir):
        with pytest.raises(ValueError, match="specify the neuron_model"):
            get_params_model_SI(None, neuron_models_dir / "FS.json")

    def test_none_filename_raises(self):
        with pytest.raises(ValueError, match="specify the json_file_name"):
            get_params_model_SI("FS", None)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_params_model_SI("FS", tmp_path / "nonexistent.json")


# =====================================================================
# get_input_config / get_network_config
# =====================================================================

class TestInputConfig:
    def test_loads_TF_config(self, config_dir):
        cfg = get_input_config(config_dir / "input_config_TF.json")
        assert "connections" in cfg
        assert "rates" in cfg
        assert cfg["connections"]["N_external_exc"] == 8000
        assert cfg["connections"]["N_external_inh"] == 2000
        assert cfg["connections"]["conn_prob"] == 0.05

    def test_loads_v0_config(self, config_dir):
        cfg = get_input_config(config_dir / "input_config_v0.json")
        assert "connections" in cfg
        assert "rates" in cfg

    def test_none_filename_raises(self):
        with pytest.raises(ValueError):
            get_input_config(None)


class TestNetworkConfig:
    def test_loads_v0(self, config_dir):
        cfg = get_network_config(config_dir / "network_config_file_v0.json")
        assert cfg["network_composition"]["tot_neurons"] == 10000
        assert cfg["network_composition"]["FS_neuron"] == 2000
        assert cfg["network_composition"]["RS_neuron"] == 8000
        assert cfg["network_composition"]["conn_prob"] == 0.05
        assert cfg["external_input"]["N_external_exc"] == 8000
        assert cfg["external_input"]["conn_prob"] == 0.05

    def test_none_filename_raises(self):
        with pytest.raises(ValueError):
            get_network_config(None)


# =====================================================================
# get_syn_info
# =====================================================================

class TestGetSynInfo:
    def test_FS_values(self, neuron_models_dir):
        import brian2 as b2
        Qe, Qi = get_syn_info(neuron_models_dir / "FS.json")
        assert abs(Qe / b2.nS - 1.5) < 1e-6
        assert abs(Qi / b2.nS - 5.0) < 1e-6

    def test_none_raises(self):
        with pytest.raises(ValueError):
            get_syn_info(None)


# =====================================================================
# adding_K_params
# =====================================================================

class TestAddingKParams:
    def test_computation(self, params_FS, network_config):
        p = adding_K_params(params_FS.copy(), network_config)
        assert p["K_e"] == 8000 * 0.05   # 400
        assert p["K_i"] == 2000 * 0.05   # 100

    def test_preserves_existing_keys(self, params_RS, network_config):
        p = adding_K_params(params_RS.copy(), network_config)
        assert "C_m" in p
        assert "K_e" in p
        assert "K_i" in p

    def test_modifies_in_place(self, params_FS, network_config):
        original = params_FS.copy()
        result = adding_K_params(params_FS.copy(), network_config)
        assert result is not params_FS  # different object
        assert "K_e" in result
