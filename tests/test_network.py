"""Tests for network creation and spike-rate extraction."""

from types import SimpleNamespace

import brian2 as b2
import numpy as np
import pytest

from ntmf.network import (
    _build_spike_matrix,
    extracting_pop_freq_and_std,
    extracting_single_pop_freq_and_std,
    network_creation,
)
from ntmf.neurons import setting_simulation_Brian


# =====================================================================
# Rate extraction with mock data
# =====================================================================

class TestBuildSpikeMatrix:
    def test_known_spike_train(self):
        """One neuron, one spike at t=0.5 s, bin_size=1.0 s → rate = 1 Hz."""
        pop = SimpleNamespace(
            i=np.array([0]),
            t=np.array([0.5]) * b2.second,
        )
        time_bins = np.array([0.0, 1.0])
        matrix = _build_spike_matrix(pop, N_pop=1, time_bins=time_bins, bin_size=1.0)
        assert matrix.shape == (1, 2)
        assert matrix[0, 0] == 1.0  # 1 spike in 1s = 1 Hz
        assert matrix[0, 1] == 0.0

    def test_multiple_neurons(self):
        """Two neurons, each with one spike."""
        pop = SimpleNamespace(
            i=np.array([0, 1]),
            t=np.array([0.3, 0.7]) * b2.second,
        )
        time_bins = np.array([0.0, 1.0])
        matrix = _build_spike_matrix(pop, N_pop=2, time_bins=time_bins, bin_size=1.0)
        assert matrix[0, 0] == 1.0
        assert matrix[1, 0] == 1.0

    def test_empty_spikes(self):
        """No spikes → all zeros."""
        pop = SimpleNamespace(i=np.array([], dtype=int), t=np.array([]) * b2.second)
        time_bins = np.array([0.0, 1.0, 2.0])
        matrix = _build_spike_matrix(pop, N_pop=5, time_bins=time_bins, bin_size=1.0)
        assert matrix.shape == (5, 3)
        assert np.all(matrix == 0)


class TestExtractingSinglePopFreqAndStd:
    def test_known_rate(self):
        """5 neurons, each firing 10 spikes in 1 s → mean rate ≈ 10 Hz."""
        # Each neuron fires 10 spikes uniformly in [0, 1)
        rng = np.random.default_rng(0)
        neuron_ids = np.repeat(np.arange(5), 10)
        spike_times = rng.uniform(0, 1, size=50) * b2.second

        pop = SimpleNamespace(i=neuron_ids, t=spike_times)
        mean_rate, std_rate = extracting_single_pop_freq_and_std(
            sim_duration=2.0 * b2.second,
            p_start=0.0 * b2.second,
            p_end=1.0 * b2.second,
            pop=pop,
            N_pop=5,
            bin_size=0.5,
        )
        # Each neuron fires 10 spikes in 1 s = 10 Hz.
        # With bin_size=0.5s over a 1s window, binning causes imprecision.
        # The mean across bins should be approximately 10 Hz.
        assert abs(mean_rate - 10.0) < 5.0  # generous: binning splits spikes across bins

    def test_with_delay(self):
        """Delay should skip early bins."""
        pop = SimpleNamespace(i=np.array([0, 0]), t=np.array([0.1, 0.2]) * b2.second)
        # With delay = 0.5 s, spikes at 0.1 and 0.2 should be excluded
        mean_rate, _ = extracting_single_pop_freq_and_std(
            sim_duration=2.0 * b2.second,
            p_start=0.0 * b2.second,
            p_end=1.5 * b2.second,
            pop=pop,
            N_pop=1,
            bin_size=0.5,
            delay=0.5 * b2.second,
        )
        # No spikes in [0.5, 1.5] window
        assert mean_rate == 0.0


class TestExtractingPopFreqAndStd:
    def test_two_populations(self):
        """Two populations with known spike counts."""
        pop1 = SimpleNamespace(i=np.array([0]), t=np.array([0.5]) * b2.second)
        pop2 = SimpleNamespace(i=np.array([0, 0]), t=np.array([0.3, 0.7]) * b2.second)

        mean_rates, std_rates = extracting_pop_freq_and_std(
            sim_duration=1.0 * b2.second,
            p_start=0.0 * b2.second,
            p_end=1.0 * b2.second,
            pop1=pop1,
            pop2=pop2,
            N_pop1=1,
            N_pop2=1,
            bin_size=0.5,
        )
        assert len(mean_rates) == 2
        assert len(std_rates) == 2
        assert mean_rates[0] >= 0
        assert mean_rates[1] >= 0

    def test_symmetric_stimulus_window(self):
        """When p_end == sim_duration - p_start, uses the symmetric branch (line 109-110)."""
        # sim_duration=2.0, p_start=0.5, p_end=1.5 → p_end == sim_duration - p_start = 1.5
        rng = np.random.default_rng(42)
        ids1 = np.repeat(np.arange(3), 10)
        ids2 = np.repeat(np.arange(3), 10)
        t1 = rng.uniform(0.5, 1.4, size=30) * b2.second
        t2 = rng.uniform(0.5, 1.4, size=30) * b2.second

        pop1 = SimpleNamespace(i=ids1, t=t1)
        pop2 = SimpleNamespace(i=ids2, t=t2)

        mean_rates, std_rates = extracting_pop_freq_and_std(
            sim_duration=2.0 * b2.second,
            p_start=0.5 * b2.second,
            p_end=1.5 * b2.second,  # == sim_duration - p_start
            pop1=pop1,
            pop2=pop2,
            N_pop1=3,
            N_pop2=3,
            bin_size=0.25,
        )
        assert len(mean_rates) == 2
        assert np.all(np.isfinite(mean_rates))


# =====================================================================
# Network creation (requires Brian2)
# =====================================================================

class TestNetworkCreation:
    @pytest.mark.slow
    def test_connectivity_order(self, neuron_models_dir, config_dir):
        """Synapse counts should be roughly conn_prob * N_pre * N_post."""
        from ntmf.config import get_network_config

        b2.start_scope()
        cfg = get_network_config(config_dir / "network_config_file_v0.json")
        p = cfg["network_composition"]["conn_prob"]
        N_FS = 20
        N_RS = 40

        stim = b2.TimedArray(b2.zeros(1000) * b2.amp, dt=1 * b2.ms)

        pop_FS = setting_simulation_Brian("FS", neuron_models_dir / "FS.json",
                                           N_cell=N_FS, curr_inj=stim)
        pop_RS = setting_simulation_Brian("RS", neuron_models_dir / "RS.json",
                                           N_cell=N_RS, curr_inj=stim)

        Qe_FS = 1.5 * b2.nS
        Qi_FS = 5.0 * b2.nS
        Qe_RS = 1.5 * b2.nS
        Qi_RS = 5.0 * b2.nS

        S_11, S_12, S_21, S_22 = network_creation(
            conn_prob=p, pop_1=pop_FS, pop_2=pop_RS,
            Qe_FS=Qe_FS, Qi_FS=Qi_FS, Qe_RS=Qe_RS, Qi_RS=Qi_RS,
            seed=42,
        )

        # S_11: FS→FS, no self → ≈ p * N_FS * (N_FS - 1)
        expected_11 = p * N_FS * (N_FS - 1)
        assert abs(len(S_11) - expected_11) / expected_11 < 0.3

        # S_12: FS→RS → ≈ p * N_FS * N_RS
        expected_12 = p * N_FS * N_RS
        assert abs(len(S_12) - expected_12) / expected_12 < 0.3

        # S_21: RS→FS → ≈ p * N_RS * N_FS
        expected_21 = p * N_RS * N_FS
        assert abs(len(S_21) - expected_21) / expected_21 < 0.3

        # S_22: RS→RS, no self → ≈ p * N_RS * (N_RS - 1)
        expected_22 = p * N_RS * (N_RS - 1)
        assert abs(len(S_22) - expected_22) / expected_22 < 0.3
