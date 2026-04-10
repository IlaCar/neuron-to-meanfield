# Test Suite Plan — neuron-to-meanfield

## 1. Functional Map

### 1.1 Configuration & Model Loading (`utils/Sim_helper.py`, JSON files)

| Function | Location | Input | Output | Notes |
|---|---|---|---|---|
| `get_input_config()` | `utils/Sim_helper.py` | JSON path | dict (connections, rates, units) | Reads `input_config_*.json`, `network_config_*.json` |
| `get_syn_info()` | `utils/Sim_helper.py` | JSON path, idx | (Qe, Qi) as Brian2 quantities | Extracts quantal conductances |
| `get_network_config()` | `utils/Sim_helper.py` | JSON path | dict | Used for network topology |
| `load_spike_data()` | `utils/Sim_helper.py` | HDF5 path | dict with spikes, network_composition, external_input | For NN validation |

### 1.2 Neuron Model Setup (`utils/Brian_function_helper.py`)

| Function | Input | Output | Notes |
|---|---|---|---|
| `setting_simulation_Brian()` | idx, N_cell, neuron_model, json_path, curr_inj | Brian2 NeuronGroup | Creates AdEx neurons (FS/RS/RS_no_adapt) with synapses |
| `voltage_clamp_synapse()` | V_hold, idx, json_path, dt | StateMonitor | Voltage clamp protocol for synaptic model |
| `network_creation()` | conn_prob, pop_1, pop_2, Qe/Qi per pop, seed | (S_11, S_12, S_21, S_22) | Recurrent FS↔FS, FS↔RS, RS↔RS synapses |
| `extracting_pop_freq_and_std()` | sim_duration, monitors, N_pop, bin_size | (mean_rates, std_rates) per pop | Binned rate extraction for 2-pop network |
| `extracting_single_pop_freq_and_std()` | sim_duration, monitors, N_pop, bin_size, delay | (mean_rate, std_rate) | Single-population version used in TF data extraction |

### 1.3 Transfer Function Core (`utils/TF_helper.py`, duplicated in `transfer_function/validation/TF_helper.py`)

| Function | Input | Output | Notes |
|---|---|---|---|
| `get_params_model_SI()` | neuron_model, json_path | params_SI dict (all in SI units) | Converts nF→F, mV→V, nS→S, ms→s |
| `get_network_config()` | json_path | dict | **Duplicate** of Sim_helper version |
| `adding_K_params()` | neuron_params, network_config | params with K_e, K_i added | K = N × conn_prob |
| `membrane_potential_fluctuations()` | data (DataFrame), params | (mu_V, sig_V, tau_V, tau_V_norm) | Vectorized; Zerlaut 2018 eq. 5–17 |
| `membrane_potential_fluctuations_sim()` | f_e, f_i, params, w_ad | (mu_V, sig_V, tau_V, tau_V_norm) | Scalar version for MF integration |
| `est_thresh()` | data, mu_V, sig_V, tau_V, alpha | est_V_th array | Inverts erfc to estimate effective threshold |
| `eff_thresh()` | mu_V, sig_V, tau_V_norm, poly_params (10) | V_th_eff | 2nd-order polynomial in (μ,σ,τ) with reference points |
| `res_1_func()` | poly_params, mu_V, sig_V, tau_V_norm, est_V_th | MSE | Residual for stage-1 fit (threshold matching) |
| `res_2_func()` | poly_params, data, params, alpha | MSE | Residual for stage-2 fit (rate matching) |
| `TF_template()` | data, params, poly_params, alpha | F_out array | Full vectorized TF: (ν_e, ν_i) → F_out |
| `TF_template_sim()` | f_e, f_i, params, poly_params, alpha, w_ad | F_out scalar | Scalar TF for MF ODE integration |
| `get_mean_error_distribution()` | neuron_model, df_data, poly_params, params, alpha, unique_inh | error per inh level | Error sliced by inhibitory input |

### 1.4 TF Optimization (`utils/TF_alpha_opt.py`)

| Function | Input | Output | Notes |
|---|---|---|---|
| `run_fits()` | alpha, df_data, mu_V, sig_V, tau_V, tau_V_norm, params_SI | (mean_error, poly_params) | Two-stage fit: est_thresh → res_1 (SLSQP) → res_2 (Nelder-Mead) |
| `discrete_alpha_search()` | df_data, params, alpha range | (best_alpha, best_error, best_poly) | Grid search over α |

### 1.5 Mean-Field Simulation (currently in `transfer_function/validation/MF_val.py` notebook)

| Function | Input | Output | Notes |
|---|---|---|---|
| `simulate_MF_FS_RS()` | time, neuron_models, params, poly_params, alphas, network_config, driving_input | rates dict over time | Euler integration of 2-pop MF ODE |

### 1.6 Data Extraction Pipeline (`.py` scripts in `data_extraction/`)

| Script | What it does |
|---|---|
| `TF_data_extraction_FS.py` | Grid sweep: 61×61 (ν_e, ν_i), 50 FS neurons, records avg/std F_out |
| `TF_data_extraction_RS.py` | Same for RS with adaptation (includes delay for adaptation onset) |
| `TF_data_extraction_RS_no_adapt.py` | Same for RS without adaptation |

### 1.7 Plotting (`utils/Plots_helper.py`) — ~30 functions

Pure visualization. Not priority for unit testing but should be importable.

---

## 2. Issues Identified (Pre-Refactoring)

| Issue | Severity | Description |
|---|---|---|
| **TF_helper.py duplicated** | HIGH | `utils/TF_helper.py` and `transfer_function/validation/TF_helper.py` are near-identical copies. Will diverge. |
| **No package structure** | HIGH | Everything imported via `sys.path.append(parent_dir)`. No `__init__.py`, no proper module. |
| **Notebook-only logic** | MEDIUM | `simulate_MF_FS_RS()` is defined inside `MF_val.ipynb`. Must be extracted to library code. |
| **Magic numbers** | MEDIUM | Reference points in `eff_thresh` (mu_0=-60e-3, sig_0=4e-3, etc.) are hardcoded. |
| **Mixed concerns** | LOW | `Brian_function_helper.py` mixes neuron creation, network creation, AND rate extraction. |
| **No error handling** | LOW | Functions silently assume correct input shapes. |
| **pdb traces left in** | LOW | `pdb.set_trace()` calls in `extracting_pop_freq_and_std` and `extracting_single_pop_freq_and_std`. |

---

## 3. Refactoring Plan

### Phase 0: Package Structure

```
neuron-to-meanfield/
├── ntmf/                          # new package
│   ├── __init__.py
│   ├── config.py                  # JSON loading: get_input_config, get_network_config, get_params_model_SI
│   ├── neurons.py                 # Brian2 neuron creation: setting_simulation_Brian, voltage_clamp_synapse
│   ├── network.py                 # Network creation + rate extraction
│   ├── transfer_function.py       # All TF math: membrane_potential_fluctuations, eff_thresh, TF_template, etc.
│   ├── optimization.py            # Alpha search + two-stage fitting: run_fits, discrete_alpha_search
│   ├── meanfield.py               # simulate_MF_FS_RS (extracted from notebook)
│   └── plotting.py                # All Plots_helper functions (unchanged)
├── utils/                         # keep as-is for backward compat, but deprecated
├── tests/
│   ├── conftest.py                # shared fixtures
│   ├── test_config.py
│   ├── test_neurons.py
│   ├── test_network.py
│   ├── test_transfer_function.py
│   ├── test_optimization.py
│   ├── test_meanfield.py
│   └── test_integration.py
```

### Phase 1: Extract `simulate_MF_FS_RS` to library

Move from `MF_val.ipynb` → `ntmf/meanfield.py`. This is the only substantial logic trapped in a notebook.

### Phase 2: Deduplicate TF_helper.py

Delete `transfer_function/validation/TF_helper.py`. Have validation notebooks import from `ntmf.transfer_function`.

---

## 4. Test Plan

### Priority Principle
> **Fast iteration = fast tests.** Brian2 simulations are slow (seconds to minutes). Tests are stratified:
> - **Unit tests**: Pure math, no Brian2 — run in <1s total
> - **Slow unit tests**: Small Brian2 sims (1 neuron, ~100ms) — run in <30s  
> - **Integration tests**: Full pipeline stages — run in <5min
> - **Marked `@pytest.mark.slow`**: Full data extraction grid sweeps — for CI nightly only

### 4.1 `test_config.py` — Configuration Loading

| Test | What it validates | Speed |
|---|---|---|
| `test_get_params_model_SI_all_models` | Load FS.json, RS.json, RS_no_adapt.json; verify all keys present, SI unit conversion correct (e.g. C_m=0.2nF → 2e-10 F) | Instant |
| `test_get_params_model_SI_missing_model` | Raises ValueError on None model | Instant |
| `test_get_params_model_SI_invalid_model` | Raises ValueError on unknown model string | Instant |
| `test_get_params_model_SI_missing_file` | Raises ValueError on None json_file_name | Instant |
| `test_get_input_config` | Load input_config_TF.json, verify structure (connections, rates, units) | Instant |
| `test_get_network_config` | Load network_config_file_v0.json, verify keys | Instant |
| `test_get_syn_info` | Load FS.json → Qe, Qi are Brian2 quantities with correct nS values | Instant |
| `test_adding_K_params` | K_e = 8000×0.05 = 400, K_i = 2000×0.05 = 100 | Instant |
| `test_load_spike_data_file_not_found` | FileNotFoundError on missing HDF5 | Instant |

**Fixture**: `params_SI_FS`, `params_SI_RS`, `params_SI_RS_na` — loaded once per session.

### 4.2 `test_neurons.py` — Brian2 Neuron Creation

| Test | What it validates | Speed |
|---|---|---|
| `test_create_single_FS` | 1 FS neuron, no errors, has correct state variables | ~1s |
| `test_create_single_RS` | 1 RS neuron | ~1s |
| `test_create_single_RS_no_adapt` | 1 RS_no_adapt neuron | ~1s |
| `test_create_N_neurons` | N_cell=10, all neurons initialized at E_L | ~1s |
| `test_invalid_model_raises` | ValueError for unknown model | Instant |
| `test_adex_equations_parse` | AdEx_eqs string compiles in Brian2 without error | Instant |
| `test_FS_no_adaptation` | FS has b=0, verify from JSON | Instant |
| `test_RS_has_adaptation` | RS has b=0.1nA, verify from JSON | Instant |

### 4.3 `test_network.py` — Network Creation & Rate Extraction

| Test | What it validates | Speed |
|---|---|---|
| `test_network_creation_connectivity` | Create 2-pop network, verify synapse counts ≈ conn_prob × N | ~5s |
| `test_extracting_pop_freq_and_std_shape` | With mock spike data, output shapes are correct | Instant (mock) |
| `test_extracting_single_pop_freq_and_std_shape` | With mock spike data, output shapes correct | Instant (mock) |
| `test_rate_extraction_with_delay` | Delay parameter shifts the measurement window correctly | Instant (mock) |
| `test_rate_extraction_known_spikes` | Construct known spike train, verify computed rate matches expected | Instant (mock) |
| `test_network_creation_seed_reproducibility` | Same seed → same connectivity matrix | ~5s |

**Mock strategy**: For rate extraction, construct fake `SpikeMonitor`-like objects with `.i` and `.t` arrays.

### 4.4 `test_transfer_function.py` — Core TF Math (HIGHEST PRIORITY)

This is where the most value is. All tests are **pure math, no Brian2, instant**.

| Test | What it validates | Speed |
|---|---|---|
| **`membrane_potential_fluctuations`** | | |
| `test_mpf_zero_input` | f_e=f_i=0 → mu_V ≈ E_L, sig_V ≈ 0 | Instant |
| `test_mpf_pure_excitatory` | f_i=0, f_e>0 → mu_V > E_L (depolarized) | Instant |
| `test_mph_pure_inhibitory` | f_e=0, f_i>0 → mu_V < E_L (hyperpolarized) | Instant |
| `test_mpf_high_exc_low_inh` | mu_V close to E_e=0mV | Instant |
| `test_mpf_sig_V_increases_with_input` | Higher f_e → larger σ_V | Instant |
| `test_mpf_tau_V_positive` | τ_V > 0 for all non-zero inputs | Instant |
| `test_mpf_tau_V_norm_dimensionless` | τ_V_norm has no units (verify numerically) | Instant |
| `test_mpf_symmetric_vectorized_scalar` | Vectorized version matches scalar version for same inputs | Instant |
| **`eff_thresh`** | | |
| `test_eff_thresh_poly_zero_coeffs` | All poly params=0 except P_0 → eff_thresh = P_0 | Instant |
| `test_eff_thresh_at_reference_point` | At (mu_0, sig_0, tau_0) → eff_thresh = P_0 (no linear/quadratic terms) | Instant |
| `test_eff_thresh_linear_term` | Only P_mu≠0 → linear in mu_V | Instant |
| **`est_thresh`** | | |
| `test_est_thresh_inverse_of_eff_thresh` | For known F_out, mu_V, sig_V, tau_V, alpha: est_thresh ≈ eff_thresh with fitted params | Instant |
| `test_est_thresh_clipping` | Very high F_out → arg clips safely, no crash | Instant |
| **`TF_template`** | | |
| `test_TF_template_high_excitation` | Very high f_e → F_out > 0 (neuron fires) | Instant |
| `test_TF_template_high_inhibition` | Very high f_i, low f_e → F_out ≈ 0 (neuron silenced) | Instant |
| `test_TF_template_non_negative` | F_out ≥ 0 for all inputs (numerical safety) | Instant |
| `test_TF_template_known_params` | With validation JSON params, compute F_out for a few (ν_e, ν_i) points and verify against saved results | Instant |
| **`TF_template_sim`** | | |
| `test_TF_sim_matches_template` | Scalar TF_template_sim matches element of vectorized TF_template | Instant |
| **Regression Tests** | | |
| `test_TF_template_matches_saved_RS_fit` | Load RS_MF_params.json, compute TF for a grid, compare with original `results_RS.json` error | Instant |
| `test_TF_template_matches_saved_FS_fit` | Same for FS | Instant |
| **Edge Cases** | | |
| `test_mpf_exact_zero_input` | f_e=f_i=0 doesn't crash (epsilon guard) | Instant |
| `test_eff_thresh_extreme_mu_V` | mu_V at +20mV and -80mV don't crash | Instant |

**Fixture**: `params_SI_with_K` — params_SI with K_e=400, K_i=100, loaded once.

### 4.5 `test_optimization.py` — Fitting Procedure

| Test | What it validates | Speed |
|---|---|---|
| `test_run_fits_converges_synthetic` | Generate synthetic data from known params, run fits, verify recovery | ~10s |
| `test_res_1_func_zero_at_true_params` | With true poly_params, res_1 = 0 | Instant |
| `test_res_2_func_zero_at_true_params` | With true poly_params, res_2 ≈ 0 | Instant |
| `test_run_fits_invalid_alpha` | alpha=0 or negative doesn't crash (graceful handling) | Instant |
| `test_discrete_alpha_search_finds_best` | On small synthetic dataset, finds α within tolerance | ~5s |
| `test_run_fits_with_saved_data` | Load actual .dat file, run single alpha, verify error matches saved result | ~5s |

**Synthetic data strategy**: Pick known poly_params + alpha, use TF_template to generate F_out, add small noise, then fit.

### 4.6 `test_meanfield.py` — MF Simulation

| Test | What it validates | Speed |
|---|---|---|
| `test_simulate_MF_constant_drive` | Constant input → rates converge to steady state | Instant |
| `test_simulate_MF_zero_drive` | Zero external drive → rates → 0 | Instant |
| `test_simulate_MF_high_exc` | High excitatory drive → non-zero rates | Instant |
| `test_simulate_MF_tau_f_dynamics` | Verify time constant of rate relaxation matches τ_f | Instant |
| `test_simulate_MF_rates_non_negative` | All rates ≥ 0 at all times | Instant |
| `test_simulate_MF_FS_silenced` | FS receives no excitation → FS rate → 0 | Instant |
| `test_simulate_MF_matches_validation_params` | Run with FS_MF_params.json + RS_MF_params.json, verify steady-state rates reasonable | Instant |

### 4.7 `test_integration.py` — End-to-End Pipeline

| Test | What it validates | Speed |
|---|---|---|
| `test_single_neuron_spikes_under_current` | 1 neuron, step current → spike count > 0 | ~2s |
| `test_TF_data_extraction_small_grid` | 3×3 grid of (ν_e, ν_i), 5 neurons, 500ms sim → produces valid .dat file | ~30s |
| `test_fitting_on_extracted_data` | Run data extraction (3×3) → fit → verify error < threshold | ~60s |
| `test_MF_sim_with_fitted_params` | Fit 3×3 data → run MF sim → rates are finite and non-negative | ~60s |
| `test_full_pipeline_saved_data` | Load real .dat files → fit → MF sim → compare with known steady-state | ~10s |

---

## 5. Implementation Order

### Step 1: Create `tests/` with `conftest.py` fixtures
- Fixtures for params_SI (all 3 models), network configs, small synthetic DataFrames
- `@pytest.mark.slow` marker definition

### Step 2: `test_config.py` + `test_transfer_function.py`
- These are pure Python/math, no refactoring needed, highest value
- Covers all the core analytical machinery

### Step 3: `test_neurons.py` + `test_network.py`
- Requires Brian2 but uses small simulations
- Mock spike monitors for rate extraction tests

### Step 4: `test_optimization.py`
- Depends on test_transfer_function being solid
- Synthetic data tests ensure fitting procedure works

### Step 5: Extract `simulate_MF_FS_RS` → `ntmf/meanfield.py`
- Minimal refactoring: just move the function
- Add `test_meanfield.py`

### Step 6: `test_integration.py`
- End-to-end tests using small grids

### Step 7 (optional): Full package refactor into `ntmf/`
- Deduplicate TF_helper.py
- Update all notebooks to import from `ntmf`
- Remove `sys.path.append` hacks

---

## 6. Questions for Prioritization

1. **Should the integration tests use the actual existing .dat simulation data files?** (They exist in `transfer_function/data_extraction/simulations/`). Using them means tests run instantly but are coupled to historical data.

2. **Is the package refactor (Step 7) a priority now, or should we keep tests working with the current `utils/` structure?** I'd recommend keeping current structure and testing against it first — refactor after green tests.

3. **Should `simulate_MF_FS_RS` be generalized to N populations, or kept as 2-population (FS+RS) for now?**

4. **The adaptation variable `w_ad` is currently hardcoded to 0 in `membrane_potential_fluctuations` and `TF_template_sim`.** Is adaptation support planned? Should tests cover the w_ad≠0 codepath?

5. **Any preference on test runner config?** I'd suggest `pytest` with `pytest.ini` or `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   markers = ["slow: slow tests requiring full Brian2 simulations"]
   ```
