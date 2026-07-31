# Implementation Plan: Notebooks Rewrite, ODE Integration & MF-vs-NN Validation

## Part 1: Notebook Rewrite Strategy

### 1.1 Import Mapping

Every notebook currently does:
```python
import sys, os
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..'))  # (or '../..')
sys.path.append(parent_dir)
```
then imports from `utils`. The rewrite replaces these with clean `ntmf` imports.

| Old import | New import |
|---|---|
| `from utils.Brian_function_helper import *` | `from ntmf.neurons import setting_simulation_Brian, voltage_clamp_synapse` |
| | `from ntmf.network import network_creation, extracting_pop_freq_and_std, extracting_single_pop_freq_and_std` |
| `from utils.Sim_helper import *` | `from ntmf.config import get_input_config, get_network_config, get_syn_info, load_spike_data` |
| `from utils.TF_helper import *` | `from ntmf.transfer_function import *` |
| `from utils.TF_alpha_opt import *` | `from ntmf.optimization import run_fits, discrete_alpha_search` |
| `from utils.Plots_helper import *` | **Keep as-is** (Plots_helper.py stays in `utils/` — it's pure visualization, not worth moving yet) |

**Key decision**: `utils/Plots_helper.py` is ~1000 lines of matplotlib functions, used with `from utils.Plots_helper import *`. Moving it would require updating 20+ notebooks for zero test coverage benefit. **Leave it in `utils/`** for now.

### 1.2 Notebook-by-Notebook Plan

**Tier 1: Transfer function pipeline (highest priority)**

| Notebook | Location | What changes |
|---|---|---|
| `TF_fitting_RS.ipynb` | `transfer_function/fitting/` | Replace `utils.*` imports → `ntmf.*`; delete `sys.path` hack; replace `TF_helper` local import with `ntmf.transfer_function`; replace `TF_alpha_opt` with `ntmf.optimization` |
| `MF_val.ipynb` | `transfer_function/validation/` | Replace imports; **delete inline `simulate_MF_FS_RS` definition** → import from `ntmf.meanfield`; delete `from TF_helper import *` (local duplicate); use `ntmf.transfer_function` |
| `NN_FS_RS_val_data.ipynb` | `transfer_function/validation/` | Replace imports; same pattern |
| `Analysing_extracted_data.ipynb` | `transfer_function/data_extraction/` | Replace imports |
| 4 analysis notebooks | `transfer_function/analysis/` | Replace `utils.TF_helper` → `ntmf.transfer_function` |

**Tier 2: Data extraction scripts**

| File | Changes |
|---|---|
| `TF_data_extraction_FS.py` | Replace imports → `ntmf.neurons`, `ntmf.config`, leave Plots_helper |
| `TF_data_extraction_RS.py` | Same |
| `TF_data_extraction_RS_no_adapt.py` | Same |

**Tier 3: Single neuron & network notebooks** (lower priority, no TF math)

| Notebook | Changes |
|---|---|
| `single_neuron_synaptic_stimulation.ipynb` | Replace imports |
| `single_neuron_current_injection.ipynb` | Replace imports |
| `Figure1_A.ipynb`, `Figure1_B.ipynb`, `Figure2_B.ipynb` | Replace imports |
| `NN_FS_RS.ipynb`, `NN_FS_RS_no_adapt.ipynb` | Replace imports |
| `Figure3_A.ipynb`, `Figure3_B.ipynb`, `Analysing_NN.ipynb` | Replace imports |
| `Voltage_clamp.ipynb` | Replace imports |

### 1.3 Implementation Order for Notebook Rewrite

1. Add `utils/Plots_helper.py` compatibility: no changes needed since it stays
2. Rewrite **Tier 1** notebooks (6 files) — the TF pipeline
3. Rewrite **Tier 2** scripts (3 files) — data extraction
4. Rewrite **Tier 3** notebooks (remaining) — lower priority
5. After rewrite, re-run `nbconvert --execute` on key notebooks to verify

### 1.4 Validation of Rewritten Notebooks

For each rewritten notebook:
- `python -m nbconvert --execute --to notebook <notebook>.ipynb --output <notebook>_executed.ipynb`
- Compare key outputs (plots, saved JSON) against pre-rewrite results
- Run test suite to confirm nothing broke

---

## Part 2: scipy ODE Integration for MF Simulation

### 2.1 Current State

The MF simulation in `ntmf/meanfield.py` uses manual Euler integration:

```python
for t in range(1, n_steps):
    # ... compute effective input rates from recurrent + external ...
    F_RS = TF_template_sim(f_e=nu_eff_exc_RS, f_i=nu_eff_inh_RS, ...)
    F_FS = TF_template_sim(f_e=nu_eff_exc_FS, f_i=nu_eff_inh_FS, ...)
    rates["RS"][t] = rates["RS"][t-1] + dt/tau_f * (F_RS - rates["RS"][t-1])
    rates["FS"][t] = rates["FS"][t-1] + dt/tau_f * (F_FS - rates["FS"][t-1])
```

**Problem**: The external driving input is time-varying (step changes). `scipy.integrate.solve_ivp` needs the RHS as a callable `f(t, y)`, so we need to interpolate the driving input.

### 2.2 ODE Formulation

State vector: **y = [ν_FS, ν_RS]** (firing rates in Hz)

```
dν_FS/dt = (TF_FS(ν_eff_e_FS, ν_eff_i_FS) − ν_FS) / τ_f
dν_RS/dt = (TF_RS(ν_eff_e_RS, ν_eff_i_RS) − ν_RS) / τ_f
```

where the effective rates depend on current ν_FS, ν_RS, and time-varying external input.

### 2.3 Implementation Plan

**File: `ntmf/meanfield.py`**

1. Create an `MFModel` class that bundles all parameters (network structure, quantal conductances, TF params) into a single object. This avoids recomputing constants at every time step.

2. Add a `_rhs(t, y)` method that:
   - Interpolates the external driving input at time `t` (using `np.interp`)
   - Computes effective input rates (external + recurrent)
   - Evaluates TF_template_sim for each population
   - Returns `[dν_FS/dt, dν_RS/dt]`

3. Add `simulate_MF_FS_RS_ode()` that:
   - Pre-computes driving input interpolation arrays
   - Calls `scipy.integrate.solve_ivp(_rhs, [t0, tf], y0, t_eval=time, method='RK45')`
   - Returns `{"FS": sol.y[0], "RS": sol.y[1]}`

4. Keep `simulate_MF_FS_RS()` (Euler) for backward compatibility

**Key design choice**: Use `scipy.integrate.solve_ivp` with `method='RK45'` (default adaptive Runge-Kutta) for accuracy, and `t_eval=time` to get output at the exact time points needed for comparison.

### 2.4 Interpolation of Driving Input

The driving input has piecewise-constant (step) structure. Use `np.interp` with the pre-computed time arrays:

```python
from scipy.interpolate import interp1d

# Pre-build interpolators for each input channel
interp_exc_FS = interp1d(time, driving_input["excitatory"]["FS"], 
                          kind='previous', fill_value=0.0, bounds_error=False)
# ... etc for exc_RS, inh_FS, inh_RS
```

Using `kind='previous'` preserves the step-function nature of the input.

### 2.5 Tests for ODE Version

Add to `tests/test_meanfield.py`:

| Test | What it validates |
|---|---|
| `test_ode_matches_euler_constant_drive` | Same constant input → ODE and Euler converge to same steady state |
| `test_ode_matches_euler_step_input` | Step input → ODE and Euler give similar trajectories |
| `test_ode_rates_non_negative` | Rates ≥ 0 at all output times |
| `test_ode_output_shape` | Output shapes match input time vector |
| `test_ode_tau_f_dynamics` | Relaxation time constant matches τ_f |

---

## Part 3: MF-vs-NN Validation

### 3.1 Existing Data

We have one saved NN simulation: `neural_network_simulation/simulations/test_0/spikes_data.h5`

This contains:
- 10-second simulation, 2000 FS + 8000 RS neurons
- Background input at 0.3 Hz throughout
- Excitatory step to FS at 5 Hz during [1, 2]s
- Excitatory step to RS at 3 Hz during [3, 4]s
- Inhibitory step to FS at 10 Hz during [5, 6]s
- Inhibitory step to RS at 30 Hz during [7, 8]s
- Network config: `network_config_file_v0.json` (conn_prob=0.05, N_ext_exc=8000, N_ext_inh=2000)

### 3.2 Validation Procedure

**Goal**: Run the MF ODE simulation with the **exact same protocol** and compare population firing rates against the NN spike data.

**Steps**:

1. **Load NN spike data** from HDF5
2. **Compute NN population rates** by binning spikes (same bin_size as used in MF output)
3. **Build MF driving input** that matches the NN protocol (same step timing and rates)
4. **Run MF simulation** with `simulate_MF_FS_RS_ode()`
5. **Compare** MF vs NN rates:
   - Overlap plots (MF line vs NN binned rates)
   - Correlation coefficient
   - Relative error in steady-state periods
   - RMSE over time

### 3.3 Implementation Plan

**File: `ntmf/validation.py`** (new module)

```python
def build_driving_input_from_hdf5(hdf5_path, time, network_config):
    """Reconstruct the driving input arrays from saved HDF5 metadata."""
    ...

def compute_nn_population_rates(spike_data, N_FS, N_RS, sim_duration, bin_size):
    """Bin NN spikes into population firing rate traces."""
    ...

def compare_mf_nn(mf_rates, nn_rates, time, bin_size):
    """Quantitative comparison between MF and NN rate traces."""
    return {
        'rmse_FS': ...,
        'rmse_RS': ...,
        'corr_FS': ...,
        'corr_RS': ...,
        'steady_state_error_FS': ...,
        'steady_state_error_RS': ...,
    }
```

**File: `tests/test_validation.py`** (new test file)

| Test | What | Speed |
|---|---|---|
| `test_build_driving_input_matches_protocol` | Reconstruct driving input from HDF5 → verify step timings | Instant |
| `test_nn_rate_extraction` | Load spikes → bin → verify rate shapes and ranges | Instant |
| `test_mf_nn_constant_period` | Compare MF vs NN rates during background-only period (t ∈ [0, 1]) | Instant |
| `test_mf_nn_excitation_period` | Compare during excitatory input period | Instant |
| `test_mf_nn_correlation` | Pearson correlation of MF vs NN rates > threshold | Instant |
| `test_mf_nn_steady_state_within_factor` | Steady-state rates within 2× of each other | Instant |

**File: `transfer_function/validation/MF_vs_NN_validation.ipynb`** (new notebook)

Interactive version showing:
- Side-by-side rate plots
- Scatter plots (MF rate vs NN rate)
- Error analysis per stimulus period
- Summary statistics table

### 3.4 Key Design Decisions

1. **Normalization**: The MF effective input rates use a reference (K_ref_exc, K_ref_inh) normalization. The validation must use the exact same reference as was used during TF fitting. This is already captured in `simulate_MF_FS_RS`.

2. **Missing Q_e/Q_i in v0 config**: The `network_config_file_v0.json` does NOT have `Q_e`/`Q_i` in the external_input section (unlike `network_config_file_val.json`). The NN simulation code uses Qe/Qi from the neuron model JSONs. The MF validation must handle both config formats.

3. **Bin alignment**: NN rates are computed by binning spikes. MF rates are continuous. Must use the same bin_size and align bin edges.

4. **What "matching" means**: The MF model is deterministic; the NN is stochastic (Poisson input). We don't expect exact matching. Validation criteria:
   - Steady-state rates within a factor of 2×
   - Temporal dynamics (rise/fall times) qualitatively similar
   - Pearson r > 0.7 when comparing binned rates

### 3.5 New NN Simulation for Validation

The existing `test_0` data uses `network_config_file_v0.json` which doesn't have Q_e/Q_i. We should also run a new NN simulation using `network_config_file_val.json` (which does have Q_e/Q_i) for a cleaner validation. This would be a new notebook: `transfer_function/validation/run_NN_validation_sim.ipynb`.

---

## Execution Order

1. **Part 2 first**: Implement `MFModel` class + `simulate_MF_FS_RS_ode()` in `ntmf/meanfield.py`; add ODE tests; verify all existing tests still pass
2. **Part 3 second**: Implement `ntmf/validation.py`; run NN simulation with val config; write comparison tests
3. **Part 1 last**: Rewrite notebooks (safe refactor since tests cover the library code)
