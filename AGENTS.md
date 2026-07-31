# AGENTS.md

## Project Overview

**neuron-to-meanfield** bridges scales in computational neuroscience: from single-neuron AdEx models to population-level mean-field descriptions. The pipeline:

1. **Simulate** individual neurons (FS, RS, RS_no_adapt) under varying synaptic input rates
2. **Extract** transfer functions: maps (ν_exc, ν_inh) → F_out (output firing rate)
3. **Fit** an analytical TF (10 polynomial params + scaling α) to the extracted data
4. **Validate** by embedding the TF in a 2-population mean-field ODE and comparing against full neural-network simulations

## Repository Structure

```
ntmf/                          # Canonical tested package (import from here)
  config.py                    # JSON loading, SI unit conversion, K-param computation
  neurons.py                   # Brian2 AdEx neuron group creation
  network.py                   # 2-population network creation, spike rate extraction
  transfer_function.py         # Membrane fluctuation stats, eff_thresh, TF_template (Zerlaut 2018)
  optimization.py              # Two-stage fitting (SLSQP + Nelder-Mead), discrete α search
  meanfield.py                 # MFModel class, simulate_MF_FS_RS (Euler), simulate_MF_FS_RS_ode (scipy RK45)
  validation.py                # Build driving input from HDF5, bin NN spikes, compare MF vs NN

utils/                         # Legacy (deprecated) — notebooks still import Plots_helper from here
  Plots_helper.py              # ~1000 lines of matplotlib helpers — kept for backward compat
  Brian_function_helper.py     # Replaced by ntmf.neurons + ntmf.network
  Sim_helper.py                # Replaced by ntmf.config
  TF_helper.py                 # Replaced by ntmf.transfer_function
  TF_alpha_opt.py              # Replaced by ntmf.optimization

config/                        # JSON configuration files
  input_config_TF.json         # TF data extraction sweep parameters
  network_config_file_v0.json  # Network topology (no Q_e/Q_i in external_input)
neuron_models/AdEx/            # Neuron model parameters (FS.json, RS.json, RS_no_adapt.json)
transfer_function/             # Notebooks and scripts for TF pipeline
  data_extraction/             # Grid-sweep scripts + saved .dat simulation data
  fitting/                     # TF_fitting_RS.ipynb + results/
  analysis/                    # Parameter distribution analysis notebooks
  validation/                  # MF_val.ipynb, NN validation, MF_vs_NN_validation.ipynb, saved MF params
single_neuron_simulation/      # Single-neuron experiment notebooks
neural_network_simulation/     # Full network simulation notebooks + saved spikes_data.h5
synaptic_model/                # Voltage clamp notebook

tests/                         # Pytest suite (99 tests, 91% coverage)
  conftest.py                  # Session-scoped fixtures (paths, params, spike data)
  test_config.py               # JSON loading, SI conversion, K params (20 tests)
  test_neurons.py              # Brian2 neuron creation (10 tests)
  test_network.py              # Spike matrix, rate extraction (7 tests)
  test_transfer_function.py    # Core TF math, regression on saved data (27 tests)
  test_optimization.py         # Fitting, α search (8 tests)
  test_meanfield.py            # Euler + ODE MF simulation (13 tests)
  test_integration.py          # Full pipeline with real .dat files (4 tests)
  test_validation.py           # MF-vs-NN comparison (10 tests)

.github/workflows/
  tests.yml                    # Fast tests on push/PR (~30s on CI)
  tests-nightly.yml            # All tests including slow, daily cron
```

## Key Conventions

### Package: `ntmf/` is the source of truth
All core logic lives in `ntmf/`. Notebooks import from `ntmf`, not `utils/`. The `utils/` directory is deprecated — only `Plots_helper.py` remains for notebook visualization.

### Units: SI everywhere
All numerical parameters in `ntmf/` are in SI units (volts, siemens, farads, seconds, hertz). JSON configs store human-readable units (mV, nS, nF, ms); `get_params_model_SI()` converts on load.

### Mean-field model: 2-population (FS + RS), no adaptation
- FS = fast-spiking inhibitory interneurons (b=0)
- RS = regular-spiking excitatory pyramidal neurons (b=0.1 nA, but w_ad=0 in TF)
- `MFModel` class bundles all network constants; `rhs(t, y)` is the ODE right-hand side
- Two integrators: `simulate_MF_FS_RS` (Euler) and `simulate_MF_FS_RS_ode` (scipy RK45)

### Transfer function formula
```
F_out = α · erfc((V_th_eff − μ_V) / (√2 · σ_V)) / (2 · τ_V)
```
where `V_th_eff` is a 2nd-order polynomial in (μ_V, σ_V, τ_V_norm) with 10 free parameters and reference points μ₀=-60mV, σ₀=4mV, τ₀=0.5s.

### Network structure
- N_RS=8000, N_FS=2000, conn_prob=0.05 → K_e=400, K_i=100
- External Poisson input with configurable rates per population
- Quantal conductances: Q_e from neuron model JSONs, Q_i likewise

### Testing
- Run fast tests: `python -m pytest tests/ -k "not slow"`
- Run all tests: `python -m pytest tests/`
- Coverage: `python -m pytest tests/ -k "not slow" --cov=ntmf --cov-report=term-missing`
- Brian2-dependent tests are marked `@pytest.mark.slow`
- Test data uses committed files: `.dat` files in `transfer_function/data_extraction/simulations/`, `spikes_data.h5` in `neural_network_simulation/simulations/test_0/`

### Notebooks
- Notebooks import `from ntmf.<module> import ...` plus `from utils.Plots_helper import *`
- No `sys.path.append` hacks
- Execute verification: `python -m nbconvert --execute --to notebook <nb>.ipynb --output <nb>_executed.ipynb`
- `*_executed.ipynb` and notebook-derived `.py` files are gitignored

## Common Tasks

### Add a new test
Add to the appropriate `tests/test_*.py`. Use existing fixtures from `conftest.py` (e.g., `validation_dir`, `neuron_models_dir`). Mark Brian2-dependent tests `@pytest.mark.slow`.

### Modify the TF pipeline
Edit `ntmf/transfer_function.py`. The vectorized path (`TF_template`) operates on DataFrames; the scalar path (`TF_template_sim`) is used by the MF ODE integrator. Both must stay consistent — tests verify this.

### Modify the MF simulation
Edit `ntmf/meanfield.py`. The `MFModel` class holds all constants; add new parameters to `__init__` and use them in `rhs()`. Both Euler and ODE integrators share the same `rhs()`, so changes propagate automatically.

### Add a new notebook
Create it in the appropriate directory. Import from `ntmf.*`. For plotting, use `from utils.Plots_helper import *`. Do not add `sys.path.append` calls.

## Dependencies

- **Brian2** 2.10+ — neural simulations (C++ codegen, needs g++)
- **NumPy**, **SciPy** — numerical math, ODE integration, optimization
- **Matplotlib**, **Seaborn** — visualization
- **Pandas** — TF data handling
- **h5py** — spike data I/O
- **pytest**, **pytest-cov** — testing
