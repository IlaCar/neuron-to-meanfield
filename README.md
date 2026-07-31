# BRIDGE
![BRIDGE graphical abstract](utils/graphical_abstract.png)

**A Computational Workflow from Single Neurons to Network of Mean-Field Models**

This repository accompanies the manuscript *"BRIDGE: A Computational Workflow from
Single Neurons to Network of Mean-Field Models"* (submitted). It provides an
open-source Python pipeline for the **bottom-up reconstruction and validation of
mean-field (MF) models** directly from single-neuron electrophysiology.

Beyond fitting, BRIDGE come with with a **diagnostic layer** that asks whether the reconstructed models can be interpreted and trusted: whether the fitted transfer-function coefficients are individually identifiable, whether the identifiability structure is a property of the formalism or of a particular population, and where the mean-field description remains a faithful surrogate for the spiking network.

The workflow proceeds in stages:

1. **Single-neuron characterisation** — fit spiking neuron models (AdEx) to
   experimental spike features and characterise their input–output behaviour.
2. **Spiking-network generation** — build the spiking neural network with recurrent connections.
3. **Transfer-function derivation** — generate transfer-function (TF) data
   from spiking simulations and fit a semi-analytical TF for each population.
4. **Mean-field model** — assemble the fitted TFs into a mean-field model and validate
   it against ground-truth spiking-network simulations.
5. **Network of mean-fields** — couple validated mean-field nodes into networks of
   interacting populations.

## Repository layout
The repository is organised by modelling stage. The reusable implementation is contained in ntmf/; the remaining folders provide model definitions,configuration files, simulation scripts, analysis notebooks, and retained outputs.
```text
neuron-to-meanfield/
├── ntmf/                         # Core Python package used across the workflow
│   ├── config.py                 # JSON loading, unit conversion, and data loading
│   ├── neurons.py                # AdEx and EGLIF single-neuron implementations
│   ├── network.py                # Spiking-network construction and rate extraction
│   ├── transfer_function.py      # Membrane statistics, effective threshold, semi-analytical TF
│   ├── optimization.py           # Two-stage fitting (SLSQP + Nelder-Mead) and alpha search
│   ├── meanfield.py              # Mean-field model and integrators (Euler, scipy RK45)
│   ├── phase_plane.py            # Nullclines, fixed points, and stability analysis
│   └── validation.py             # Mean-field vs spiking-network comparison
│
├── neuron_models/                # Cellular and synaptic parameters in JSON format
│   ├── AdEx/                     # FS, adaptive RS, and non-adaptive RS models
│   └── EGLIF/                    # EGLIF cerebellar neuron example (GoC)
│
├── config/                       # Network and external-input configuration files
├── AdEx_parameters_search/       # Optional fitting of AdEx models to spike features
├── single_neuron_simulation/     # Current- and synaptic-stimulation notebooks
├── synaptic_model/               # Synaptic characterisation by voltage clamp
├── neural_network_simulation/    # Recurrent spiking-network simulations and analysis
│
├── transfer_function/            # TF data extraction, fitting, and analysis notebooks
│   ├── data_extraction/          # Numerical TF generation from spiking simulations
│   ├── fitting/                  # Semi-analytical TF fitting and retained parameters
│   └── analysis/                 # Fit quality and parameter-identifiability analyses
│
├── mean_field/
│   ├── phase_plane/              # Interactive phase-plane analysis
│   └── validation/               # MF-versus-spiking-network validation workflows
│
├── network_of_mean_field/        #  # Coupled network of mean-field nodes
├── utils/                        # Plotting and figure-assembly utilities
├── tests/                        # Unit and integration tests
├── validate_notebooks.py         # Headless notebook-execution checker
└── pyproject.toml                # Package metadata and dependencies
```

## How the folders map onto the workflow

| Stage | Main locations | Role in the workflow | Main outputs |
|---|---|---|---|
| 0. Optional neuronal parameter search | `AdEx_parameters_search/` | Extract experimental spike features and screen AdEx parameter sets using Sobol sampling. | Ranked neuronal models and fitting-error summaries in JSON/PDF format. |
| 1. Single-neuron and synaptic characterisation | `neuron_models/`, `single_neuron_simulation/`, `synaptic_model/` | Define the cellular model and inspect its response to current injection, synaptic activation, and voltage clamp. | Voltage traces, firing-rate curves, and characterised synaptic responses. |
| 2. Spiking-network simulation | `config/`, `neural_network_simulation/` | Specify network composition, recurrent connectivity, and external drive, then simulate the full spiking network. | Spike trains, population firing rates, and optional HDF5 simulation files. |
| 3. Transfer-function reconstruction | `transfer_function/data_extraction/`, `transfer_function/fitting/` | Sample excitatory and inhibitory input-rate grids, measure neuronal output rates, and fit the semi-analytical TF. | Numerical TF datasets and retained fitted coefficients in `transfer_function/fitting/results/`. |
| 4. TF diagnostics | `transfer_function/analysis/` | Examine fitting errors, parameter distributions, identifiability, and cross-population differences. | Diagnostic plots and summaries of robust and weakly constrained parameters. |
| 5. Mean-field analysis and validation | `mean_field/phase_plane/`, `mean_field/validation/` | Construct the MF model from fitted TFs, analyse its phase plane, and compare MF predictions with spiking-network simulations. | Nullclines, fixed points, stability information, residual maps, and MF-versus-network comparisons. |
| 6. Network of mean fields | `network_of_mean_field/` | Define and simulate multiple coupled MF populations from a JSON topology. | Population-rate trajectories, network diagrams, and signal-propagation analyses. |

## Suggested workflow

The stages can be followed end to end, but each folder can also be used
independently when the required upstream outputs are already available.
Pre-generated neuronal models, numerical TF datasets, and fitted TF parameters are included for the examples used in the manuscript.

### 1. Select or fit a neuronal model

Use an existing model from `neuron_models/`, or perform a new parameter search with:

- `AdEx_parameters_search/extracting_features/Features_extraction.ipynb`
- `AdEx_parameters_search/screening/AdEx_search.py`

The resulting parameters can then be stored in the same JSON structure as the
models provided in `neuron_models/AdEx/` or `neuron_models/EGLIF/`.

### 2. Characterise the single neuron and its synapses

The principal entry points are:

- `single_neuron_simulation/single_neuron_current_injection.ipynb`
- `single_neuron_simulation/single_neuron_synaptic_stimulation.ipynb`
- `synaptic_model/Voltage_clamp.ipynb`

The EGLIF example is provided in the two notebooks prefixed with
`EGLIF_single_neuron_`. Figure-specific notebooks reproduce the corresponding
single-neuron panels used in the manuscript.

### 3. Build and simulate the spiking network

Network composition and external drive are read from JSON files in `config/`.
The main AdEx examples are:

- `neural_network_simulation/NN_FS_RS.ipynb` for adaptive RS neurons;
- `neural_network_simulation/NN_FS_RS_no_adapt.ipynb` for non-adaptive RS neurons;
- `neural_network_simulation/Analysing_NN.ipynb` for loading and visualising saved
  spiking-network activity.

The scripts and notebooks use the model constructors in `ntmf.neurons`, the
network-building utilities in `ntmf.network`, and the loaders in `ntmf.config`.

### 4. Generate and fit population transfer functions

Run the appropriate extraction script in `transfer_function/data_extraction/` to
measure output firing rates over a grid of presynaptic input rates. For example:

```bash
cd transfer_function/data_extraction
python TF_data_extraction_FS.py
python TF_data_extraction_RS.py
```

The resulting `.dat` files are read by the notebooks in
`transfer_function/fitting/`. Separate notebooks are provided for FS, adaptive
RS, non-adaptive RS, and the explicit-adaptation formulation. Retained fitting
results are stored in `transfer_function/fitting/results/`.

### 5. Analyse and validate the reconstructed mean field

Use `mean_field/phase_plane/MF_phase_plane.ipynb` to inspect nullclines, fixed
points, and local stability. Use
`mean_field/validation/MF_NN_comparison.ipynb` to compare the MF predictions with
the corresponding spiking-network simulations across input conditions.

The validation stage combines the fitted TF parameters with the same neuronal,
synaptic, connectivity, and external-input definitions used by the spiking model.
This preserves the parameter correspondence between the two scales.

### 6. Couple multiple mean-field populations

Network topologies are defined by JSON files in `network_of_mean_field/`. Each
file specifies the populations and their directed edges, including connection
probability, strength, receptor type, and delay. The main entry points are:

- `network_of_mean_field/network_MF_signal_propagation.ipynb` for interactive
  construction, stimulation, and visualisation;
- `network_of_mean_field/run_network_MF_simulation.py` for a script-based example.

## Citation

A citation and archived release DOI will be added on publication. Until then, please
cite the repository and the accompanying manuscript (in preparation).
