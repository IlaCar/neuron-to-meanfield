import matplotlib.pyplot as plt
import h5py
import sys
import os

# Adding the project folder to sys.path
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..', '..'))
sys.path.append(parent_dir)
# We do this so that we can directly import files in the utils folder


from ntmf.neurons import *
from utils.Plots_helper import *
from ntmf.config import *
from ntmf.network import *
from ntmf.meanfield import *
from ntmf.transfer_function import *

saving_hdf5 = True

# --- Timing ---
# Structurally identical to NN_heatmap_data.py:
#   input is present only over [input_start, input_end],
#   and the STEADY STATE is measured over [ss_start, input_end], i.e. one
#   settling second is discarded after the input turns on (delay = 1 s),
#   exactly as in the TF pipeline, so the RS adaptation onset overshoot is
#   not counted.
T = 5
dt = 1e-3
time = np.arange(0, T, dt)
input_start, input_end = 1.0, 4.0               # input ON window (s)
ss_start, ss_end = 2.0, 4.0                     # steady-state analysis window (s)
win = (time >= ss_start) & (time <= ss_end)     # averaging mask ([2, 6] s)
input_mask = (time >= input_start) & (time < input_end)

# metadata describing the input span (kept for the h5 external_input group)
p_start, p_end = input_start, input_end

exc_interval_FS = (p_start, p_end, 'FS')
exc_interval_RS = (p_start, p_end, 'RS')
exc_intervals = [exc_interval_FS, exc_interval_RS] # used for plotting

inh_interval_FS = (p_start, p_end, 'FS')
inh_interval_RS = (p_start, p_end, 'RS')
inh_intervals = [inh_interval_FS, inh_interval_RS] # used for plotting

data_folder = '../../neuron_models/AdEx'
idx = 0 # in this case only one model per cell type is provided
# idx of the fitted TF params
FS_idx = 0
RS_idx = 0

# Define the ranges for the loops
exc_freq_range = np.arange(0, 31, 1) 
inh_freq_range = np.arange(0, 31, 1) 

# --- Network Setup  ---
config_file_name = '../../config/network_config_file_val_heatmap.json'
network_config = get_network_config(json_file_name=config_file_name)

neuron_model = 'FS'
params_SI_FS = get_params_model_SI(neuron_model = neuron_model,
                             json_file_name = os.path.join(data_folder, neuron_model + '.json'))

neuron_model = 'RS'
params_SI_RS = get_params_model_SI(neuron_model = neuron_model,
                             json_file_name = os.path.join(data_folder, neuron_model + '.json'))

params_SI_FS = adding_K_params(neuron_params = params_SI_FS,
                    network_config = network_config)

params_SI_RS = adding_K_params(neuron_params = params_SI_RS,
                    network_config = network_config)

with open('10_best_params_TF_FS.json', 'r') as f:
    data_FS = json.load(f)

with open('10_best_params_TF_RS.json', 'r') as f:
    data_RS = json.load(f) 

# --- OU Process Generation ---
tau_ou, sigma_ou = 0.05, 2.0

for exc_val in exc_freq_range:
    for inh_val in inh_freq_range:
        
        print(f"Running simulation: Exc={exc_val}Hz, Inh={inh_val}Hz")
       
        def windowed_ou(mu):
            ou = generate_ou_process(time, dt, mu, tau_ou, sigma_ou)
            ou[~input_mask] = 0.0               # input ON only over [1, 6] s
            return ou

        driving_input_ou = {
        'excitatory': {
            'FS': windowed_ou(exc_val),
            'RS': windowed_ou(exc_val)
            },
        'inhibitory': {
            'FS': windowed_ou(inh_val),
            'RS': windowed_ou(inh_val)
            }
        }

        out_ou = simulate_MF_FS_RS_ode(
            time = time,
            neuron_models = ['FS', 'RS'],
            params = {'FS': params_SI_FS, 'RS': params_SI_RS},
            poly_params = {'FS': data_FS[FS_idx]['polynomial_params'],
                           'RS': data_RS[RS_idx]['polynomial_params']},
            alphas = {'FS': data_FS[FS_idx]['alpha'],
                     'RS': data_RS[RS_idx]['alpha']},
            network_config = network_config,
            driving_input = driving_input_ou
        )
        

        # Update naming convention for plots
        plot_name_base = f"exc_{exc_val}_inh_{inh_val}"
        sim_path = os.path.join('simulations/val_MF_[0_30_1]')
        if not os.path.exists(sim_path): os.makedirs(sim_path)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        for pop, color in zip(['FS','RS'], [color_palette["FS"], color_palette["RS"]]):
            ax.plot(time, out_ou[pop], label=f'{pop}, [solve_ivp]', color=color)
        ax.plot(time, driving_input_ou['excitatory']['FS'], color=syn_colors["E"], label='exc_input', alpha=0.3)
        ax.plot(time, driving_input_ou['inhibitory']['FS'], color=syn_colors["I"], label='inh_input', alpha=0.3)
        ax.axvspan(ss_start, ss_end, color='0.85', zorder=0, label='analysis window [2,6]')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.legend(fontsize=8, ncol=2)
        plt.suptitle("Mean field simulation (OU input) — FS and RS pop")
        plt.tight_layout()
        plt.savefig(os.path.join(sim_path, f"test_{plot_name_base}.png"))
        plt.close(fig) # Close to save memory

        # Average over the analysis window ONLY (first second discarded)
        mean_rate_FS = np.mean(out_ou['FS'][win])
        mean_rate_RS = np.mean(out_ou['RS'][win])

        if saving_hdf5:
            h5_filename = f"sim_data_{plot_name_base}.h5"
            with h5py.File(os.path.join(sim_path, h5_filename), "w") as f:
                
                # --- external input ---   
                ei = f.create_group("external_input")
                ei.create_dataset("exc_input_FS_time_s", data = exc_interval_FS[0:2])       
                ei.create_dataset("exc_input_FS_freq_Hz", data = exc_val)
                ei.create_dataset("exc_input_RS_time_s", data = exc_interval_RS[0:2])       
                ei.create_dataset("exc_input_RS_freq_Hz", data = exc_val)
                ei.create_dataset("inh_input_FS_time_s", data = inh_interval_FS[0:2])       
                ei.create_dataset("inh_input_FS_freq_Hz", data = inh_val)
                ei.create_dataset("inh_input_RS_time_s", data = inh_interval_RS[0:2])       
                ei.create_dataset("inh_input_RS_freq_Hz", data = inh_val)
                
                # --- stats ---
                stats = f.create_group("stats")
                stats.create_dataset("FS_avg_freq", data=mean_rate_FS)
                stats.create_dataset("RS_avg_freq", data=mean_rate_RS)

                # --- reproducibility metadata ---
                sp = f.create_group("sim_params")
                sp.create_dataset("model", data="MF")
                sp.create_dataset("dt_s", data=dt)
                sp.create_dataset("T_s", data=T)
                sp.create_dataset("input_window_s", data=[input_start, input_end])
                sp.create_dataset("analysis_window_s", data=[ss_start, ss_end])
                sp.create_dataset("tau_ou_s", data=tau_ou)
                sp.create_dataset("sigma_ou", data=sigma_ou)
                sp.create_dataset("ou_seed", data=0)
                sp.create_dataset("FS_TF_idx", data=FS_idx)
                sp.create_dataset("RS_TF_idx", data=RS_idx)
