import matplotlib.pyplot as plt
import h5py
import sys
import os

# Adding the project folder to sys.path
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..', '..'))
sys.path.append(parent_dir)
# We do this so that we can directly import files in the utils folder


from utils.Brian_function_helper import *
from utils.Plots_helper import *
from TF_helper import *

saving_hdf5 = True

T = 5 
dt = 1e-3
time = np.arange(0, T, dt)
p_start, p_end = 0, 5

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
network_file_name = 'network_config_file_val_heatmap.json'
network_config = get_network_config(json_file_name=network_file_name)

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

with open('FS_MF_params.json', 'r') as f:
    data_FS = json.load(f)

with open('RS_MF_params.json', 'r') as f:
    data_RS = json.load(f) 

# --- OU Process Generation ---
tau_ou, sigma_ou = 0.05, 2.0
       
        print(f"Running simulation: Exc={exc_val}Hz, Inh={inh_val}Hz")
       
        driving_input_ou = {
        'excitatory': {
            'FS': generate_ou_process(time, dt, exc_val, tau_ou, sigma_ou),
            'RS': generate_ou_process(time, dt, exc_val, tau_ou, sigma_ou)
            },
        'inhibitory': {
            'FS': generate_ou_process(time, dt, inh_val, tau_ou, sigma_ou),
            'RS': generate_ou_process(time, dt, inh_val, tau_ou, sigma_ou)
            }
        }

        rates_ou, adapt_ou = simulate_MF_FS_RS_adapt_solve_ivp(
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
        sim_path = os.path.join('simulations/val_MF_[0_30]')
        if not os.path.exists(sim_path): os.makedirs(sim_path)

        fig, axs = plt.subplots(2,1, figsize=(8,5), sharex=True)
        for pop, color in zip(['FS','RS'], [color_palette["FS"],color_palette["RS"]]):
            axs[0].plot(time, rates_ou[pop], label=f'{pop}, [solve_ivp]', color=color)
            axs[1].plot(time, adapt_ou[pop], label=pop, color=color)
        axs[0].plot(time, driving_input_ou['excitatory']['FS'], color = syn_colors["E"], label='exc_input', alpha = 0.3)
        axs[0].plot(time, driving_input_ou['inhibitory']['FS'], color = syn_colors["I"], label='inh_input', alpha = 0.3)
        axs[0].legend()
        axs[1].set_xlabel("Time (s)")
        axs[0].set_ylabel("Firing rate (Hz)")
        axs[1].set_ylabel("Adaptation (A)")
        plt.legend()
        plt.suptitle("Mean field simulation with adapt (OU input) \n FS and RS pop")
        plt.savefig(os.path.join(sim_path, f"test_{plot_name_base}.png"))
        plt.close(fig) # Close to save memory

        mean_rate_FS, mean_rate_RS = np.mean(rates_ou['FS']), np.mean(rates_ou['RS'])

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
