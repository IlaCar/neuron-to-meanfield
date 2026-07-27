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

saving_hdf5 = True

sim_duration = 7 * b2.second
dt = 0.1 * b2.ms  # time resolution
times = b2.arange(0, sim_duration, dt)

p_start, p_end = 1 * b2.second, 6 * b2.second   # input ON window
ss_start = 4 * b2.second                        # steady-state analysis starts 1 s after onset -> measure [2, 6] s

exc_interval_FS = (p_start/b2.second, p_end/b2.second, 'FS')
exc_interval_RS = (p_start/b2.second, p_end/b2.second, 'RS')
exc_intervals = [exc_interval_FS, exc_interval_RS] # used for plotting

inh_interval_FS = (p_start/b2.second, p_end/b2.second, 'FS')
inh_interval_RS = (p_start/b2.second, p_end/b2.second, 'RS')
inh_intervals = [inh_interval_FS, inh_interval_RS] # used for plotting

data_folder = '../../neuron_models/AdEx'
idx = 0 # in this case only one model per cell type is provided

# --- Network Setup (constant across the grid, so build it once) ---
config_file_name = '../../config/network_config_file_val_heatmap.json'
network_config = get_network_config(json_file_name=config_file_name)

# Define the ranges for the loops
exc_freq_range = np.arange(0, 31, 1) 
inh_freq_range = np.arange(0, 31, 1) 

# --- OU Process parameters ---
tau_ou, sigma_ou = 0.05, 2.0

for exc_val in exc_freq_range:
    for inh_val in inh_freq_range:
        # As usual, we need to reset the Brian scope for every new simulation iteration
        b2.start_scope()
        
        print(f"Running simulation: Exc={exc_val}Hz, Inh={inh_val}Hz")

        # Update the rates based on current loop values
        rate_exc_external_input_FS = exc_val * b2.Hz
        rate_exc_external_input_RS = exc_val * b2.Hz
        rate_inh_external_input_FS = inh_val * b2.Hz
        rate_inh_external_input_RS = inh_val * b2.Hz

        # --- Re-setup Current Injection ---
        stim_t_start = 0
        stim_t_end = int(sim_duration)
        stim_amplitude = 0 * b2.nA
        stim_unit = 1. * b2.ms
        tmp_size = 2 + stim_t_end  
        tmp = np.zeros((tmp_size, 1)) * b2.amp
        tmp[stim_t_start: stim_t_end + 1, 0] = stim_amplitude
        curr_array = b2.TimedArray(tmp, dt=1. * stim_unit).values.flatten()
        current_timed_array = b2.TimedArray(curr_array * b2.amp, dt=stim_unit)

        # --- Neuron groups ---
        G_FS = setting_simulation_Brian(idx=idx, N_cell=network_config['network_composition']['FS_neuron'],
                                        neuron_model='FS', json_file_name=os.path.join(data_folder, 'FS.json'),
                                        curr_inj=current_timed_array)
        
        G_RS = setting_simulation_Brian(idx=idx, N_cell=network_config['network_composition']['RS_neuron'],
                                        neuron_model='RS', json_file_name=os.path.join(data_folder, 'RS.json'),
                                        curr_inj=current_timed_array)

        conn_prob = network_config['network_composition']['conn_prob']
        Qe_FS, Qi_FS = get_syn_info(json_file_name=os.path.join(data_folder, 'FS.json'))
        Qe_RS, Qi_RS = get_syn_info(json_file_name=os.path.join(data_folder, 'RS.json'))

        S_11, S_12, S_21, S_22 = network_creation(conn_prob=conn_prob, pop_1=G_FS, pop_2=G_RS, 
                                                 Qe_FS=Qe_FS, Qi_FS=Qi_FS, Qe_RS=Qe_RS, Qi_RS=Qi_RS, seed=12345)

        # --- OU Process Generation ---
        def get_ou_timed_array(base_rate):
            ou = generate_ou_process(times/b2.second, dt/b2.second, base_rate/b2.Hz, tau_ou, sigma_ou)
            arr = ou * b2.Hz
            arr[:int(p_start/dt)] = 0 * b2.Hz # Windowing logic
            arr[int(p_end/dt):] = 0 * b2.Hz
            return b2.TimedArray(arr, dt=dt)

        rate_timed_array_exc_FS = get_ou_timed_array(rate_exc_external_input_FS)
        rate_timed_array_exc_RS = get_ou_timed_array(rate_exc_external_input_RS)
        rate_timed_array_inh_FS = get_ou_timed_array(rate_inh_external_input_FS)
        rate_timed_array_inh_RS = get_ou_timed_array(rate_inh_external_input_RS)

        # --- Poisson Groups & Synapses ---
        N_ext_exc = network_config['external_input']['N_external_exc']
        N_ext_inh = network_config['external_input']['N_external_inh']
        ext_conn_prob = network_config['external_input']['conn_prob']

        P_ed_exc_FS = b2.PoissonGroup(N_ext_exc, rates='rate_timed_array_exc_FS(t)')
        P_ed_exc_RS = b2.PoissonGroup(N_ext_exc, rates='rate_timed_array_exc_RS(t)')
        P_ed_inh_FS = b2.PoissonGroup(N_ext_inh, rates='rate_timed_array_inh_FS(t)')
        P_ed_inh_RS = b2.PoissonGroup(N_ext_inh, rates='rate_timed_array_inh_RS(t)')

        # Synapse connection
        S_ed_exc_FS = b2.Synapses(P_ed_exc_FS, G_FS, on_pre='GsynE_post+=Qe_FS')
        S_ed_exc_FS.connect(p=ext_conn_prob)
        S_ed_exc_RS = b2.Synapses(P_ed_exc_RS, G_RS, on_pre='GsynE_post+=Qe_RS')
        S_ed_exc_RS.connect(p=ext_conn_prob)
        
        S_ed_inh_FS = b2.Synapses(P_ed_inh_FS, G_FS, on_pre='GsynI_post+=Qi_FS')
        S_ed_inh_FS.connect(p=ext_conn_prob)
        S_ed_inh_RS = b2.Synapses(P_ed_inh_RS, G_RS, on_pre='GsynI_post+=Qi_RS')
        S_ed_inh_RS.connect(p=ext_conn_prob)

        mon_spike_FS = b2.SpikeMonitor(G_FS)
        mon_spike_RS = b2.SpikeMonitor(G_RS)

        b2.run(sim_duration, namespace={'current': current_timed_array,
                                'rate_timed_array_exc_FS': rate_timed_array_exc_FS,
                                'rate_timed_array_exc_RS': rate_timed_array_exc_RS,
                                'rate_timed_array_inh_FS': rate_timed_array_inh_FS,
                                'rate_timed_array_inh_RS': rate_timed_array_inh_RS,
                                'Qe_FS': Qe_FS,
                                'Qi_FS': Qi_FS,
                                'Qe_RS': Qe_RS,
                                'Qi_RS': Qi_RS})

        # --- SAVING & PLOTTING ---       
        exc_intervals = [(p_start/b2.second, p_end/b2.second, 'FS'), (p_start/b2.second, p_end/b2.second, 'RS')]
        inh_intervals = [(p_start/b2.second, p_end/b2.second, 'FS'), (p_start/b2.second, p_end/b2.second, 'RS')]

        # Update naming convention for plots
        plot_name_base = f"exc_{exc_val}_inh_{inh_val}"
        sim_path = os.path.join('simulations/val_NN_[0_30_1]')
        if not os.path.exists(sim_path): os.makedirs(sim_path)

        for b_size, label in [(50*b2.ms, "50ms"), (1000*b2.ms, "1sec")]:
            fig = plotting_pop_freq_and_std(sim_duration=sim_duration, pop1=mon_spike_FS, pop2=mon_spike_RS, 
                                            N_pop1=network_config['network_composition']['FS_neuron'],
                                            N_pop2=network_config['network_composition']['RS_neuron'],
                                            bin_size=b_size, exc_intervals=exc_intervals, 
                                            inh_intervals=inh_intervals, input_boxes=False)
            plt.savefig(os.path.join(sim_path, f"test_{label}_{plot_name_base}.png"))
            plt.close(fig) # Close to save memory

        mean_rates, std_rates = extracting_pop_freq_and_std(sim_duration=sim_duration, p_start=ss_start, p_end=p_end,
                                                           pop1=mon_spike_FS, pop2=mon_spike_RS, 
                                                           N_pop1=network_config['network_composition']['FS_neuron'],
                                                           N_pop2=network_config['network_composition']['RS_neuron'])

        if saving_hdf5:
            h5_filename = f"sim_data_{plot_name_base}.h5"
            with h5py.File(os.path.join(sim_path, h5_filename), "w") as f:
                
                # --- external input ---   
                ei = f.create_group("external_input")
                ei.create_dataset("exc_input_FS_time_s", data = exc_interval_FS[0:2])       
                ei.create_dataset("exc_input_FS_freq_Hz", data = rate_exc_external_input_FS / b2.Hz)
                ei.create_dataset("exc_input_RS_time_s", data = exc_interval_RS[0:2])       
                ei.create_dataset("exc_input_RS_freq_Hz", data = rate_exc_external_input_RS / b2.Hz)
                ei.create_dataset("inh_input_FS_time_s", data = inh_interval_FS[0:2])       
                ei.create_dataset("inh_input_FS_freq_Hz", data = rate_inh_external_input_FS / b2.Hz)
                ei.create_dataset("inh_input_RS_time_s", data = inh_interval_RS[0:2])       
                ei.create_dataset("inh_input_RS_freq_Hz", data = rate_inh_external_input_RS / b2.Hz)
                
                # --- stats ---
                stats = f.create_group("stats")
                stats.create_dataset("FS_avg_freq", data=mean_rates[0])
                stats.create_dataset("FS_std_freq", data=std_rates[0])
                stats.create_dataset("RS_avg_freq", data=mean_rates[1])
                stats.create_dataset("RS_std_freq", data=std_rates[1])

                # --- reproducibility metadata ---
                sp = f.create_group("sim_params")
                sp.create_dataset("model", data="NN")
                sp.create_dataset("dt_s", data=float(dt/b2.second))
                sp.create_dataset("T_s", data=float(sim_duration/b2.second))
                sp.create_dataset("input_window_s", data=[float(p_start/b2.second), float(p_end/b2.second)])
                sp.create_dataset("analysis_window_s", data=[float(ss_start/b2.second), float(p_end/b2.second)])
                sp.create_dataset("tau_ou_s", data=tau_ou)
                sp.create_dataset("sigma_ou", data=sigma_ou)
                sp.create_dataset("ou_seed", data=0)
                sp.create_dataset("network_seed", data=12345)
