### This code is used to simulate the data used to fit the transfer function ###
import sys
import os
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

# Adding the project folder to sys.path
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..', '..'))
sys.path.append(parent_dir)
# We do this so that we can directly import files in the utils folder


from ntmf.neurons import *
from utils.Plots_helper import *

b2.seed(1234) # fixing seed for reproducibility 

###########################################
### Defining duration of the simulation ###
###########################################

sim_duration = 3000 * b2.ms
dt = 0.1 * b2.ms  # time resolution
times = b2.arange(0, sim_duration, dt)

data_folder = '../../neuron_models/EGLIF'
idx = 0 

# set to True if you want to save the output
save_json = True
save_figures = False

##################################################
### Defining "manual" current injection protocol ###
##################################################
# Manual TimedArray implementation (replaces neurodynex3)
stim_t_start = 0 * b2.ms
stim_t_end = sim_duration
stim_amplitude = 0 * b2.nA

tmp = np.zeros(len(times)) * b2.amp
start_idx = int(stim_t_start / dt)
end_idx = int(stim_t_end / dt)
tmp[start_idx:end_idx] = stim_amplitude

current_timed_array = b2.TimedArray(tmp, dt=dt)

###################################
### Setting up the neuron model ###
###################################
neuron_model = 'GoC' 
simulation_folder = 'simulations/GoC'
N_cell_GoC = 10 # Population size

#####################################
### Defining external drive/input ###
#####################################
# input ranges
rate_inh_external_input_range   = np.linspace(3, 180, 20)[::2] 
rate_exc_external_input_range_m = np.linspace(4, 80, 20)[::2]
rate_exc_external_input_range   = np.linspace(0, 20, 20)[::2]


### information to save ###
in_inh = []
in_exc_m = []
in_exc = []

# Storing both average and std
Out_Freq_avg = np.zeros((len(rate_inh_external_input_range), 
                         len(rate_exc_external_input_range_m),
                         len(rate_exc_external_input_range)))

Out_Freq_std = np.zeros((len(rate_inh_external_input_range), 
                         len(rate_exc_external_input_range_m),
                         len(rate_exc_external_input_range)))

# Start the nested loops
for i, ext_input_i in enumerate(rate_inh_external_input_range):
    ext_input_inh = ext_input_i * b2.Hz
    for j, ext_input_e_m in enumerate(rate_exc_external_input_range_m):
        ext_input_exc_m = ext_input_e_m * b2.Hz
        for k, ext_input_e in enumerate(rate_exc_external_input_range):
            ext_input_exc = ext_input_e * b2.Hz   
        
            in_inh.append(ext_input_i)
            in_exc_m.append(ext_input_e_m)
            in_exc.append(ext_input_e)
            
            b2.start_scope()
            
            # Setup neuron population
            G_neuron, params = setting_EGLIF_simulation_Brian_Vmin_membrane_fluc_GoC(
                idx = idx,
                N_cell = N_cell_GoC,
                neuron_model = neuron_model,
                json_file_name = os.path.join(data_folder, neuron_model + '.json'),
                curr_inj = current_timed_array
            )
            G_neuron.run_on_event('vmin_event', 'V = Vmin')
            
            print(f"Simulating {i * len(rate_exc_external_input_range_m) * len(rate_exc_external_input_range) + j * len(rate_exc_external_input_range) + k + 1}/{len(rate_inh_external_input_range) * len(rate_exc_external_input_range_m) * len(rate_exc_external_input_range)} | " f"Inh: {ext_input_i} Hz | " f"Exc mossy: {ext_input_e_m} Hz | " f"Exc: {ext_input_e} Hz")
        
            # Defining TimedArrays for inputs
            rate_array_inh = b2.zeros(len(times)) * b2.Hz
            rate_array_exc_m = b2.zeros(len(times)) * b2.Hz
            rate_array_exc = b2.zeros(len(times)) * b2.Hz

            p_stim_start, p_stim_end = 1 * b2.second, 2 * b2.second
            
            rate_array_inh[(times >= p_stim_start) & (times < p_stim_end)] = ext_input_inh 
            rate_array_exc_m[(times >= p_stim_start) & (times < p_stim_end)] = ext_input_exc_m     
            rate_array_exc[(times >= p_stim_start) & (times < p_stim_end)] = ext_input_exc     

            rate_timed_array_inh = b2.TimedArray(rate_array_inh, dt=dt)
            rate_timed_array_exc_m = b2.TimedArray(rate_array_exc_m, dt=dt)
            rate_timed_array_exc = b2.TimedArray(rate_array_exc, dt=dt)
            
            # Synaptic parameters from params
            Q_i, Q_e_m, Q_e = params[0][idx]['model']['Q_i'], params[0][idx]['model']['Q_e_m'], params[0][idx]['model']['Q_e']
            
            T_i, T_e_m, T_e = params[0][idx]['model']['T_i'], params[0][idx]['model']['T_e_m'], params[0][idx]['model']['T_e']
            N_ext_inh, N_ext_exc_m = params[0][idx]['model']['K_i'], params[0][idx]['model']['K_e_m']
            N_ext_exc =  params[0][idx]['model']['K_e']
            # Input Groups
            P_ed_inh = b2.PoissonGroup(N_ext_inh, rates='rate_timed_array_inh(t)')
            P_ed_exc_m = b2.PoissonGroup(N_ext_exc_m, rates='rate_timed_array_exc_m(t)')
            P_ed_exc = b2.PoissonGroup(N_ext_exc, rates='rate_timed_array_exc(t)')
            
           
            # Synapses
            w_i, w_e_m, w_e = (Q_i*np.e/T_i)*b2.nS/b2.ms, (Q_e_m*np.e/T_e_m)*b2.nS/b2.ms, (Q_e*np.e/T_e)*b2.nS/b2.ms
            
            S_i = b2.Synapses(P_ed_inh, G_neuron, on_pre='xI += w_i')
            S_i.connect() 
            S_i.delay = params[0][idx]['model']['delay_i'] * b2.ms
            
            S_e_m = b2.Synapses(P_ed_exc_m, G_neuron, on_pre='xE_m += w_e_m')
            S_e_m.connect()
            S_e_m.delay = params[0][idx]['model']['delay_e_m'] * b2.ms

            S_e = b2.Synapses(P_ed_exc, G_neuron, on_pre='xE += w_e')
            S_e.connect()
            S_e.delay = params[0][idx]['model']['delay_e'] * b2.ms
            
            
            # Monitors
            mon_neuron = b2.StateMonitor(G_neuron, ['V'], record=0)
            mon_spike_neuron = b2.SpikeMonitor(G_neuron)
            mon_Poisson_input_inhibition = b2.SpikeMonitor(P_ed_inh)
            
            # Run
            b2.run(sim_duration, namespace={
                'external_current': current_timed_array,
                'rate_timed_array_inh': rate_timed_array_inh,
                'rate_timed_array_exc_m': rate_timed_array_exc_m,
                'rate_timed_array_exc': rate_timed_array_exc,
                'w_i': w_i, 'w_e_m': w_e_m, 'w_e': w_e
            })
    
            ###########################################
            ### Population Statistics (Avg & STD) ###
            ###########################################
            window_start, window_end = 1.0, 2.0
            duration = window_end - window_start
            
            individual_rates = []
            for neuron_idx in range(N_cell_GoC):
                spikes = mon_spike_neuron.t[mon_spike_neuron.i == neuron_idx] / b2.second
                n_count = np.sum((spikes >= window_start) & (spikes < window_end))
                individual_rates.append(n_count / duration)
            
            Out_Freq_avg[i,j,k] = np.mean(individual_rates)
            Out_Freq_std[i,j,k] = np.std(individual_rates)

##########################
### Saving the results ###
##########################
if save_json:
    os.makedirs(simulation_folder, exist_ok=True)
    print('... Saving .dat files...')
    # Flattening everything for the column stack
    data = np.column_stack((
        in_inh, 
        in_exc_m, 
        in_exc,
        Out_Freq_avg.flatten(), 
        Out_Freq_std.flatten()
    ))
    header = 'input_inh input_exc_m input_exc avg_f_out std_f_out'   
    np.savetxt(os.path.join(simulation_folder, 'testing_GoC_TF_data.dat'), 
               data, fmt='%.2f', header=header, comments='') 
    print('Done!')
