# This file is a support to the notebook Figure1_B.ipynb
# Here we run a complete set of current injection and we save the data in a json file

import sys
import os
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..'))
sys.path.append(parent_dir)


from utils.Brian_function_helper import *

### specify the neuron model ###
neuron_model = 'RS' 

### specify range of amplitudes ###
amplitudes = np.arange(0.1, 1.05, 0.05)


# Defining the delay and the duration of the stimulus (step current injection)
s_delay = 100        # ms
s_duration = 1000     # ms
dt = 0.1                # ms

stim_delay = s_delay/dt 
stim_duration = s_duration/dt

# Explicitly defining the time
Time = 1200          # ms
time = np.arange(0,Time+dt,dt)

data_folder = '../neuron_models/AdEx'
idx = 0 # in this case only one model per cell type is provided

for amp in amplitudes:
    # Defining the current injection structure
    stim_t_start = int(stim_delay * dt)
    stim_t_end = int(stim_duration * dt + stim_t_start)
    stim_unit = 1. * b2.ms
    stim_amplitude = amp * b2.nA
    stim_time = Time * b2.ms
    
    # Setting up the step current injection protocol
    tmp_size = 2 + stim_t_end  
    tmp = np.zeros((tmp_size, 1)) * b2.amp
    tmp[stim_t_start: stim_t_end + 1, 0] = stim_amplitude
    curr_array = b2.TimedArray(tmp, dt=1. * stim_unit).values.flatten()  # to solve the issue with dimention of the current vector
    current_timed_array = b2.TimedArray(curr_array * b2.amp, dt=stim_unit)

    G_inh = setting_simulation_Brian(idx = idx,
                                 neuron_model = neuron_model,
                                 json_file_name = os.path.join(data_folder, neuron_model + '.json'),
                                 curr_inj = current_timed_array)

    mon_neuron = b2.StateMonitor(G_inh, ['v', 'Is'], record=True)
    #mon_neuron_rate = b2.PopulationRateMonitor(G_inh) # not ideal, only 1 neuron in the population
    mon_spike_neuron = b2.SpikeMonitor(G_inh)

    b2.run(stim_time, namespace={'current': current_timed_array})
    mask_neuron = (mon_spike_neuron.t >= stim_t_start* b2.ms) & (mon_spike_neuron.t < stim_t_end* b2.ms)
    n_spikes_neuron_interval_neuron = mask_neuron.sum() 
    
    new_entry_neuron   = {"amp (nA)": float(amp), "freq (Hz)": int(n_spikes_neuron_interval_neuron)}
    print(new_entry_neuron)
    filename_neuron   = f"extracted_data/{neuron_model}_f-i_data.json"

    # Load existing data if file exists
    if os.path.exists(filename_neuron):
        with open(filename_neuron, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    # Append new entry
    data.append(new_entry_neuron)
    
    # Save back to the file
    with open(filename_neuron, 'w') as f:
        json.dump(data, f, indent=4)

print('... simulations concluded ...')