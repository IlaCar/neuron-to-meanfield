import sys
import os

# Adding the project folder to sys.path
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..', '..'))
sys.path.append(parent_dir)
# We do this so that we can directly import files in the utils folder

from ntmf.config import get_input_config, get_syn_info
from ntmf.neurons import setting_simulation_Brian
from ntmf.network import extracting_single_pop_freq_and_std
from utils.Plots_helper import *

simualtion_folder = os.path.join('TF_RS_delay_v1')
isExist = os.path.exists(simualtion_folder)
if not isExist:
   os.makedirs(simualtion_folder)

# making sure that brian resets everything
b2.start_scope()
# defining a seed
b2.seed(12345)

# Defining duration of the stimulation
sim_duration = 7 * b2.second
dt = 0.1 * b2.ms  # time resolution
times = b2.arange(0, sim_duration, dt)

# Setting up the neuron model folder to use
data_folder = '../../neuron_models/AdEx'
idx = 0 # in this case only one model per cell type is provided

# Defining neuron model
neuron_model = 'RS'

# Setting up the artificial 'fake' current injection
stim_t_start = 0
stim_t_end = int(sim_duration)
stim_amplitude = 0 * b2.nA
stim_unit = 1. * b2.ms
stim_time = sim_duration
# Setting up the step current injection protocol
tmp_size = 2 + stim_t_end  
tmp = np.zeros((tmp_size, 1)) * b2.amp
tmp[stim_t_start: stim_t_end + 1, 0] = stim_amplitude
curr_array = b2.TimedArray(tmp, dt=1. * stim_unit).values.flatten()  # to solve the issue with dimention of the current vector
current_timed_array = b2.TimedArray(curr_array * b2.amp, dt=stim_unit)


# Defining external drive/input ranges
input_config = get_input_config(json_file_name = '../../config/input_config_TF.json') 
# number of external neurons - number of incoming spikes
N_external_exc = input_config['connections']['N_external_exc']
N_external_inh = input_config['connections']['N_external_inh']
conn_prob =  input_config['connections']['conn_prob']

rate_exc_background = input_config['rates']['background_freq'] * b2.Hz # firing rate of the external input

step_freq_inh = 0.5 #Hz
step_freq_exc = 0.5 #Hz

external_input_inh_range = np.arange(0, 30.5, step_freq_inh) #Hz
external_input_exc_range = np.arange(0, 30.5, step_freq_exc) #Hz

# Two seconds stimulation
p_start = 2 * b2.second
p_end = 5 * b2.second
delay = 1 * b2.second

input_interval = [(p_start / b2.second, p_end / b2.second)]

# Duration of the background input activity
p_start_0 = 0 * b2.second
p_end_0 = sim_duration

bin_size = 0.25 # seconds

# Quantal increment in synaptic conductances:
Qe, Qi = get_syn_info(json_file_name = os.path.join(data_folder, 'RS.json'))
### information to save ###
in_inh = []
in_exc = []

FREQ_avg = np.zeros((len(external_input_inh_range), len(external_input_exc_range)))
FREQ_std = np.zeros((len(external_input_inh_range), len(external_input_exc_range)))

# Nested for loops
n_sim = 1
num_sim = len(external_input_inh_range) * len(external_input_exc_range)
print(f'Number of simulations to run: {num_sim}')

for i, ext_input_inh in enumerate(external_input_inh_range):
    for j, ext_input_exc in enumerate(external_input_exc_range):
        in_inh.append(ext_input_inh)
        in_exc.append(ext_input_exc)
        b2.start_scope()
        # Setting up the neuron models and number of neurons per population
        N_cell_RS = 50
        G_RS = setting_simulation_Brian(idx = idx,
                                         N_cell = N_cell_RS,
                                         neuron_model = neuron_model,
                                         json_file_name = os.path.join(data_folder, neuron_model + '.json'),
                                         curr_inj = current_timed_array)
        
        # Poissoninan background input activity
        rate_array = b2.zeros(len(times)) * b2.Hz
        rate_array[(times >= p_start_0) & (times < p_end_0)] = rate_exc_background 
                # Create the TimedArray
        rate_timed_array_bg = b2.TimedArray(rate_array, dt=dt)
        
        rate_array_inh = b2.zeros(len(times)) * b2.Hz
        rate_array_exc = b2.zeros(len(times)) * b2.Hz
        
        rate_array_inh[(times >= p_start) & (times < p_end)] = ext_input_inh * b2.Hz
        rate_timed_array_inh = b2.TimedArray(rate_array_inh, dt=dt)
        rate_array_exc[(times >= p_start) & (times < p_end)] = ext_input_exc * b2.Hz     
        rate_timed_array_exc = b2.TimedArray(rate_array_exc, dt=dt)

        P_ed_bg = b2.PoissonGroup(N_external_exc, rates='rate_timed_array_bg(t)')
        P_ed_inh = b2.PoissonGroup(N_external_inh, rates='rate_timed_array_inh(t)')
        P_ed_exc = b2.PoissonGroup(N_external_exc, rates='rate_timed_array_exc(t)')

        # Background input (All the cell types receive the same)
        S_ed_background_bg = b2.Synapses(P_ed_bg, G_RS, on_pre='GsynE_post+=Qe')
        S_ed_background_bg.connect(p = conn_prob)
        
        # Connecting the external input to the neuron models
        S_ed_inh = b2.Synapses(P_ed_inh, G_RS, on_pre='GsynI_post+=Qi')
        S_ed_inh.connect(p = conn_prob)           
        S_ed_exc = b2.Synapses(P_ed_exc, G_RS, on_pre='GsynE_post+=Qe')
        S_ed_exc.connect(p = conn_prob)            

        # Defining the Monitors (variables to record)
        mon_spike_RS = b2.SpikeMonitor(G_RS)

        #ids_extra_check = len(external_input_inh_range)
        ids_extra_check = 10000 # to avoid extra recordings and plottings
        if n_sim % ids_extra_check == 0:
            ### adding extra monitors
            mon_RS = b2.StateMonitor(G_RS, ['v'], record=True)               

        ### running ###
        print(f'Running -- inh: {ext_input_inh} Hz and exc: {ext_input_exc} Hz -- background {rate_exc_background / b2.Hz} Hz')
        b2.run(sim_duration, namespace={'current': current_timed_array,
                                        'rate_timed_array_bg': rate_timed_array_bg,
                                        'rate_timed_array_inh': rate_timed_array_inh,
                                        'rate_timed_array_exc': rate_timed_array_exc,
                                        'Qe': Qe,
                                        'Qi': Qi})

        mean_rate, std_rate = extracting_single_pop_freq_and_std(sim_duration = sim_duration, 
                          p_start = p_start,
                          p_end = p_end,
                          pop = mon_spike_RS, 
                          N_pop = N_cell_RS,
                          bin_size =  bin_size, # bin_size in seconds
                          delay = delay # in seconds
                          )

        FREQ_avg[i,j] = mean_rate
        FREQ_std[i,j] = std_rate

        ### plotting ###
        if n_sim % ids_extra_check == 0:
            print(f'... running {n_sim}/{num_sim}...')

            fig = disconnected_network_raster_plot_TF(neuron_model = neuron_model,
                                                     pop = mon_spike_RS,
                                                     N_pop = N_cell_RS,
                                                     input_interval = input_interval,
                                                     x_lim = (0, sim_duration/b2.second))
                                      
            plt.savefig(os.path.join(simualtion_folder, f'inh_{ext_input_inh}_exc_{ext_input_exc}_raster.png'))

            
            fig = plotting_3_traces(neuron_model = neuron_model,
                                    pop = mon_RS, 
                                    input_interval = input_interval)             
            plt.savefig(os.path.join(simualtion_folder, f'inh_{ext_input_inh}_exc_{ext_input_exc}_sel_traces.png'))               
            
            fig = plotting_single_pop_freq_and_std(sim_duration = sim_duration, 
                          neuron_model = neuron_model, 
                          pop = mon_spike_RS, 
                          N_pop = N_cell_RS,
                          bin_size = bin_size, # bin_size in seconds
                          input_interval = input_interval, 
                          )
            plt.savefig(os.path.join(simualtion_folder, f'inh_{ext_input_inh}_exc_{ext_input_exc}.png'))

        n_sim += 1

print(f'Simulations completed!')

print(in_inh, in_exc, FREQ_avg.flatten(), FREQ_std.flatten())

print('... Saving .dat files...')
data = np.column_stack((in_inh, in_exc, FREQ_avg.flatten(), FREQ_std.flatten()))
header = 'input_inh input_exc avg_f_out std_f_out'
np.savetxt(os.path.join(simualtion_folder,'testing_TF_data_RS.dat'), data, fmt='%.2f', header=header, comments='')

print('Done!')
