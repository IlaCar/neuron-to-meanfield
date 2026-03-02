import json
import numpy as np
import os
import matplotlib.pyplot as plt
import brian2 as b2

implemented_neuron_models = ['FS', 'RS', 'RS_no_adapt']

# Definying the AdEx model #
AdEx_eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Itot = (GsynI+GsynE)*v : ampere 
Is = current(t) : ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
'''

# Definying conductance-based synaptic model #
syn_eqs = '''
dgE/dt = -gE/tau_syn_e : siemens
dgI/dt = -gI/tau_syn_i : siemens

IE = -gE*(V_hold - Ee) : ampere
II = -gI*(V_hold - Ei) : ampere
Itot = IE + II : ampere

tau_syn_e : second
tau_syn_i : second
Ee : volt
Ei : volt
V_hold : volt
'''

# -------------------- #
def setting_simulation_Brian(idx = None, N_cell = None, neuron_model = None, json_file_name = None, curr_inj = None, sim_info = False):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Plese, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    
    if sim_info == True:
        print(f'Imported data: {json_file_name}')
    
    if neuron_model == 'FS' or neuron_model == 'RS' or neuron_model == 'RS_no_adapt':
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']

        G = b2.NeuronGroup(N_cell, AdEx_eqs, threshold = f'v > {V_th_value} * mV',
                                            reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                            refractory = f'{t_ref_value} * ms',
                                            method = 'heun',
                                            name = neuron_model)
        #init variables:
        G.v = data[0][idx]['init']['v']*b2.mV
        G.w = data[0][idx]['init']['w']*b2.nA
        G.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G.GsynE = data[0][idx]['init']['g_E']*b2.nS

        #parameter values:
        G.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G.gl = data[0][idx]['model']['g_L'] * b2.uS
        G.El = data[0][idx]['model']['E_L'] * b2.mV
        G.a = data[0][idx]['model']['a'] * b2.nS
        G.tau_w = data[0][idx]['model']['tau_w'] * b2.ms        
        G.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G.Dt = data[0][idx]['model']['Delta_T'] * b2.mV

        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G.Tsyn = data[0][idx]['model']['tau_syn'] * b2.ms

        return G

# -------------------- #
def voltage_clamp_synapse(V_hold = None, idx = None, json_file_name = None, dt = None, sim_info = False):

    if idx is None:
        idx = 0
        
    if json_file_name is None:
        raise ValueError("Plese, specify the json_file_name containing the synaptic model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    
    if sim_info is True:
        print(f'Imported data: {json_file_name}')

    if dt is None:
        dt = 0.1
    
    b2.start_scope()
    b2.defaultclock.dt = dt * b2.ms

    G = b2.NeuronGroup(1, syn_eqs, method='exact')

    # Parameters
    G.tau_syn_e = data[0][0]['model']['tau_e'] * b2.ms
    G.tau_syn_i = data[0][0]['model']['tau_i'] * b2.ms
    G.Ee = data[0][0]['model']['E_e'] * b2.mV
    G.Ei = data[0][0]['model']['E_i'] * b2.mV

    sim_time = data[0][0]['simulation']['sim_duration'] * b2.ms
    t_event = data[0][0]['simulation']['t_pulse'] * b2.ms
    
    if V_hold is not None:
        V_hold = V_hold
    else:
        V_hold = -60 #mV
        
    G.V_hold = V_hold * b2.mV

    # Initial conductances
    G.gE = data[0][0]['init']['g_E'] * b2.nS
    G.gI = data[0][0]['init']['g_I'] * b2.nS

    # Inject synaptic event
    @b2.network_operation(dt=dt*b2.ms)
    def inject_event():
        if abs(b2.defaultclock.t - t_event) < 0.5*dt*b2.ms:
            G.gE += data[0][0]['model']['Q_e'] * b2.nS
            G.gI += data[0][0]['model']['Q_i'] * b2.nS

    M = b2.StateMonitor(G, ['gE','gI','IE','II','Itot'], record=True)

    net = b2.Network(G, inject_event, M)
    net.run(sim_time)
    
    return M

# -------------------- #
def network_creation(conn_prob = None, 
                     pop_1 = None, pop_2 = None,
                     Qe = None, Qi = None,
                     seed = None):

    if seed is not None:
        b2.seed(seed)  # to control the connectivity

    S_11 = b2.Synapses(pop_1, pop_1, on_pre='GsynI_post+=Qi', name = 'S_11')
    S_11.connect('i!=j',p=conn_prob)
    
    S_12 = b2.Synapses(pop_1, pop_2, on_pre='GsynI_post+=Qi', name = 'S_12')
    S_12.connect(p=conn_prob)
    
  
    S_21 = b2.Synapses(pop_2, pop_1, on_pre='GsynE_post+=Qe', name = 'S_21') 
    S_21.connect(p=conn_prob)
    
    S_22 = b2.Synapses(pop_2, pop_2, on_pre='GsynE_post+=Qe', name = 'S_22') 
    S_22.connect('i!=j', p=conn_prob)
       


    return S_11, S_12, S_21, S_22

# -------------------- #
def extracting_pop_freq_and_std(sim_duration = None, 
                              p_start = None,
                              p_end = None,
                              pop1 = None, 
                              pop2 = None, 
                              N_pop1 = None,
                              N_pop2 = None,
                              bin_size = None):

    # Parameters
    if bin_size == None:
        bin_size = 0.1 # seconds
    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix_FS = np.zeros((N_pop1, len(time_bins)))
    spike_matrix_RS = np.zeros((N_pop2, len(time_bins)))
    
    # Fill the spike matrix
    for i, t in zip(pop1.i, pop1.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_FS[i, bin_idx] += 1
    
    for i, t in zip(pop2.i, pop2.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_RS[i, bin_idx] += 1
    
    # Convert to rate (Hz)
    spike_matrix_FS /= bin_size
    spike_matrix_RS /= bin_size    
    
    # Compute mean and std
    mean_rate_FS = np.mean(spike_matrix_FS, axis=0)
    std_rate_FS = np.std(spike_matrix_FS, axis=0)
    mean_rate_RS = np.mean(spike_matrix_RS, axis=0)
    std_rate_RS = np.std(spike_matrix_RS, axis=0)

    # Defining the stimulation interval   
    if p_end == sim_duration - p_start:
        left_bound = int((p_start / bin_size).item())
        right_bound = - left_bound + 1
        
    # Computing the average only over the stimulation
    mean_rate_FS_stim = np.mean(mean_rate_FS[left_bound:right_bound])
    mean_rate_RS_stim = np.mean(mean_rate_RS[left_bound:right_bound])

    mean_rates = [mean_rate_FS_stim, mean_rate_RS_stim]

    # Computing the standard deviation only over the stimulation
    std_rate_FS_stim = np.std(std_rate_FS[left_bound:right_bound])
    std_rate_RS_stim = np.std(std_rate_RS[left_bound:right_bound])

    std_rates = [std_rate_FS_stim, std_rate_RS_stim]

    if len(time_bins) != len(mean_rate_FS):
        import pdb
        pdb.set_trace()
    
    return mean_rates, std_rates

# -------------------- #
def extracting_single_pop_freq_and_std(sim_duration = None, 
                              p_start = None,
                              p_end = None,
                              pop = None, 
                              N_pop = None,
                              bin_size = None,
                              delay = None):

    # Parameters
    if bin_size == None:
        bin_size = 0.1  # seconds
    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix = np.zeros((N_pop, len(time_bins)))

    # Fill the spike matrix
    for i, t in zip(pop.i, pop.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix[i, bin_idx] += 1

    # Convert to rate (Hz)
    spike_matrix /= bin_size
  
    # Compute mean and std
    mean_rate = np.mean(spike_matrix, axis=0)
    std_rate = np.std(spike_matrix, axis=0)

    # Defining the stimulation interval
    if delay is not None:
        left_bound = int(((p_start + delay) / bin_size).item())
    else:
        left_bound = int((p_start / bin_size).item())      
    right_bound = left_bound + int(((p_end - p_start) / bin_size).item()) + 1

    # Computing the average only over the stimulation
    mean_rate_stim = np.mean(mean_rate[left_bound:right_bound])

    # Computing the standard deviation only over the stimulation
    std_rate_stim = np.std(std_rate[left_bound:right_bound])

    if len(time_bins) != len(mean_rate):
        import pdb
        pdb.set_trace()
    
    return mean_rate_stim, std_rate_stim

# -------------------- #
def network_creation(conn_prob = None, 
                     pop_1 = None, pop_2 = None,
                     Qe_FS = None, Qi_FS = None,
                     Qe_RS = None, Qi_RS = None,
                     seed = None):

    if seed is not None:
        b2.seed(seed)  # to control the connectivity

    S_11 = b2.Synapses(pop_1, pop_1, on_pre='GsynI_post+=Qi_FS', name = 'S_11')
    S_11.connect('i!=j',p=conn_prob)
    
    S_12 = b2.Synapses(pop_1, pop_2, on_pre='GsynI_post+=Qi_RS', name = 'S_12')
    S_12.connect(p=conn_prob)
    
  
    S_21 = b2.Synapses(pop_2, pop_1, on_pre='GsynE_post+=Qe_FS', name = 'S_21') 
    S_21.connect(p=conn_prob)
    
    S_22 = b2.Synapses(pop_2, pop_2, on_pre='GsynE_post+=Qe_RS', name = 'S_22') 
    S_22.connect('i!=j', p=conn_prob)
    

    return S_11, S_12, S_21, S_22
