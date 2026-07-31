import json
import numpy as np

def data_extract_features(trace_time, trace_voltage,  stim_delay, stim_duration, curr, dt):
    
    results = []
    time = trace_time

    freq = None
    inv_first_ISI = None
    inv_last_ISI = None
    time_to_first_spike = None
    time_to_second_spike = None
    time_to_third_spike = None
    time_to_last_spike = None      
    volt_stimend = None

    current = curr #pA
   
    ### checking spikes ###
    peak_index = []
    for i in range(1, len(trace_voltage)):
        if trace_voltage[i] > 10 and trace_voltage[i-1]< 10:
            peak_index.append(i)

    freq = len(peak_index)/(stim_duration/1e3)
    if len(peak_index) > 0:
        time_to_first_spike = time[peak_index[0]] - stim_delay #ms
    if len(peak_index) >= 2 :
        inv_first_ISI = 1e3/((time[peak_index[1]]-time[peak_index[0]])) #Hz
        time_to_second_spike = time[peak_index[1]] - stim_delay #ms
    if len(peak_index) >= 3 :    
        inv_second_ISI = 1e3/((time[peak_index[1]]-time[peak_index[0]])) #Hz
        time_to_third_spike = time[peak_index[2]] - stim_delay #ms 
    if len(peak_index) > 3 :   
        time_to_last_spike = time[peak_index[-1]] - stim_delay #ms 
        inv_last_ISI = 1e3/((time[peak_index[-1]]-time[peak_index[-2]])) #Hz

    # if last spike arrives before half of the stim_duration, we also compute the volt_stimend
    if time_to_last_spike < stim_duration / 2:
        index_end_stim = int((stim_delay+stim_duration)/dt)
        volt_stimend = np.mean(trace_voltage[index_end_stim - int(30/dt): index_end_stim])

    results.append((current, freq, inv_first_ISI, inv_last_ISI, time_to_first_spike, time_to_second_spike, time_to_third_spike, time_to_last_spike, volt_stimend))

    return results

#Saving a json file containing the extracted features
def append_to_json_file(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)