import json
import brian2 as b2
import h5py
import os
import pandas as pd

# ---------------------------------------------------
def get_input_config(idx = None, json_file_name = None):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the input configuration parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)

        return data[0]

# -------------------- #
def get_syn_info(idx = None, json_file_name = None):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")   
    with open(json_file_name, 'r') as file:
        data = json.load(file)

    Qe = data[0][idx]['model']['Q_e'] * b2.nS
    Qi = data[0][idx]['model']['Q_i'] * b2.nS
    
    return Qe, Qi

# -------------------- #
def get_network_config(idx = None, json_file_name = None):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)

        return data[0]

# -------------------- #
def load_spike_data(fname):
    """
    Load data from simulations/*.h5 file.
    """
    data = {}

    with h5py.File(fname, "r") as f:

        # ---- network_composition ----
        nc = {}
        for key in f["network_composition"].keys():
            nc[key] = f["network_composition"][key][()]
        data["network_composition"] = nc

        # ---- external input ----
        ei = {}
        for key in f["external_input"].keys():
            ei[key] = f["external_input"][key][()]
        data["external_input"] = ei
        
        # ---- spikes ----
        spikes = {}
        data["sim_duration"] = f["spikes"]["sim_duration"][()]
        for pop in ["FS", "RS"]:
            group = f["spikes"][pop]
            spikes[pop] = {
                "i": group["i"][()],
                "t": group["t"][()],
            }
        data["spikes"] = spikes

    return data
    
# -------------------- #
def load_sim_data(folder_path, model_type="NN"):
    """
    Parses h5 files in a folder and returns a list of results.
    model_type: "NN" (collects avg and std) or "MF" (collects avg)
    """
    results = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5") and "exc_" in filename:
            # Parse exc and inh values from filename: sim_data_exc_10_inh_5.h5
            parts = filename.replace('.h5', '').split('_')
            exc = float(parts[3])
            inh = float(parts[5])
            
            with h5py.File(os.path.join(folder_path, filename), "r") as f:
                # Extract stats for both populations
                res = {
                    'exc': exc, 
                    'inh': inh, 
                    'fs_avg': f['stats']['FS_avg_freq'][()],
                    'rs_avg': f['stats']['RS_avg_freq'][()]
                }
                
                # Extract Standard Deviations only if it's the NN model
                if model_type == "NN":
                    res['fs_std'] = f['stats']['FS_std_freq'][()]
                    res['rs_std'] = f['stats']['RS_std_freq'][()]
                
                results.append(res)
                
    return pd.DataFrame(results)





