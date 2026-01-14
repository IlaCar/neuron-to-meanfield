import json
import brian2 as b2
import h5py

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






