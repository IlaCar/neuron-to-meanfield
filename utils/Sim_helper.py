import json
import brian2 as b2
# ---------------------------------------------------
def get_input_config(idx = None, json_file_name = None):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the input configuration parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)

        return data[0]

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

def get_network_config(idx = None, json_file_name = None):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)

        return data[0]






