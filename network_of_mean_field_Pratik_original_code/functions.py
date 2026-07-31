# Definitions of the different function used:
import numpy as np
from scipy.special import erfc
import pandas as pd
import sys

def TF_general(cell_param,input_node_activity,adapt,record=False,I=0):
    # cell_param: cellular parameters
    # input_node_activity: a dict containing firing rates of input nodes to the given node
    # adapt: adaptation

    f=np.zeros(len(input_node_activity))

    for i in range(len(input_node_activity)):
        f[i]=input_node_activity[cell_param.order[i]]

    Q = np.asarray(cell_param.Q,dtype=float)
    T = np.asarray(cell_param.T,dtype=float)
    E = np.asarray(cell_param.E,dtype=float)
    Gl = cell_param.Gl
    El = cell_param.El
    Cm = cell_param.Cm
    P=cell_param.P

    muG0 = Q*T*f
    muG = np.sum(muG0) + Gl

    muV = (np.sum(muG0*E) + Gl*El - adapt + I)/muG 

    Tm = Cm/muG;
    
    U = (Q/muG)*(E-muV);

    sV = np.sqrt(np.sum((f*U**2*T**2)/(2*(T+Tm))))

    Tv_numerator = f * U**2 * T**2
    TV_denominator = (f * U**2 * T**2) / (T + Tm)
    Tv = np.sum(Tv_numerator) / np.sum(TV_denominator)

    TvN = Tv*Gl/Cm;
    
    muV0=-60e-3;
    DmuV0=10e-3;
    sV0=4e-3;
    DsV0=6e-3;
    TvN0=0.5;
    DTvN0=1.;

    P=cell_param.P
    
    vthre=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[5]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[6]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[7]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[8]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[9]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;

    frout = 1/(2.*Tv)*erfc((vthre-muV)/(np.sqrt(2)*sV) );
 
    
    if record: return frout,muV,sV
    else: return frout;


def adaptation_equation_general(cell_param,input_node_activity,ff,adapt):
    # cell_param: cellular parameters
    # input_node_activity: a dict containing firing rates of input nodes to the given node
    # ff : Firing rate of given node

    f=np.zeros(len(input_node_activity))

    for i in range(len(input_node_activity)):
        f[i]=input_node_activity[cell_param.order[i]]

    Q = np.asarray(cell_param.Q,dtype=float)
    T = np.asarray(cell_param.T,dtype=float)
    E = np.asarray(cell_param.E,dtype=float)
    Gl = cell_param.Gl
    El = cell_param.El
    Tw = cell_param.Tw
    a = cell_param.a
    b = cell_param.b

    muG0 = Q*T*f
    muG = np.sum(muG0) + Gl

    muV = (np.sum(muG0*E) + Gl*El - adapt)/muG 

    adapt_new = -adapt/Tw + b*ff + a*(muV-El)/Tw
    
    return adapt_new;

def OU(tfin,dt):
    
    np.random.seed(20)

    # Ornstein-Ulhenbeck process
    theta = 1/(5*1.e-3 )  # Mean reversion rate
    mu = 0    # Mean of the process
    sigma = 1.0   # Volatility or standard deviation
    T = tfin        # Total time period

    # Initialize the variables
    t = np.arange(0, T, dt)         # Time vector
    n = len(t)                      # Number of time steps
    x = np.zeros(n)                 # Array to store the process values
    x[0] = 0  # Initial value

    # Generate the process using the Euler-Maruyama method
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        x[i] = x[i-1] + dx
    return x

def double_gaussian(t, t0, T1, T2, amplitude):
    return amplitude*(np.exp(-(t-t0)**2/2./T1**2)*(t<t0)+ np.exp(-(t-t0)**2/2./T2**2)*(t>t0))

def read_nodes_csv(file_path):
    populations = []
    population_types = {}
    node_population = {}
    number_of_neurons = {}
    initial_firing_rates = {}
    e=""
    
    try:
        nodes_df = pd.read_csv(file_path)

        if nodes_df['Population Name'].duplicated().any():
            raise ValueError("Duplicate values found in 'Population Name' column. Please give unique name to every node")
        
        for _, row in nodes_df.iterrows():
            population_name = str(row['Population Name'])
            population_type = row['Population Type']
            node_type = row['Node Type']
            neurons = row['Neurons']
            firing_rate = row['Firing Rate']
            
            populations.append(population_name)
            population_types[population_name] = population_type
            node_population[population_name] = node_type
            number_of_neurons[population_name] = neurons
            initial_firing_rates[population_name] = firing_rate
        
        return populations, population_types, node_population, number_of_neurons, initial_firing_rates, e
    
    except Exception as e:
        e=f"Error reading nodes CSV: {e}"
        return [], {}, {}, {}, {}, e
    

def read_edges_csv(file_path):
    edges = []
    edge_probabilities = {}
    edge_strenghs = {}
    edge_delay = {}
    edge_receptor = {}
    
    try:
        edges_df = pd.read_csv(file_path)
        
        for _, row in edges_df.iterrows():
            population1 = str(row['Population1'])
            population2 = str(row['Population2'])
            probability = row['Probability']
            strength = row['Strength']
            delay = row['Delay']
            receptor = row['Receptor']

            edges.append((population1, population2))
            edge_probabilities[(population1, population2)] = probability
            edge_strenghs[(population1, population2)] = strength
            edge_delay[(population1, population2)] = delay
            edge_receptor[(population1, population2)] = receptor
        
        return edges, edge_probabilities, edge_strenghs, edge_delay, edge_receptor
    
    except Exception as e:
        print(f"Error reading edges CSV: {e}")
        return [], {}
    

def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)



