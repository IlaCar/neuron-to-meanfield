import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from matplotlib.gridspec import GridSpec
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import sys
from networkx.drawing.nx_pydot import graphviz_layout
from netgraph import InteractiveGraph

from cellular_parameters import loadparams
from functions import OU,double_gaussian,read_nodes_csv,read_edges_csv,TF_general
from mean_field_network_second_order import *


# List of nodes and connections between them:
nodes_file_path = f'mean_field_data/nodes_two_mean_field.csv'  # CSV file with node details
edges_file_path = f'mean_field_data/edges_two_mean_field.csv'  # CSV file with edge details

# Reading the information about the nodes and connectivity
populations, population_types, node_population, number_of_neurons, initial_firing_rates, error_msg = read_nodes_csv(nodes_file_path)
edges, edge_probabilities, edge_strength, edge_delay, edge_receptor = read_edges_csv(edges_file_path)

if error_msg.strip():
    print(error_msg)
    sys.exit(1)

# Neuronal Parameters
state='awake'  # use sleep or awake
cell_param={}
cell_param['RS']=loadparams(f'RS-FS_{state}')['RS']
cell_param['FS']=loadparams(f'RS-FS_{state}')['FS']

# mapping row number to the node
population_info={}
temp=0
for n in populations:
    population_info[n]=int(temp)
    temp=temp+1

# ==========================================
# Simulation Timing Parameters
# ==========================================

T=15e-3  # in seconds
dt = 0.0001 # in seconds
t_start = 0 # in seconds
t_end = 3.0  # in seconds
n_steps = int((t_end - t_start) / dt)
n_steps = n_steps+1
times=np.arange(t_start,t_end+2*dt,dt)

# ==========================================
# Defining different stimulus
# ==========================================
i_ext=0.4
ou = OU(t_end+dt+dt,dt) + i_ext
sensory_stimulus_square=np.zeros(n_steps+1)
sensory_stimulus_square[int(1/dt):int(1.5/dt)]+=3
sensory_stimulus_Dgauss=double_gaussian(times,1,0.02,0.5,2.0)
constant_input=np.full(n_steps+1,1)
amplitude,frequency,offset=10,2,0; osc_input=offset+amplitude/2*(1-np.cos(2*frequency*np.pi*times))
zero_input=np.zeros(n_steps+1)

# ==========================================
# Assigning external stimulus to the populations
# ==========================================
RS1_input=sensory_stimulus_square
FS1_input=ou
RS2_input=ou
FS2_input=ou
FS3_input=ou
RS10_input=ou

TC_input=ou
RE_input=ou

external_stimulus={'RS1': RS1_input,'FS10': FS1_input,'FS20': FS1_input,'RS2': RS2_input,'FS2': FS2_input,'FS3':FS3_input,'RS10':RS10_input}  
external_stimulus_receptor={'RS1':'Glutamate','FS10':'Glutamate','FS20':'Glutamate','RS2':'Glutamate','FS2':'Glutamate','FS3':'Glutamate','RS10':'Glutamate'}
external_stimulus_number={'RS1':8000,'FS10':2000,'FS20':1000,'RS2':8000,'FS2':2000,'FS3':2000,'RS10':1000}
external_stimulus_prob={'RS1':0.05,'FS10':0.05,'FS20':0.05,'RS2':0.05,'FS2':0.05,'FS3':0.05,'RS10':0.05}

# ==========================================
# Simulation
# ==========================================
G,firing_rates,adaptation,correlation,correlation_mapping=mean_field_network_second_order(cell_param,populations,population_info,population_types,node_population,number_of_neurons,edges,edge_probabilities,edge_strength,edge_delay,edge_receptor,initial_firing_rates,external_stimulus,external_stimulus_receptor,external_stimulus_number,external_stimulus_prob,T,t_start,t_end,dt,n_steps)


# %%
# ==========================================
# Setting up visualization of Results
# ==========================================

colors = ['#1f77b4', '#b4476d', '#ff7f0e', '#0e7dff', '#d62728', '#28d6b6', '#2ca02c', '#a02ca0', '#9467bd', '#bd9467', '#ffbb78', '#78bbff', '#17becf', '#cf7e17', '#ff6347', '#47ff63', '#00b5e2', '#e2b500', '#e2007a', '#7ae2b5', '#006400', '#640063', '#d2691e', '#1ed2d2', '#ff4500', '#00aaff', '#8b0000', '#00b38b', '#f4a300', '#00a3f4']
i=0

population_dict={}
for key, value in node_population.items():
    if value not in population_dict:
        population_dict[value] = []
    population_dict[value].append(key)  

cluster_graph = nx.complete_graph(len(population_dict.keys()))
cluster_pos = nx.spring_layout(cluster_graph,scale=20.0)

pos={}

for k, (net_name, node_pops) in enumerate(population_dict.items()):
    center = cluster_pos[k]
    sub_G = G.subgraph(node_pops)
    sub_pos = nx.circular_layout(sub_G, center=center, scale=5.0)
    pos.update(sub_pos)
    
## ---- Color based on type and Receptor ----
uniq_population_types=np.unique(np.array(list(population_types.values())))
pop_colors_dict={}
for t in uniq_population_types:
    pop_colors_dict[t]=colors[i]
    i=i+1

uniq_receptor_types=np.unique(np.array(list(edge_receptor.values())))
receptor_colors_dict={}
for t in uniq_receptor_types:
    receptor_colors_dict[t]=colors[i]
    i=i+1

pop_colors=[]
for node in G.nodes:
    pop_colors.append(pop_colors_dict[population_types[node]])

edge_colors=[]
for e in G.edges:
    edge_colors.append(receptor_colors_dict[edge_receptor[e]])

#%%
# ==========================================
# Visualization of Network
# ==========================================
fig = plt.figure(figsize=(50, 30))
gs = GridSpec(1,1,figure=fig)
plot1=fig.add_subplot(gs[:,:])

nx.draw(G, pos, ax=plot1, with_labels=False, node_size=2000, node_color=pop_colors, edge_color=edge_colors,
        font_size=12, font_weight='bold', width=2, arrows=True, arrowsize=8, arrowstyle="->", connectionstyle="arc3,rad=-0.09") # Draw the network

edge_labels = {(u, v): f"{G[u][v]['probability']:.2f}" for u, v in G.edges}
# nx.draw_networkx_edge_labels(G, pos, ax=plot1, edge_labels=edge_labels, font_size=10, font_color='black') # Draw connection probabilities
                
node_labels = {node: f"{node}" for node in G.nodes}
nx.draw_networkx_labels(G, pos, ax=plot1, labels=node_labels, font_size=15) # Drawing the Labels 

# neurons_N_labels = {p: f"N: {G.nodes[p]['neurons_N']}" for p in G.nodes}
# neurons_N_pos = {p: (pos[p][0], pos[p][1] - 0.04) for p in G.nodes}  # Slightly offset
# nx.draw_networkx_labels(G, neurons_N_pos, ax=plot1, labels=neurons_N_labels, font_size=10, font_color='black') # Displaying number of neurons in population

plot1.set_title('Visualization of the connectivity between neuronal populations')

#%%
# ==========================================
# INTERACTIVE Visualization
# ==========================================
fig = plt.figure(figsize=(50, 20))
gs = GridSpec(1, 1, figure=fig)
plot2 = fig.add_subplot(gs[:, :])

# Dictionary mappings required by Netgraph
pop_colors_dict = {node: pop_colors_dict[population_types[node]] for node in G.nodes}
edge_colors_dict = {e: receptor_colors_dict[edge_receptor[e]] for e in G.edges}

plot_instance = InteractiveGraph(G,node_positions=pos,ax=plot2,node_size=3,node_color=pop_colors_dict,node_labels=node_labels,node_label_fontdict=dict(size=8, weight='bold', color='black'),
    edge_color=edge_colors_dict,edge_width=0.4,edge_labels=edge_labels,edge_label_fontdict=dict(size=6, color='black'),arrows=True,edge_layout='arc')

plot2.set_title('Interactive Visualization', fontsize=10)


# %%
# ==========================================
# Plotting Activity
# ==========================================

no_of_nodes = len(population_dict)
figure_per_row = 2 
num_rows = int(np.ceil(no_of_nodes / figure_per_row))

fig2, ax2 = plt.subplots(num_rows, figure_per_row, figsize=(12, 12)) # Create subplots based on the number of nodes 
axes = np.atleast_1d(ax2).flatten() 

for idx, (p_type, pops_in_type) in enumerate(population_dict.items()):  # Iterate over each node and its associated populations
    splt = axes[idx]
    
    for i, pop in enumerate(pops_in_type): # Plot every population belonging to same node on the same axis
        splt.plot(times, firing_rates[population_info[pop]],color=colors[i % len(colors)], label=f'{pop}', linewidth=1.0)

    splt.set_title(f"Node: {p_type}", fontsize=10, fontweight='bold')
    splt.set_ylabel("Firing Rate (Hz)", fontsize=8)
    splt.set_xlabel("Time (in seconds)", fontsize=8)
    splt.set_xlim(0, t_end) # to crop the plot along x-axis
    splt.legend(fontsize=8)


for idx in range(no_of_nodes, len(axes)): # To Hide any extra empty subplots 
    fig2.delaxes(axes[idx]) 

fig2.subplots_adjust(hspace=0.2, wspace=0.4)
plt.show()