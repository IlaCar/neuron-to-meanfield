import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from cellular_parameters import loadparams
from functions import OU,TF_general,adaptation_equation_general



def mean_field_network_second_order(cell_param,populations,population_info,population_types,node_population,number_of_neurons,edges,edge_probabilities,edge_strength,edge_delay,edge_receptor,initial_firing_rates,external_stimulus,external_stimulus_receptor,external_stimulus_number,external_stimulus_prob,T,t_start,t_end,dt,n_steps):

    ## Making directed graph with self loops based on node and edge information 
    G = nx.DiGraph()

    for pop in populations:
        G.add_node(pop, type=population_types[pop], neurons_N=number_of_neurons[pop], population=node_population[pop], subnet=node_population[pop])

    for edge in edges:
        source, target = edge
        probability = edge_probabilities[edge]  
        G.add_edge(source,target,probability=probability)

    #------------------------
    #  Initialization
    #------------------------
    global_scaling_factor=1.0
    # node_models_used=set(population_types.values())

    firing_rates=np.zeros((np.size(populations),n_steps+1),dtype=float) # for storing the firing rates of every pop - [ row for every neuronal poulation ]
    adaptation=np.zeros((np.size(populations),n_steps+1),dtype=float) # for storing the adaptation  - [ row for every neuronal poulation ]

    for n in population_info.keys():
        firing_rates[population_info[n]]=initial_firing_rates[n]   # setting up initial firing rate [ assuming that at t<=0, pop's firing rate is equal to initial firing rate ]
        adaptation[population_info[n]]=firing_rates[population_info[n]]*cell_param[n[0:2]].b*cell_param[n[0:2]].Tw

    population_dict = {} # dictionary to make list of populations belong to each population
    correlation_mapping = {}  # symbol table for accesing and storing the correlations
    temp=0

    for key, value in node_population.items():
        if value not in population_dict:
            population_dict[value] = []
        population_dict[value].append(key)  

    no_of_correlations=0
    for p in population_dict.keys():
        no_of_correlations=no_of_correlations + len(population_dict[p])**2
        correlation_mapping[p]={}

        populations_in_p=population_dict[p]
        for c in range(len(population_dict[p])):
            for k in range(len(population_dict[p])):
                idx=f'{population_info[populations_in_p[c]]}_{population_info[populations_in_p[k]]}'  # correlation between node c and k
                correlation_mapping[p][idx]=temp  # assigning unique no to each correlation
                temp=temp+1

    correlation=np.ones((no_of_correlations,n_steps+1))  # for saving correlation among neuronal populations

    #------------------------
    #  Simulation
    #------------------------
    max_delay=int((max(edge_delay.values())*1e-3)/dt)  # getting maximum delay among populations in given network
    print("max delay:",max_delay*dt*1000," milli-seconds")

    eps=np.finfo(float).eps # minimum value of step size of numerical simulation
    min_df=eps**(1./3.)
    df=min_df*100
    print("eps:",f"{eps:.2e}","-- minimum value of df:",f"{min_df:.2e}","-- Value used:",f"{df:.2e}")
      
    for t in np.arange(0,n_steps,1):
        print(f"Progress: {int(100 * t / n_steps)}% ", end="\r", flush=True)

        for node,nodes_pop in population_dict.items():

            F_tf={}
            input_node_activity_all={}

            for pop in nodes_pop:
                node_models=np.array([], dtype=str)
                for i in G.predecessors(pop):
                    node_models=np.append(node_models,edge_receptor[i,pop])
                
                node_models=np.unique(node_models)
                input_node_activity={key: 0 for key in node_models}

                for i in G.predecessors(pop):
                    delay=int((edge_delay[i,pop]/dt)*1e-3)
                    input_node_activity[edge_receptor[i,pop]]=input_node_activity[edge_receptor[i,pop]]+(G.nodes[i]['neurons_N']*G[i][pop]['probability']*firing_rates[population_info[i],t-delay]*edge_strength[i,pop])
                
                input_node_activity[external_stimulus_receptor[pop]]=input_node_activity[external_stimulus_receptor[pop]] + external_stimulus[pop][t]*external_stimulus_prob[pop]*external_stimulus_number[pop]

                corr_term=0

                for lamda in nodes_pop:
                    for eta in nodes_pop:

                        if (lamda,pop) in edge_receptor.keys() and (eta,pop) in edge_receptor.keys():

                            lamda_model=edge_receptor[lamda,pop]
                            eta_model=edge_receptor[eta,pop]

                            if lamda_model != eta_model:
                                diff1=input_node_activity.copy()
                                diff1[lamda_model]=diff1[lamda_model]+df
                                diff1[eta_model]=diff1[eta_model]+df

                                diff2=input_node_activity.copy()
                                diff2[lamda_model]=diff2[lamda_model]+df
                                diff2[eta_model]=diff2[eta_model]-df

                                diff3=input_node_activity.copy()
                                diff3[lamda_model]=diff3[lamda_model]-df
                                diff3[eta_model]=diff3[eta_model]+df

                                diff4=input_node_activity.copy()
                                diff4[lamda_model]=diff4[lamda_model]-df
                                diff4[eta_model]=diff4[eta_model]-df
                                                        
                                corr_term =  corr_term + (correlation[correlation_mapping[node][f'{population_info[lamda]}_{population_info[eta]}']][t]*((TF_general(cell_param[population_types[pop]],diff1,adaptation[population_info[pop],t])-TF_general(cell_param[population_types[pop]],diff2,adaptation[population_info[pop],t])-TF_general(cell_param[population_types[pop]],diff3,adaptation[population_info[pop],t])+TF_general(cell_param[population_types[pop]],diff4,adaptation[population_info[pop],t]))/(4*df*df)))
                                del diff1,diff2,diff3,diff4

                            if lamda_model == eta_model:
                                diff1=input_node_activity.copy()
                                diff1[lamda_model]=diff1[lamda_model]+df

                                diff2=input_node_activity.copy()
                                diff2[lamda_model]=diff2[lamda_model]-df
                        
                                corr_term =  corr_term + (correlation[correlation_mapping[node][f'{population_info[lamda]}_{population_info[eta]}']][t]*(TF_general(cell_param[population_types[pop]],diff1,adaptation[population_info[pop],t])-2*TF_general(cell_param[population_types[pop]],input_node_activity,adaptation[population_info[pop],t])+TF_general(cell_param[population_types[pop]],diff2,adaptation[population_info[pop],t]))/(df*df))
                                del diff1,diff2

                        # if corr_term < 0:
                        #     print("---> ",t,lamda,eta,corr_term,correlation[correlation_mapping[node][f'{population_info[lamda]}_{population_info[eta]}']][t])

                input_node_activity_all[pop]=input_node_activity

                has_nan = any(np.isnan(v) for v in input_node_activity.values())
                if has_nan==True:
                    print("Invalid Values: ",t,pop,input_node_activity,firing_rates[population_info[pop],t])

                Fe=TF_general(cell_param[population_types[pop]],input_node_activity,adaptation[population_info[pop],t])
                F_tf[pop]=Fe

                firing_rates[population_info[pop],t+1] = firing_rates[population_info[pop],t]+(((Fe-firing_rates[population_info[pop],t]) + corr_term)*dt)/T
                adaptation[population_info[pop],t+1] = adaptation[population_info[pop],t] + dt * (adaptation_equation_general(cell_param[population_types[pop]],input_node_activity,firing_rates[population_info[pop],t],adaptation[population_info[pop],t]))

                if (firing_rates[population_info[pop],t+1]<0) | (np.isnan(firing_rates[population_info[pop],t+1])):
                    print(t,pop,firing_rates[population_info[pop],t+1])


            for mu in nodes_pop:
                for vee in nodes_pop:
                    mu_model=population_types[mu]
                    vee_model=population_types[vee]

                    cross_term=0
                    for lamda in nodes_pop:

                        if (lamda,mu) in edge_receptor.keys():
                            mu_model=edge_receptor[lamda,mu]
                        
                            diff1=input_node_activity_all[mu].copy()
                            diff1[mu_model]=diff1[mu_model]+df

                            diff2=input_node_activity_all[mu].copy()
                            diff2[mu_model]=diff2[mu_model]-df
                            
                            cross_term = cross_term + (correlation[correlation_mapping[node][f'{population_info[vee]}_{population_info[lamda]}']][t]*((TF_general(cell_param[population_types[mu]],diff1,adaptation[population_info[mu],t])-TF_general(cell_param[population_types[mu]],diff2,adaptation[population_info[mu],t]))/(2*df)))
                            del diff1,diff2

                        if (lamda,vee) in edge_receptor.keys():
                            vee_model=edge_receptor[lamda,vee]

                            diff3=input_node_activity_all[vee].copy()
                            diff3[vee_model]=diff3[vee_model]+df

                            diff4=input_node_activity_all[vee].copy()
                            diff4[vee_model]=diff4[vee_model]-df

                            cross_term = cross_term + (correlation[correlation_mapping[node][f'{population_info[mu]}_{population_info[lamda]}']][t]*((TF_general(cell_param[population_types[vee]],diff3,adaptation[population_info[vee],t])-TF_general(cell_param[population_types[vee]],diff4,adaptation[population_info[vee],t]))/(2*df)))
                            del diff3,diff4

                           
                    if mu == vee:
                        cross_term = cross_term + (F_tf[mu]/G.nodes[mu]['neurons_N'])*((1/T)-F_tf[mu])

                    correlation[correlation_mapping[node][f'{population_info[mu]}_{population_info[vee]}']][t+1] = correlation[correlation_mapping[node][f'{population_info[mu]}_{population_info[vee]}']][t] + (dt/T)*(((F_tf[mu]-TF_general(cell_param[population_types[mu]],input_node_activity_all[mu],adaptation[population_info[mu],t]))*(F_tf[vee]-TF_general(cell_param[population_types[vee]],input_node_activity_all[vee],adaptation[population_info[vee],t]))) + cross_term - (2*correlation[correlation_mapping[node][f'{population_info[mu]}_{population_info[vee]}']][t]))
                    # print("Correlation:",t,correlation[correlation_mapping[node][f'{population_info[mu]}_{population_info[vee]}']][t+1])

            
            
    return G,firing_rates,adaptation,correlation,correlation_mapping

