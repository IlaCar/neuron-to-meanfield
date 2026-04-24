import json
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

import brian2 as b2

# -------------------- #
color_palette = {"FS": '#cb181d',           # red
                 "RS": '#238b45',           # green
                 "RS_no_adapt": '#2171b5',  # blue
                 "input": '#9ecae1'}        # light blue

syn_colors = {
    "E": "#3d9a8e",     # teal
    "I": "#6a3d9a",     # purple
    "Total": "#000000"  # black
}
# -------------------- #                 
def get_pretty_voltage(volt, thresh):
    for i in range(len(volt) - 1):
        if volt[i] > thresh * b2.mV and volt[i+1] < volt[i]:
            volt[i] = 0 * b2.mV #forcing peak to be 0 mV
    return volt

# -------------------- #
def plotting_3_traces_per_population(pop1 = None, 
                                     pop2 = None, 
                                     ext_input_0 = None,
                                     ext_input_1 = None,
                                     ext_input_2 = None,
                                     ext_input_3 = None,
                                     ext_input_4 = None,
                                     pretty_plot = False,
                                     RS_adaptation = True):

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    if pretty_plot == False:
        ax[0].plot(pop1.t/b2.second, pop1.v[0] / b2.mV, color='#67000d')
        ax[0].plot(pop1.t/b2.second, pop1.v[1] / b2.mV, color='#cb181d')
        ax[0].plot(pop1.t/b2.second, pop1.v[2] / b2.mV, color='#fb6a4a')
    else:
        ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[0], -50) / b2.mV, '--', color='#67000d')
        ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[1], -50) / b2.mV, '--', color='#cb181d')
        ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[2], -50) / b2.mV, '--', color='#fb6a4a')  
    ax[0].set_title('Selected FS traces')

    if RS_adaptation == True:
        color_RS = color_palette['RS']
        col_0 = '#00441b'
        col_1 = '#238b45'
        col_2 = '#74c476'
    else:
        color_RS = color_palette['RS_no_adapt']
        col_0 = '#08306b'
        col_1 = '#2171b5'
        col_2 = '#6baed6'
    

    if pretty_plot == False:    
        ax[1].plot(pop2.t/b2.second, pop2.v[0] / b2.mV, color=col_0)
        ax[1].plot(pop2.t/b2.second, pop2.v[1] / b2.mV, color=col_1)
        ax[1].plot(pop2.t/b2.second, pop2.v[2] / b2.mV, color=col_2)
    else:
        ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[0], -50) / b2.mV, '--', color=col_0)   
        ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[1], -50) / b2.mV, '--', color=col_1)
        ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[2], -50) / b2.mV, '--', color=col_2)
    
    ax[1].set_title('Selected RS traces')
   
    if ext_input_0 != None:
        ax[-1].plot(ext_input_0.t / b2.second, ext_input_0.i, '.', color='k', alpha=0.3, markersize=1)
    ax[-1].plot(ext_input_1.t / b2.second, ext_input_1.i, '.', color=color_palette['FS'], markersize=1)
    ax[-1].plot(ext_input_2.t / b2.second, ext_input_2.i, '.', color=color_RS, markersize=1)
    ax[-1].plot(ext_input_3.t / b2.second, ext_input_3.i, '.', color=color_palette['FS'], markersize=1)
    ax[-1].plot(ext_input_4.t / b2.second, ext_input_4.i, '.', color=color_RS, markersize=1)
    
    ax[-1].set_ylabel('External Input Neuron index')
    ax[-1].set_title('External Poisson input spike raster')
    
    ax[-1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Membrane potential (mV)')
    
    plt.tight_layout()
    
    return fig

# -------------------- #
def plotting_3_traces(neuron_model = None,
                      pop = None, 
                      input_interval = None):

    if neuron_model == 'FS':
        colors = ['#67000d', '#cb181d', '#fb6a4a']
    if neuron_model == 'RS':
        colors = ['#00441b', '#238b45', '#74c476']
    if neuron_model == 'RS_no_adapt':    
        colors = ['#08306b', '#2171b5', '#6baed6']    
    fig, ax = plt.subplots(figsize=(10, 6), sharex=True)
    ax.plot(pop.t/b2.second, pop.v[0] / b2.mV, color=colors[0])
    ax.plot(pop.t/b2.second, get_pretty_voltage(pop.v[0], -50) / b2.mV, '--', color=colors[0])
    ax.plot(pop.t/b2.second, pop.v[1] / b2.mV, color=colors[0])
    ax.plot(pop.t/b2.second, get_pretty_voltage(pop.v[1], -50) / b2.mV, '--', color=colors[1])
    ax.plot(pop.t/b2.second, pop.v[2] / b2.mV, color=colors[0])
    ax.plot(pop.t/b2.second, get_pretty_voltage(pop.v[2], -50) / b2.mV, '--', color=colors[2])
    ax.set_title(f'Selected {neuron_model} traces')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane potential (mV)')
    
    plt.tight_layout()
    
    return fig

# -------------------- #
def add_input_boxes(ax,
                    exc_intervals = None,
                    inh_intervals = None,
                    input_interval = None,
                    n_neurons = None,
                    box_height = 300,
                    pad = 200,
                    alpha = 0.3,
                    annotate = True,
                    ):
    """
    Add excitatory (+) and inhibitory (-) input boxes to a raster plot.
    """

    if n_neurons is None:
        raise ValueError("n_neurons must be provided")
    
    # y positions
    y_pad = - pad - box_height

    # excitatory inputs
    if exc_intervals is not None:
        for t0, t1, pop in exc_intervals:
            rect = patches.Rectangle(
                #(t0, y_exc),
                (t0, y_pad),
                t1 - t0,
                box_height,
                facecolor=syn_colors['E'],
                edgecolor=None,
                alpha=alpha,
            )
            ax.add_patch(rect)

            if annotate:
                ax.text(
                    (t0 + t1) / 2,
                    y_pad + box_height / 2,
                    f"{pop}+",
                    ha="center",
                    va="center",
                    fontsize=10
                )

    # inhibitory inputs
    if inh_intervals is not None:
        for t0, t1, pop in inh_intervals:
            rect = patches.Rectangle(
                (t0, y_pad),
                t1 - t0,
                box_height,
                facecolor=syn_colors['I'],
                edgecolor=None,
                alpha=alpha,
            )
            ax.add_patch(rect)

            if annotate:
                ax.text(
                    (t0 + t1) / 2,
                    y_pad + box_height / 2,
                    f"{pop}−",
                    ha="center",
                    va="center",
                    fontsize=10
                )
                
    # general input -- exc and inh together
    if input_interval is not None:
        for t0, t1 in input_interval:
            rect = patches.Rectangle(
                (t0, y_pad),
                t1 - t0,
                box_height,
                facecolor=color_palette["input"],
                edgecolor=None,
                alpha=alpha,
            )
            ax.add_patch(rect)

            if annotate:
                ax.text(
                    (t0 + t1) / 2,
                    y_pad + box_height / 2,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                )
    
    return ax

# -------------------- #
def network_raster_plot(pop1=None, 
                        pop2=None, 
                        N_pop1=None,
                        N_pop2=None,
                        markersize=None,
                        exc_intervals = None,
                        inh_intervals = None,
                        x_lim=None,
                        RS_adaptation=True,
                        input_boxes = True
                        ):
    
    m_size = 1 if markersize is None else markersize
        
    fig, ax = plt.subplots(figsize=(10, 6))
    #fig, ax = plt.subplots(figsize=(10, 3))
    offset = 0

    if pop1 is not None:
        ax.plot(
            pop1.t / b2.second,
            pop1.i + offset,
            ',',
            color=color_palette['FS'],
            markersize=m_size,
            rasterized = True
        )
        offset += N_pop1

    if pop2 is not None:
        if RS_adaptation == True:
            color = color_palette['RS']
        else:
            color = color_palette['RS_no_adapt']
        ax.plot(
            pop2.t / b2.second,
            pop2.i + offset,
            ',',
            color=color,
            markersize=m_size,
            rasterized = True            
        )
        offset += N_pop2
   
    # X axis
    if x_lim is not None:
        ax.set_xlim(x_lim)

    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Y axis: population boundaries
    total_neurons = N_pop1 + N_pop2
    yticks = [0, N_pop1, total_neurons]
    ax.set_yticks(yticks)
    ax.set_ylabel('Neurons')
    ax.set_ylim(-800, total_neurons + 800)

    # Population labels
    ax.text(
        0.01,  # x in axes coords
        N_pop1 / 2,
        'FS',
        color = color_palette['FS'],
        transform=ax.get_yaxis_transform(),
        va='center',
        ha='left',
        fontsize=12,
    )

    ax.text(
        0.01,
        N_pop1 + N_pop2 / 2,
        'RS',
        color = color,        
        transform=ax.get_yaxis_transform(),
        va='center',
        ha='left',
        fontsize=12,
    )

    ax.set_title('Network Raster Plot')

    # Input boxes 
    if input_boxes == True:
        ax = add_input_boxes(
        ax=ax,
        exc_intervals=exc_intervals,
        inh_intervals=inh_intervals,
        n_neurons=total_neurons,
        box_height=300,
        pad=200,
        alpha=0.3,
        annotate=True,
    )
    
    return fig

# -------------------- #
def bin_align_intervals(intervals, bin_size, time_bins):
    """
    Convert continuous-time intervals into intervals covering
    exactly the bins that contain input.
    """
    aligned = []

    #for t0, t1, pop in intervals:
    for t0, t1, *res in intervals:
        pop = res[0] if res else None
        
        # bins that contain input
        k0 = int(np.floor(t0 / bin_size))
        k1 = int(np.ceil(t1 / bin_size)) - 1

        # clip to valid bins
        k0 = max(k0, 0)
        k1 = min(k1, len(time_bins) - 1)

        #import pdb
        #pdb.set_trace()
        
        # map back to plotted bin coordinates
        t0_aligned = time_bins[k0]
        t1_aligned = time_bins[k1]

        if pop is not None:
            aligned.append((t0_aligned, t1_aligned, pop))
        else:
            aligned.append((t0_aligned, t1_aligned))

    return aligned
    
# -------------------- #
def plotting_pop_freq_and_std(sim_duration = None, 
                              pop1 = None, 
                              pop2 = None, 
                              N_pop1 = None,
                              N_pop2 = None,
                              bin_size = None,
                              exc_intervals = None,
                              inh_intervals = None,
                              RS_adaptation = True,
                              input_boxes = True):

    # Parameters
    if bin_size == None:
        bin_size = 0.1 * b2.second
    bin_edges = np.arange(0, sim_duration + bin_size, bin_size)
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

    if RS_adaptation == True:
        color_RS = color_palette['RS']
    else:
        color_RS = color_palette['RS_no_adapt']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    #fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_bins, mean_rate_FS, '.-', label='avg FS freq', color=color_palette['FS'])
    ax.plot(time_bins, mean_rate_RS, '.-', label='avg RS freq', color=color_RS)
    
    ax.fill_between(
        time_bins,
        np.clip(mean_rate_FS - std_rate_FS, 0, None),    # to avoid negative firing rate
        mean_rate_FS + std_rate_FS,
        color=color_palette['FS'], alpha=0.3, label='± FS std'
    )
    
    ax.fill_between(
        time_bins,
        np.clip(mean_rate_RS - std_rate_RS, 0, None),    # to avoid negative firing rate
        mean_rate_RS + std_rate_RS,
        color=color_RS, alpha=0.3, label='± RS std'
    )
    
    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(MultipleLocator(1))    
    ax.set_ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    ax.set_title('Network Population firing rate ± std')

    
    exc_intervals_binned = bin_align_intervals(exc_intervals, bin_size, time_bins)
    inh_intervals_binned = bin_align_intervals(inh_intervals, bin_size, time_bins)
   
    # Input boxes 
    if input_boxes == True:
        ax = add_input_boxes(
        ax = ax,
        exc_intervals = exc_intervals_binned,
        inh_intervals = inh_intervals_binned,
        n_neurons = N_pop1 + N_pop2,
        box_height = 3,
        pad = 10,
        alpha = 0.3,
        annotate = True,
    )
    ticks = ax.get_yticks()
    ax.set_yticks(ticks[ticks >= 0])
    plt.legend()
    plt.tight_layout()
    
    return fig

# -------------------- #
def network_raster_plot_h5(pop1 = None, 
                        pop2 = None, 
                        N_pop1 = None,
                        N_pop2 = None,
                        markersize = None,
                        x_lim = None,
                        exc_intervals = None,
                        inh_intervals = None,
                        input_boxes = True):
    
    if markersize == None:
        m_size = 1
    else:
        m_size = markersize
        
    fig, ax = plt.subplots(figsize=(10, 6))
    offset = 0

    if pop1 != None:
        # Plot FS interneurons
        ax.plot(pop1['t'], pop1['i'] + offset, ',', color=color_palette['FS'], markersize=m_size)
        offset += N_pop1

    if pop2 != None:    
        # Plot RS excitatory neurons
        ax.plot(pop2['t'], pop2['i'] + offset, ',', color=color_palette['RS'], markersize=m_size)
        offset += N_pop2

    
    if x_lim is not None:
        ax.set_xlim(x_lim)

    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Y axis: population boundaries
    total_neurons = N_pop1 + N_pop2
    yticks = [0, N_pop1, total_neurons]
    ax.set_yticks(yticks)
    ax.set_ylabel('Neurons')
    ax.set_ylim(-800, total_neurons + 800)

    # Population labels
    ax.text(
        0.01,  # x in axes coords
        N_pop1 / 2,
        'FS',
        transform=ax.get_yaxis_transform(),
        va='center',
        ha='left',
        fontsize=12
    )

    ax.text(
        0.01,
        N_pop1 + N_pop2 / 2,
        'RS',
        transform=ax.get_yaxis_transform(),
        va='center',
        ha='left',
        fontsize=12
    )

    ax.set_title('Network Raster Plot')

    # Input boxes 
    if input_boxes == True:
        ax = add_input_boxes(
            ax=ax,
            exc_intervals=exc_intervals,
            inh_intervals=inh_intervals,
            n_neurons=total_neurons,
            box_height=300,
            pad=200,
            alpha=0.3,
            annotate=True,
        )
    
    return fig

# -------------------- #
def plotting_pop_freq_and_std_h5(sim_duration = None, 
                              pop1 = None, 
                              pop2 = None, 
                              N_pop1 = None,
                              N_pop2 = None,
                              bin_size = None,
                              exc_intervals = None,
                              inh_intervals = None,
                              RS_adaptation = True):

    # Parameters
    if bin_size == None:
        bin_size = 0.1 * b2.second
    bin_edges = np.arange(0, sim_duration + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix_FS = np.zeros((N_pop1, len(time_bins)))
    spike_matrix_RS = np.zeros((N_pop2, len(time_bins)))
    
    # Fill the spike matrix
    for i, t in zip(pop1['i'], pop1['t'] / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_FS[i, bin_idx] += 1
    
    for i, t in zip(pop2['i'], pop2['t'] / b2.second):
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

    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(time_bins, mean_rate_FS, '.-', label='avg FS freq', color=color_palette['FS'])
    if RS_adaptation == True:
        color_RS = color_palette['RS']
    else:
        color_RS = color_palette['RS_no_adapt']
    plt.plot(time_bins, mean_rate_RS, '.-', label='avg RS freq', color=color_RS)
    
    plt.fill_between(
        time_bins,
        np.clip(mean_rate_FS - std_rate_FS, 0, None),    # to avoid negative firing rate
        mean_rate_FS + std_rate_FS,
        color=color_palette['FS'], alpha=0.3, label='± FS std'
    )
    
    plt.fill_between(
        time_bins,
        np.clip(mean_rate_RS - std_rate_RS, 0, None),    # to avoid negative firing rate
        mean_rate_RS + std_rate_RS,
        color=color_RS, alpha=0.3, label='± RS std'
    )

    exc_intervals_binned = bin_align_intervals(exc_intervals, bin_size, time_bins)
    inh_intervals_binned = bin_align_intervals(inh_intervals, bin_size, time_bins)
   
    # Input boxes 
    ax = add_input_boxes(
        ax = ax,
        exc_intervals = exc_intervals_binned,
        inh_intervals = inh_intervals_binned,
        n_neurons = N_pop1 + N_pop2,
        box_height = 3,
        pad = 10,
        alpha = 0.3,
        annotate = True,
    )

    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(MultipleLocator(1))    
    ax.set_ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    ax.set_title('Network Population firing rate ± std')    
    
    ticks = ax.get_yticks()
    ax.set_yticks(ticks[ticks >= 0])
    plt.legend()
    plt.tight_layout()
    
    return fig

# -------------------- #
def disconnected_network_raster_plot_TF(neuron_model=None,
                                       pop=None, 
                                       N_pop=None,
                                       markersize=None,
                                       input_interval = None,
                                       x_lim=None
                                       ):
    
    m_size = 1 if markersize is None else markersize
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pop.t / b2.second, pop.i, 'o', color=color_palette[neuron_model], markersize=m_size)
   
    # X axis
    if x_lim is not None:
        ax.set_xlim(x_lim)

    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel(f'{neuron_model} Neurons')
    ax.set_title('Disconnected Network Raster Plot')

   
    # Input boxes 
    ax = add_input_boxes(
        ax = ax,
        input_interval = input_interval,
        n_neurons = N_pop,
        box_height = 3,
        pad = 5,
        alpha = 0.3,
        annotate = True,
    )
    ticks = [0, N_pop]
    ax.set_yticks(ticks)
    ax.set_ylim(-10, N_pop*1.1)
    plt.tight_layout()

    return fig

# -------------------- #
def plotting_single_pop_freq_and_std(sim_duration = None,
                                     neuron_model = None, 
                                     pop = None, 
                                     N_pop = None,
                                     bin_size = None,
                                     input_interval = None,
                                     input_boxes = True):

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
    
    if neuron_model == 'FS':
        color = color_palette['FS']
    if neuron_model == 'RS':
        color = color_palette['RS']
    if neuron_model == 'RS_no_adapt':
        color = color_palette['RS_no_adapt']
        
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(time_bins, mean_rate, label='avg freq', color = color)
    
    plt.fill_between(
        time_bins,
        np.clip(mean_rate - std_rate, 0, None),    # to avoid negative firing rate
        mean_rate + std_rate,
        color= color, alpha=0.3, label='± std'
    )

    input_interval_binned = bin_align_intervals(input_interval, bin_size, time_bins)
   
    # Input boxes 
    if input_boxes == True:
        ax = add_input_boxes(
        ax = ax,
        input_interval = input_interval_binned,
        n_neurons = N_pop,
        box_height=2,
        pad=2,
        alpha=0.3,
        annotate = True,
    )
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    plt.title(f'Network {neuron_model} Population firing rate ± std')
    plt.legend()
    ticks = ax.get_yticks()
    ax.set_yticks(ticks[ticks >= 0])    
    plt.tight_layout()
    
    return fig

# -------------------- #
# The heatmap and annotate_heatmap are taken from:
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
# and modified a bit
def heatmap_vars(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", plt_title=None,
            x_lab=None, y_lab=None,**kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, origin='lower', **kwargs) ## added origin = 'lower' to have (0,0) at the bottom left corner

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7, aspect=20, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.   
    if len(col_labels) < 20:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
    else:
        ax.set_xticks(np.arange(len(col_labels))[::4])
        ax.set_xticklabels(col_labels[::4])
    
    if len(row_labels) < 20:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticks(np.arange(len(row_labels))[::4])
        ax.set_yticklabels(row_labels[::4])    
        
	# Create x and y labels
    if x_lab == None and y_lab == None:
        ax.set_xlabel('Incoming on RS freq (Hz)', va="top")
        ax.set_ylabel('Incoming on FS freq (Hz)')
    else:
        ax.set_xlabel(x_lab, va="top")
        ax.set_ylabel(y_lab)
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(True)


    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(plt_title)
    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# -------------------- #
def heatmap_InOut(exc_vals, inh_vals, data_array, neuron_type):
    fig = plt.figure(figsize=(10,6))
    if neuron_type == 'FS':
        cmap_color = 'Reds'
    if neuron_type == 'RS':
        cmap_color = 'Greens'    
    if neuron_type == 'RS_no_adapt':
        cmap_color = 'Blues'  
        
    plt.imshow(data_array, aspect='auto', origin='lower',
               extent=[exc_vals[0], exc_vals[-1], inh_vals[0], inh_vals[-1]],
               cmap=cmap_color)
    
    plt.colorbar(label='Output frequency (Hz)')
    plt.xlabel('Freq Excitatory synapses (Hz)')
    plt.ylabel('Freq Inhibitory synapses (Hz)')
    plt.title(f'{neuron_type} input–output relation')
    
    return fig
    
# -------------------- #
def plot_InOut_relation(exc_vals, inh_vals, data_array, std_data_array, neuron_type):
    
    fig = plt.figure(figsize=(10,6))
    
    if neuron_type == 'FS':
        cmap_color = 'autumn_r'
    if neuron_type == 'RS':
        cmap_color = 'summer'
    if neuron_type == 'RS_no_adapt':
        cmap_color = 'winter_r'    
    cmap = cm.get_cmap(cmap_color, len(inh_vals))

    for i in range(len(inh_vals)):
        if i % 5 == 0:
            plt.plot(exc_vals, data_array[i], '-o', color=cmap(i), label=f'{inh_vals[i]}')
        else:
            plt.plot(exc_vals, data_array[i], '-o', color=cmap(i))
        plt.errorbar(exc_vals, data_array[i], std_data_array[i], fmt='-o', color = cmap(i), alpha=0.5)
    plt.xlabel('Freq Excitatory synapses (Hz)')
    plt.ylabel('Output frequency (Hz)')
    plt.title(f'{neuron_type} input-output relation')
    plt.legend(title='Freq Inh Syn (Hz)')
    
    return fig

# -------------------- #
def plot_contours(exc_vals, inh_vals, data_array, neuron_type):
    
    fig = plt.figure(figsize=(10,6))
    
    if neuron_type == 'FS':
        cmap_color = 'Reds'
    if neuron_type == 'RS':
        cmap_color = 'Greens'
    if neuron_type == 'RS_no_adapt':
        cmap_color = 'Blues' 
    
    cs = plt.contourf(exc_vals, inh_vals, data_array, levels=20, cmap=cmap_color)

    plt.colorbar(cs, label='Output frequency (Hz)')
    plt.xlabel('Freq Excitatory synapses (Hz)')
    plt.ylabel('Freq Inh Syn (Hz)')
    plt.title(f'{neuron_type} input–output contours')
    
    return fig

# -------------------- #
def plot_gain(exc_vals, inh_vals, data_array, neuron_type):
    
    fig = plt.figure(figsize=(10,6))
        
    gain = np.gradient(data_array, exc_vals, axis=1)
    plt.imshow(
        gain,
        aspect='auto',
        origin='lower',
        extent=[exc_vals[0], exc_vals[-1], inh_vals[0], inh_vals[-1]],
        cmap='coolwarm'
    )
    plt.colorbar(label='Gain (Hz/Hz)')
    plt.xlabel('Freq Excitatory synapses (Hz)')
    plt.ylabel('Freq Inh Syn (Hz)')
    plt.title(f'{neuron_type} gain map')
    
    return fig

# -------------------- #
def mesh_3d(exc_vals, inh_vals, data_array, neuron_type):
   
    if neuron_type == 'FS':
        cmap_color = 'Reds'
    if neuron_type == 'RS':
        cmap_color = 'Greens'
    if neuron_type == 'RS_no_adapt':
        cmap_color = 'Blues' 
        
    X, Y = np.meshgrid(exc_vals, inh_vals)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data_array, cmap = cmap_color)
    ax.set_xlabel('Freq Excitatory synapses (Hz)')
    ax.set_ylabel('Freq Inhibitory synapses (Hz)')   
    ax.set_zlabel('Output frequency (Hz)')
    ax.set_title(f'{neuron_type} input–output surface')
    
    return fig

# -------------------- #
def plot_TF_fitting(neuron_model, df_data, mean_error):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title(f'Transfer function of {neuron_model} cell')
    ax.set_ylabel('Output rate (Hz)')
    ax.set_xlabel('Excitatory input (Hz)')
    
       
    inp_exc = df_data['input_exc'].to_numpy()
    out_rate = df_data['avg_f_out'].to_numpy()
    fit_rate = df_data['fit_rate']
    ax.plot(inp_exc, out_rate, 'o', color= color_palette[neuron_model], label=f'{neuron_model} data')
    ax.plot(inp_exc, fit_rate, 'kx', markersize=7, label='fit')
    
    ax.text(0.5, 0.95, f'mean error: {mean_error:.2f} Hz', transform=ax.transAxes, ha='center')
    ax.legend()
    
    return fig

# -------------------- #
def plot_TF_fitting_viridis(neuron_model, df_data, mean_error, unique_inh, colors):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(f'Transfer function of {neuron_model} cell')
    ax.set_ylabel('Output rate (Hz)')
    ax.set_xlabel('Excitatory input (Hz)')
    
    
    for inh_val, c in zip(unique_inh, colors):
        mask = df_data['input_inh'] == inh_val
        inp_exc = df_data.loc[mask, 'input_exc'].to_numpy()
        out_rate = df_data.loc[mask, 'avg_f_out'].to_numpy()
        fit_rate = df_data.loc[mask, 'fit_rate']
    
        ax.plot(inp_exc, out_rate, 'o', color=c, alpha=0.5)
        ax.plot(inp_exc, fit_rate, 'x', color=c, markersize=7)
    ax.text(0.5, 0.95, f'mean error: {mean_error:.2f} Hz', transform=ax.transAxes, ha='center')    
    # colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=unique_inh.min(), vmax=unique_inh.max()))
    cbar = plt.colorbar(sm, ax=ax, alpha = 0.5)
    cbar.set_label('Inhibitory input (Hz)')
    
    return fig

# -------------------- #
def plots_TF_fitting(neuron_model, df_data, std_data, poly_params_2, params_SI, alpha, unique_inh, colors, y_lim = None):
    
    from ntmf.transfer_function import res_2_func
    distr_mean_error = np.zeros(len(unique_inh))
    idx = 0
    for fixed_inh, c in zip(unique_inh, colors):
        fig, ax = plt.subplots()
        ax.set_title(f'Transfer function of {neuron_model} cell')
        ax.set_ylabel('Output rate (Hz)')
        ax.set_xlabel('Excitatory input (Hz)')
    
        # Select a fixed inhibitory input
        tol = 1e-6  # tolerance in case of floating point noise
        mask = np.isclose(df_data['input_inh'], fixed_inh, atol=tol)
        
        inp_exc = df_data.loc[mask, 'input_exc'].to_numpy()
        out_rate = df_data.loc[mask, 'avg_f_out'].to_numpy()
        
        fit_rate = df_data.loc[mask,'fit_rate']
        mean_error = res_2_func(poly_params_2, data=df_data.loc[mask], params=params_SI, alpha=alpha)

        distr_mean_error[idx] = mean_error

        plt.errorbar(inp_exc, out_rate, std_data[idx], linestyle='None', color = c)
        ax.plot(inp_exc, out_rate, 'o', color = c, label=f'data (inh={fixed_inh} Hz)')
        ax.plot(inp_exc, fit_rate, 'kx', markersize=7, label='fit')
        
        ax.text(0.2, 0.95, f'mean error: {mean_error:.2f} Hz', transform=ax.transAxes, ha='center')
        ax.legend(loc = 'lower right')
        ax.set_ylim(y_lim if y_lim else (-5, 100))

        idx += 1
    return fig, distr_mean_error

# -------------------- #
def plots_TF_distr_mean_error(neuron_model, inh_vals, mean_error,
                              alpha = None, alpha_idx = None, fig = None):

    if fig == None:
        fig = plt.figure(figsize=(7,6))
    if alpha == None:
        fig = plt.figure(figsize=(7,6))
        plt.plot(inh_vals, mean_error, '.-', color = color_palette[neuron_model])

    else:
        if neuron_model == 'FS':
            cmap = plt.cm.get_cmap('Reds_r', 15)
        if neuron_model == 'RS':
            cmap = plt.cm.get_cmap('Greens_r', 15)   
        if neuron_model == 'RS_no_adapt':
            cmap = plt.cm.get_cmap('Blues_r', 15) 
       
        #fig = plt.figure('Mean error as function of alphas', figsize=(7,6))
        plt.plot(inh_vals, mean_error, '.-', color = cmap(alpha_idx), label=f'alpha = {alpha}')
        plt.legend(reverse=True)

    plt.title(f'{neuron_model} mean error distribution')
    plt.xlabel('Inhibitory input (Hz)')
    plt.ylabel('Mean error (Hz)')
    return fig 
    
# -------------------- #
def make_TF_gif(neuron_model, df_data, std_data, poly_params_2, params_SI, alpha, unique_inh, colors, gif_name, y_lim=None):

    import imageio.v2 as imageio
    from ntmf.transfer_function import res_2_func

    frames = []

    idx = 0
    for fixed_inh, c in zip(unique_inh, colors):
        fig, ax = plt.subplots()

        ax.set_title(f'Transfer function of {neuron_model} cell')
        ax.set_ylabel('Output rate (Hz)')
        ax.set_xlabel('Excitatory input (Hz)')

        tol = 1e-6
        mask = np.isclose(df_data['input_inh'], fixed_inh, atol=tol)

        inp_exc = df_data.loc[mask, 'input_exc'].to_numpy()
        out_rate = df_data.loc[mask, 'avg_f_out'].to_numpy()
        fit_rate = df_data.loc[mask, 'fit_rate']

        mean_error = res_2_func(poly_params_2,
                                data=df_data.loc[mask],
                                params=params_SI,
                                alpha=alpha)

        plt.errorbar(inp_exc, out_rate, std_data[idx], linestyle='None', color = c)
        ax.plot(inp_exc, out_rate, 'o', color=c, label=f'data (inh={fixed_inh} Hz)')
        ax.plot(inp_exc, fit_rate, 'kx', markersize=7, label='fit')

        ax.text(0.2, 0.95, f'mean error: {mean_error:.2f} Hz',
                transform=ax.transAxes, ha='center')

        ax.legend(loc='lower right')
        ax.set_ylim(y_lim if y_lim else (-5, 100))
        idx += 1
        
        # --- convert fig to image ---
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)  # to clean memory
        
    # --- write GIF ---
    gif = imageio.mimsave(gif_name, frames, fps=2.5, loop=0, palettesize=256)

    print(f"GIF saved as {gif_name}")

    return gif_name

# -------------------- #
def plot_residuals_TF_fitting(neuron_model, df_data):

    # Pivot into 2D matrix for plotting
    residual_map = df_data.pivot_table(
        index='input_inh',
        columns='input_exc',
        values='residual'
    )
    
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(
        residual_map,
        cmap='coolwarm', center=0,
        cbar_kws={'label': 'Residual (data - fit) [Hz]'},
        ax=ax
    )
    
    ax.set_title(f'Residuals between data and fit ({neuron_model})')
    ax.set_xlabel('Excitatory input (Hz)')
    ax.set_ylabel('Inhibitory input (Hz)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig
    
# -------------------- #
def plot_abs_residuals_TF_fitting(neuron_model, df_data):

    abs_residual_map = df_data.pivot_table(
        index='input_inh',
        columns='input_exc',
        values='abs_residual'
    )
    
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(
        abs_residual_map,
        cmap='magma',
        cbar_kws={'label': '|Residual| [Hz]'},
        ax=ax
    )
    
    ax.set_title(f'Absolute residuals between data and fit ({neuron_model})')
    ax.set_xlabel('Excitatory input (Hz)')
    ax.set_ylabel('Inhibitory input (Hz)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig

# -------------------- #
def plot_violin(neuron_model, poly_z, params_name):
    fig = plt.figure(figsize=(10,5))
    
    parts = plt.violinplot(poly_z,showmeans=True)

    color = color_palette[neuron_model]

    # The following code is used "to appy the color to the violins":
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.2)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        parts['cmeans'].set_alpha(0.4)
    for key in ['cbars', 'cmins', 'cmaxes']:
        if key in parts:
            parts[key].set_color('black')
            parts[key].set_alpha(0.4)
            parts[key].set_linewidth(1)    
    
    plt.axhline(0, linestyle="-", color="gray", alpha = 0.2)
    plt.ylim(-3,3)
    plt.xticks(range(1, len(params_name) + 1), params_name, rotation=45, ha='right')
    plt.xlabel("Polynomial parameters")
    plt.ylabel("Z-score")
    plt.title(f"Normalized variability of polynomial parameters for {neuron_model} transfer function")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_poly_zscore_heatmap(neuron_model, poly_z, params_name):
    
    fig = plt.figure(figsize=(7,5))
    plt.imshow(poly_z, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    plt.colorbar(label="Z-score")
    plt.xticks(range(0, len(params_name)), params_name, rotation=45, ha='right')
    plt.xlabel("Polynomial Coefficients")
    plt.ylabel("Model #")
    plt.title(f"Normalized polynomial parameters \n {neuron_model} transfer function")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_poly_correlation_error(neuron_model, corrs, params_name):


    fig = plt.figure(figsize=(7,5))
    plt.bar(range(len(corrs)), corrs, color = color_palette[neuron_model], alpha = 0.3, edgecolor = 'k')
    plt.axhline(0, linestyle="--", color="gray", alpha = 0.5)
    plt.xticks(range(0, len(params_name)), params_name, rotation=45, ha='right')
    plt.xlabel("Polynomial parameters")
    plt.ylabel("Correlation with mean_error")
    plt.title(f"Polynomial parameters correlation with mean_error \n {neuron_model}")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_corr_matrix(neuron_model, corr_matrix, params_name):

    fig = plt.figure(figsize=(7,5))
    plt.imshow(corr_matrix, vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(0, len(params_name)), params_name, rotation=45, ha='right')
    plt.yticks(range(0, len(params_name)), params_name, rotation=45, ha='right')
    plt.title(f"Correlation between polynomial parameters \n {neuron_model} transfer function")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_scree(neuron_model, expl_var):

    fig = plt.figure(figsize=(7,5))
    plt.plot(range(1, len(expl_var)+1), expl_var, '.-', color = color_palette[neuron_model], alpha = 0.5)
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree plot")
    plt.xticks(range(1,11))
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    return fig

# -------------------- #
def plot_pca_parameter_contributions(neuron_model, pc1, pc2):
    
    x = np.arange(len(pc1))
    width = 0.38
    
    fig = plt.figure(figsize=(7,5))
    
    plt.bar(x - width/2, pc1, width, label='PC1', color='#f1a340', edgecolor = 'k')
    plt.bar(x + width/2, pc2, width, label='PC2', color='#998ec3', edgecolor = 'k')
    
    plt.xticks(x, pc1.index, rotation=45)
    plt.ylabel("Loading weight")
    plt.title(f"Principal component {neuron_model} parameter contributions")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    return fig

# -------------------- #
def plot_pca_biplot(neuron_model, params_name, X_pca, loadings, expl_var):
    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.scatter(X_pca[:,0], X_pca[:,1],
               color=color_palette[neuron_model], alpha=0.7)
    
    # Arrows
    scale = 1
    for i in range(len(loadings)):
        ax.arrow(0, 0,
                 loadings[0,i]*scale,
                 loadings[1,i]*scale,
                 head_width=0.03,
                 color=color_palette[neuron_model],
                 alpha=0.7)
        ax.text(loadings[0,i]*scale,
                loadings[1,i]*scale,
                params_name[i])
    
    ax.set_xlabel(f"PC1 ({expl_var[0]*100:.1f}%)",
                  color='#f1a340', weight='bold')
    ax.set_ylabel(f"PC2 ({expl_var[1]*100:.1f}%)",
                  color='#998ec3', weight='bold')
    
    ax.set_aspect('equal', adjustable='box')
    
    ax.grid()
    ax.set_title(f"PCA biplot: \n {neuron_model} poly coefficients")
    plt.tight_layout()

    return fig



def plot_membrane_potential_fluctuations(data=None, mu_V=None, sig_V=None, tau_V=None):
    """Visualize membrane potential fluctuations as a 4-panel heatmap.

    Panels: mu_V, sig_V, tau_V, and output firing rate.
    """
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    inh_vals = sorted(data['input_inh'].unique())
    exc_vals = sorted(data['input_exc'].unique())

    n_inh = len(inh_vals)
    n_exc = len(exc_vals)

    out_freq = data['avg_f_out'].to_numpy().reshape(n_inh, n_exc)
    mu_V = mu_V.reshape(n_inh, n_exc)
    sig_V = sig_V.reshape(n_inh, n_exc)
    tau_V = tau_V.reshape(n_inh, n_exc)

    heatmap_kwargs = dict(
        cmap='viridis',
        xticklabels=exc_vals,
        yticklabels=inh_vals,
    )

    datasets = [
        (mu_V, 'Mean Membrane Potential [V]'),
        (sig_V, 'Membrane Potential Std [V]'),
        (tau_V, 'Autocorrelation Time [s]'),
        (out_freq, 'Output Frequency [Hz]'),
    ]

    for ax, (data_map, title) in zip(axs, datasets):
        sns.heatmap(data_map, ax=ax, **heatmap_kwargs)
        ax.set_title(title)
        ax.set_xlabel('Freq. exc [Hz]')
        ax.set_ylabel('Freq. inh [Hz]')

        # Reduce label density
        step_x = max(1, len(exc_vals) // 10)
        step_y = max(1, len(inh_vals) // 5)
        ax.set_xticks(np.arange(0.5, n_exc, step_x))
        ax.set_xticklabels(np.round(exc_vals[::step_x], 1), rotation=60)
        ax.set_yticks(np.arange(0.5, n_inh, step_y))
        ax.set_yticklabels(np.round(inh_vals[::step_y], 1), rotation=0)

    plt.tight_layout(h_pad=1.5)
    return fig



