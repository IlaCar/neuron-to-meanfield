import json
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

import brian2 as b2

# -------------------- #
color_palette = {"FS": '#f46d43',           # orange
                 "RS": '#225ea5',           # blue
                 "RS_no_adapt": '#41b6c4',  # teal
                 "SST": '#762a83',          # purple                 
                 "input": '#9ecae1',        # light blue
                 "GoC": "#2171b5",          # blue
                 "current": "#737373"       # grey
                }        

syn_colors = {
    "E": "#006837",     # green
    "I": "#a50026",     # red
    "Total": "#000000"  # black
}

_cmap_anchors = {
    'FS':          ['#fff5eb', '#fdae61', '#f46d43', '#c2410c'],  # orange
    'RS':          ['#f7fbff', '#6baed6', '#225ea5', '#08306b'],  # blue (adaptive)
    'RS_no_adapt': ['#f0fdfd', '#a6e1e6', '#41b6c4', '#0e6e78'],  # teal (non-adaptive)
}
neuron_cmaps = {
    k: LinearSegmentedColormap.from_list(f'{k}_seq', v)
    for k, v in _cmap_anchors.items()
}

_line_anchors = {
    'FS':          ['#fdae61', '#f46d43', '#c2410c'],  # orange
    'RS':          ['#6baed6', '#225ea5', '#08306b'],  # blue (adaptive)
    'RS_no_adapt': ['#a6e1e6', '#41b6c4', '#0e6e78'],  # teal (non-adaptive)
}
neuron_line_cmaps = {
    k: LinearSegmentedColormap.from_list(f'{k}_lines', v)
    for k, v in _line_anchors.items()
}

pc_colors = {
    "PC1": "#54278f",  # dark purple
    "PC2": "#9e9ac8",  # mid purple
    "PC3": "#cbc9e2",  # light purple
}

_ARROW_COLOR = "#525252"     # neutral grey, keeps neuron colour free for the scores
_DIVERGING_CMAP = "coolwarm"

# Fallbacks for neuron models without a dedicated colour (e.g. neuron_3_inputs, neuron_4_inputs)
_DEFAULT_NEURON_COLOR = "#525252"                             # neutral grey
_DEFAULT_TRACE_COLORS = ['#252525', '#525252', '#969696']     # dark -> light grey

def get_neuron_color(neuron_model):
    """Colour for *neuron_model*, falling back to neutral grey for unlisted models."""
    return color_palette.get(neuron_model, _DEFAULT_NEURON_COLOR)
    
_default_seq_cmap = LinearSegmentedColormap.from_list('default_seq', ['#f7f7f7', '#bdbdbd', '#737373', '#252525'])
_default_line_cmap = LinearSegmentedColormap.from_list('default_lines', ['#bdbdbd', '#737373', '#252525'])

def get_neuron_cmap(neuron_model):
    """Sequential colormap for *neuron_model* (grey for unlisted models)."""
    return neuron_cmaps.get(neuron_model, _default_seq_cmap)

def get_neuron_line_cmap(neuron_model):
    """Line-family colormap for *neuron_model* (grey for unlisted models)."""
    return neuron_line_cmaps.get(neuron_model, _default_line_cmap)
    
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
        ax[0].plot(pop1.t/b2.second, pop1.v[0] / b2.mV, color='#c2410c')
        ax[0].plot(pop1.t/b2.second, pop1.v[1] / b2.mV, color='#f46d43')
        ax[0].plot(pop1.t/b2.second, pop1.v[2] / b2.mV, color='#fdae61')
    else:
        ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[0], -50) / b2.mV, '--', color='#c2410c')
        ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[1], -50) / b2.mV, '--', color='#f46d43')
        ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[2], -50) / b2.mV, '--', color='#fdae61')  
    ax[0].set_title('Selected FS traces')

    if RS_adaptation == True:
        color_RS = color_palette['RS']
        col_0 = '#08306b'
        col_1 = '#225ea5'
        col_2 = '#6baed6'
    else:
        color_RS = color_palette['RS_no_adapt']
        col_0 = '#0e6e78'
        col_1 = '#41b6c4'
        col_2 = '#a6e1e6'
    

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
        colors = ['#c2410c', '#f46d43', '#fdae61']
    if neuron_model == 'RS':
        colors = ['#08306b', '#225ea5', '#6baed6']
    if neuron_model == 'RS_no_adapt':    
        colors = ['#0e6e78', '#41b6c4', '#a6e1e6']    
    if neuron_model not in ('FS', 'RS', 'RS_no_adapt'):
        colors = _DEFAULT_TRACE_COLORS
    fig, ax = plt.subplots(figsize=(10, 6), sharex=True)
    ax.plot(pop.t/b2.second, pop.v[0] / b2.mV, color=colors[0])
    ax.plot(pop.t/b2.second, get_pretty_voltage(pop.v[0], -50) / b2.mV, '--', color=colors[0])
    ax.plot(pop.t/b2.second, pop.v[1] / b2.mV, color=colors[0])
    ax.plot(pop.t/b2.second, get_pretty_voltage(pop.v[1], -50) / b2.mV, '--', color=colors[1])
    ax.plot(pop.t/b2.second, pop.v[2] / b2.mV, color=colors[0])
    ax.plot(pop.t/b2.second, get_pretty_voltage(pop.v[2], -50) / b2.mV, '--', color=colors[2])
    ax.set_title(f'Selected {neuron_model} traces')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Membrane potential (mV)')
    
    plt.tight_layout()
    
    return fig

# -------------------- #
def add_stim_spans(ax,
                   exc_intervals=None,
                   inh_intervals=None,
                   input_interval=None,
                   alpha=0.1,
                   zorder=0,
                   ):
    """
    Shade stimulus intervals as full-height spans behind a trace plot.
    x in data units (time); spans the whole y-axis automatically.
    """
    def _draw(intervals, color):
        if intervals is None:
            return
        for item in intervals:
            t0, t1 = item[0], item[1]   # ignores any pop label
            ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=zorder)

    _draw(exc_intervals, syn_colors['E'])
    _draw(inh_intervals, syn_colors['I'])
    _draw(input_interval, color_palette["input"])

    return ax
    
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

    # Strip any Brian2 units for downstream numpy operations
    spike_matrix_FS = np.asarray(spike_matrix_FS)
    spike_matrix_RS = np.asarray(spike_matrix_RS)
       
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
    
    # Strip any Brian2 units for downstream numpy operations
    spike_matrix_FS = np.asarray(spike_matrix_FS)
    spike_matrix_RS = np.asarray(spike_matrix_RS)
    
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
    ax.plot(pop.t / b2.second, pop.i, 'o', color=get_neuron_color(neuron_model), markersize=m_size)
   
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
    
    # Strip any Brian2 units for downstream numpy operations
    spike_matrix = np.asarray(spike_matrix)
        
    # Compute mean and std
    mean_rate = np.mean(spike_matrix, axis=0)
    std_rate = np.std(spike_matrix, axis=0)

    '''
    if neuron_model == 'FS':
        color = color_palette['FS']
    if neuron_model == 'RS':
        color = color_palette['RS']
    if neuron_model == 'RS_no_adapt':
        color = color_palette['RS_no_adapt']
    '''
    color = get_neuron_color(neuron_model)
    
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
def heatmap_InOut(exc_vals, inh_vals, data_array, neuron_type,
                  x_label='Freq Excitatory synapses (Hz)',
                  y_label='Freq Inhibitory synapses (Hz)',
                  title_suffix=''):
    fig = plt.figure(figsize=(10,6))
        
    plt.imshow(data_array, aspect='auto', origin='lower',
               extent=[exc_vals[0], exc_vals[-1], inh_vals[0], inh_vals[-1]],
               cmap=get_neuron_cmap(neuron_type))
    
    plt.colorbar(label='Output frequency (Hz)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{neuron_type} input–output relation{title_suffix}')
    
    return fig
    
# -------------------- #
def plot_InOut_relation(exc_vals, inh_vals, data_array, std_data_array, neuron_type,
                        x_label='Freq Excitatory synapses (Hz)',
                        legend_title='Freq Inh Syn (Hz)',
                        label_every=5,
                        title_suffix=''):
    fig = plt.figure(figsize=(10, 6))

    cmap = get_neuron_line_cmap(neuron_type)
    colors = cmap(np.linspace(0, 1, len(inh_vals)))  # light (low inh) -> dark (high inh)

    for i in range(len(inh_vals)):
        c = colors[i]
        label = f'{inh_vals[i]:g}' if i % label_every == 0 else None
        plt.plot(exc_vals, data_array[i], '-o', color=c, label=label)
        plt.errorbar(exc_vals, data_array[i], std_data_array[i], fmt='-o', color=c, alpha=0.5)

    plt.xlabel(x_label)
    plt.ylabel('Output frequency (Hz)')
    plt.title(f'{neuron_type} input-output relation{title_suffix}')
    plt.legend(title=legend_title)
    return fig

# -------------------- #
def plot_contours(exc_vals, inh_vals, data_array, neuron_type,
                  x_label='Freq Excitatory synapses (Hz)',
                  y_label='Freq Inh Syn (Hz)',
                  title_suffix=''):
    
    fig = plt.figure(figsize=(10,6))
       
    cs = plt.contourf(exc_vals, inh_vals, data_array, levels=20, cmap=get_neuron_cmap(neuron_type))

    plt.colorbar(cs, label='Output frequency (Hz)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{neuron_type} input–output contours{title_suffix}')
    
    return fig

# -------------------- #
def plot_gain(exc_vals, inh_vals, data_array, neuron_type,
              x_label='Freq Excitatory synapses (Hz)',
              y_label='Freq Inh Syn (Hz)',
              title_suffix=''):
    
    fig = plt.figure(figsize=(10,6))
        
    gain = np.gradient(data_array, exc_vals, axis=1)
    plt.imshow(
        gain,
        aspect='auto',
        origin='lower',
        extent=[exc_vals[0], exc_vals[-1], inh_vals[0], inh_vals[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Gain (Hz/Hz)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{neuron_type} gain map{title_suffix}')
    
    return fig

# -------------------- #
def mesh_3d(exc_vals, inh_vals, data_array, neuron_type,
            x_label='Freq Excitatory synapses (Hz)',
            y_label='Freq Inhibitory synapses (Hz)',
            title_suffix=''):
          
    X, Y = np.meshgrid(exc_vals, inh_vals)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data_array, cmap = get_neuron_cmap(neuron_type))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)   
    ax.set_zlabel('Output frequency (Hz)')
    ax.set_title(f'{neuron_type} input–output surface{title_suffix}')
    
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
    ax.plot(inp_exc, out_rate, 'o', color= get_neuron_color(neuron_model), label=f'{neuron_model} data')
    ax.plot(inp_exc, fit_rate, 'kx', markersize=7, alpha=0.3, label='fit')
    
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
    
        ax.plot(inp_exc, out_rate, 'o', color=c, alpha=0.25)
        ax.plot(inp_exc, fit_rate, 'x', color=c, markersize=7)
    ax.text(0.5, 0.95, f'mean error: {mean_error:.2f} Hz', transform=ax.transAxes, ha='center')    
    # colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=unique_inh.min(), vmax=unique_inh.max()))
    cbar = plt.colorbar(sm, ax=ax, alpha = 0.5)
    cbar.set_label('Inhibitory input (Hz)')
    
    return fig

# -------------------- #
def plots_TF_fitting(neuron_model, df_data, std_data, poly_params_2, params_SI, alpha, unique_inh, colors, y_lim = None, w_ad = 0.0):

    from ntmf.transfer_function import res_2_func
    distr_mean_error = np.zeros(len(unique_inh))
    w_ad_arr = np.asarray(w_ad)   # scalar, or per-row array aligned to df_data
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
        w_slice = float(w_ad_arr) if w_ad_arr.ndim == 0 else w_ad_arr[np.asarray(mask)]
        mean_error = res_2_func(poly_params_2, data=df_data.loc[mask], params=params_SI, alpha=alpha, w_ad=w_slice)

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
                              alpha=None, alpha_idx=None, n_alphas=None,
                              fig=None, ax=None, cmap_range=(0.0, 1.0)):

    """Plot the mean TF error against inhibitory input.

    If `alpha` is given, the curve is coloured by its rank in the alpha
    sequence (`alpha_idx` out of `n_alphas`) using the population-specific
    sequential colormap `neuron_line_cmaps[neuron_model]`, so that repeated
    calls onto the same figure produce a light-to-dark family of curves.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    elif ax is None:
        ax = fig.gca() if fig.axes else fig.add_subplot(111)
    else:
        fig = ax.figure

    if alpha is None:
        ax.plot(inh_vals, mean_error, '.-',
                color=color_palette[neuron_model])
    else:
        cmap = neuron_line_cmaps[neuron_model]
        if alpha_idx is None or n_alphas is None or n_alphas < 2:
            frac = 1.0
        else:
            lo, hi = cmap_range
            frac = lo + (hi - lo) * alpha_idx / (n_alphas - 1)
        ax.plot(inh_vals, mean_error, '.-',
                color=cmap(frac), label=rf'$\alpha$ = {alpha:.3g}')
        ax.legend(reverse=True, frameon=False)

    ax.set_title(f'{neuron_model} mean error distribution')
    ax.set_xlabel('Inhibitory input (Hz)')
    ax.set_ylabel('Mean error (Hz)')
    return fig
    
# -------------------- #
def make_TF_gif(neuron_model, df_data, std_data, poly_params_2, params_SI, alpha, unique_inh, colors, gif_name, y_lim=None, w_ad=0.0):
 
    import io
    import imageio.v2 as imageio
    from ntmf.transfer_function import res_2_func
 
    frames = []
    w_ad_arr = np.asarray(w_ad)   # scalar, or per-row array aligned to df_data
 
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
 
        w_slice = float(w_ad_arr) if w_ad_arr.ndim == 0 else w_ad_arr[np.asarray(mask)]
        mean_error = res_2_func(poly_params_2,
                                data=df_data.loc[mask],
                                params=params_SI,
                                alpha=alpha,
                                w_ad=w_slice)
 
        plt.errorbar(inp_exc, out_rate, std_data[idx], linestyle='None', color = c)
        ax.plot(inp_exc, out_rate, 'o', color=c, label=f'data (inh={fixed_inh} Hz)')
        ax.plot(inp_exc, fit_rate, 'kx', markersize=7, label='fit')
 
        ax.text(0.2, 0.95, f'mean error: {mean_error:.2f} Hz',
                transform=ax.transAxes, ha='center')
 
        ax.legend(loc='lower right')
        ax.set_ylim(y_lim if y_lim else (-5, 100))
        idx += 1
        
        # --- convert fig to image ---
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi)
        buf.seek(0)
        image = imageio.imread(buf)[:, :, :3]
        buf.close()
        frames.append(image)
 
        plt.close(fig)  # to clean memory
        
    # --- write GIF ---
    os.makedirs(os.path.dirname(gif_name), exist_ok=True)
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
# -------------------- #
def plot_poly_correlation_error(neuron_model, corr_df):
    """Correlation of each coefficient with the mean error, raw and controlling for alpha.

    The raw correlation is confounded whenever the coefficients are slaved to
    alpha, since both quantities are then functions of alpha. The partial
    correlation is shown alongside it.

    Parameters
    ----------
    neuron_model : str
    corr_df : pandas.DataFrame
        Output of `Analysis_helper.correlation_with_error`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.arange(len(corr_df))
    width = 0.38

    fig = plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, corr_df["r_raw"], width, label="raw",
            color=color_palette[neuron_model], alpha=0.45, edgecolor='k')
    plt.bar(x + width / 2, corr_df["r_partial"], width,
            label=r"controlling for $\alpha$",
            color=color_palette[neuron_model], alpha=0.95, edgecolor='k')

    plt.axhline(0, linestyle="-", color="k", linewidth=0.8)
    plt.ylim(-1.05, 1.05)
    plt.xticks(x, corr_df["params_name"], rotation=45, ha='right')
    plt.xlabel("Polynomial parameters")
    plt.ylabel("Correlation with mean error")
    plt.title(f"Polynomial parameters correlation with the mean error \n {neuron_model}")
    plt.legend()
    plt.tight_layout()

    return fig

# -------------------- #
def plot_corr_matrix(neuron_model, corr_matrix, params_name):
    """Correlation matrix of the fitted polynomial coefficients.

    Uses a diverging colormap centred on zero. The previous version left
    `cmap` unset, so the default sequential map was applied to a signed
    quantity and r = 0 was not rendered as a neutral colour.

    Parameters
    ----------
    neuron_model : str
    corr_matrix : ndarray of shape (n_params, n_params)
    params_name : list of str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(corr_matrix, cmap=_DIVERGING_CMAP, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(params_name)), params_name, rotation=45, ha='right')
    plt.yticks(range(len(params_name)), params_name)
    plt.title(f"Correlation between polynomial parameters \n {neuron_model} transfer function")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_scree(neuron_model, expl_var, n_models=None):
    """Scree plot of the PCA explained-variance ratio.
    
    Only min(n_models - 1, n_params) components can carry variance; the
    remaining eigenvalues are numerically zero and are not plotted.

    Parameters
    ----------
    neuron_model : str
    expl_var : ndarray of shape (n_components,)
    n_models : int, optional
        Number of models in the ensemble. If given, the axis is truncated
        at the maximum attainable rank, `n_models - 1`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_show = len(expl_var) if n_models is None else min(len(expl_var), n_models - 1)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(range(1, n_show + 1), expl_var[:n_show], '.-',
             color=color_palette[neuron_model], alpha=0.7, ms=10)
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title(f"Scree plot \n {neuron_model} transfer function "
              f"(rank $\\leq$ {n_show})")
    plt.xticks(range(1, n_show + 1))
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    return fig



# -------------------- #
def plot_pca_parameter_contributions(neuron_model, pc1, pc2):
    """Loading weights of the first two principal components.

    Recoloured with `pc_colors`: the previous orange (#f1a340) collided with
    the FS ramp, so in an FS figure the same hue denoted both the population
    and a principal component.

    Parameters
    ----------
    neuron_model : str
    pc1, pc2 : pandas.Series
        Loading vectors, indexed by coefficient name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.arange(len(pc1))
    width = 0.38

    fig = plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, pc1, width, label='PC1',
            color=pc_colors['PC1'], edgecolor='k', linewidth=0.5)
    plt.bar(x + width / 2, pc2, width, label='PC2',
            color=pc_colors['PC2'], edgecolor='k', linewidth=0.5)

    plt.xticks(x, pc1.index, rotation=45, ha='right')
    plt.ylabel("Loading weight")
    plt.title(f"Principal component parameter contributions \n {neuron_model}")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    return fig

# -------------------- #
def plot_pca_biplot(neuron_model, params_name, X_pca, loadings, expl_var,
                    color_by=None, color_by_label=None):
    """PCA biplot of the retained fits.

    Three changes with respect to the previous version. The loading arrows
    are rescaled to the extent of the scores, which is required as soon as
    the PCA is run on standardised data (unit-norm loadings are otherwise
    invisible next to scores of order 3). The arrows are drawn in neutral
    grey so that the neuron colour is reserved for the scores. The scores
    can optionally be coloured by a third variable, typically alpha.

    Parameters
    ----------
    neuron_model : str
    params_name : list of str
    X_pca : ndarray of shape (n_models, n_components)
        PCA scores.
    loadings : ndarray of shape (n_components, n_params)
        `pca.components_`.
    expl_var : ndarray
        `pca.explained_variance_ratio_`.
    color_by : ndarray of shape (n_models,), optional
        Values used to colour the scores, on the neuron's sequential ramp.
    color_by_label : str, optional
        Colorbar label for `color_by`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7.5, 7))

    if color_by is None:
        ax.scatter(X_pca[:, 0], X_pca[:, 1],
                   color=color_palette[neuron_model], s=90,
                   edgecolor='k', linewidth=0.5, zorder=3)
    else:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_by,
                        cmap=neuron_cmaps[neuron_model], s=110,
                        edgecolor='k', linewidth=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label=color_by_label, shrink=0.8)

    # Scale the arrows to the extent of the scores.
    scale = 0.8 * np.abs(X_pca[:, :2]).max() / np.abs(loadings[:2]).max()

    for i in range(loadings.shape[1]):
        ax.arrow(0, 0, loadings[0, i] * scale, loadings[1, i] * scale,
                 head_width=0.03 * scale, color=_ARROW_COLOR,
                 alpha=0.8, length_includes_head=True, zorder=2)
        ax.text(loadings[0, i] * scale * 1.08, loadings[1, i] * scale * 1.08,
                params_name[i], fontsize=8, color=_ARROW_COLOR, ha='center')

    ax.set_xlabel(f"PC1 ({expl_var[0] * 100:.1f}%)",
                  color=pc_colors['PC1'], weight='bold')
    ax.set_ylabel(f"PC2 ({expl_var[1] * 100:.1f}%)",
                  color=pc_colors['PC2'], weight='bold')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)
    ax.set_title(f"PCA biplot: \n {neuron_model} poly coefficients")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_coefficients_vs_alpha(neuron_model, alphas, poly_z, params_name):
    """Z-scored polynomial coefficients as a function of the fitted alpha.

    If the coefficients trace smooth curves in alpha rather than scattering,
    the retained ensemble samples a one-dimensional degenerate valley and
    the coefficients are not independent degrees of freedom.

    Parameters
    ----------
    neuron_model : str
    alphas : ndarray of shape (n_models,)
    poly_z : ndarray of shape (n_models, n_params)
    params_name : list of str

    Returns
    -------
    matplotlib.figure.Figure
    """
    order = np.argsort(alphas)
    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(8, 5))
    for i, name in enumerate(params_name):
        plt.plot(alphas[order], poly_z[order, i], 'o-', ms=4, lw=1,
                 alpha=0.8, color=cmap(i % 10), label=name)

    plt.axhline(0, color='gray', lw=0.8, alpha=0.4)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Coefficient (Z-score)")
    plt.title(r"Polynomial coefficients against the fitted $\alpha$"
              + f"\n {neuron_model} transfer function")
    plt.legend(fontsize=7, ncol=2, loc='best')
    plt.tight_layout()

    return fig

# -------------------- #
def plot_error_vs_alpha(neuron_model, alphas, errors):
    """Mean fitting error against the fitted alpha.

    Parameters
    ----------
    neuron_model : str
    alphas : ndarray of shape (n_models,)
    errors : ndarray of shape (n_models,)

    Returns
    -------
    matplotlib.figure.Figure
    """
    order = np.argsort(alphas)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas[order], errors[order], '-', color='0.6', zorder=1)
    sc = ax.scatter(alphas, errors, c=alphas, cmap=neuron_cmaps[neuron_model],
                    s=110, edgecolor='k', linewidth=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label=r"$\alpha$", shrink=0.8)

    best = np.argmin(errors)
    ax.annotate(rf"best: $\alpha$ = {alphas[best]:.3f}",
                (alphas[best], errors[best]),
                xytext=(10, 10), textcoords='offset points', fontsize=9)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Mean error")
    ax.set_title(r"Fitting error against $\alpha$"
                 + f"\n {neuron_model} transfer function")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_alpha_trend(neuron_model, trend_df):
    """Correlation of each coefficient with alpha, and the variance alpha explains.

    Parameters
    ----------
    neuron_model : str
    trend_df : pandas.DataFrame
        Output of `Analysis_helper.alpha_trend`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    r2_col = [c for c in trend_df.columns if c.startswith("R2_")][0]
    x = np.arange(len(trend_df))

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)

    axs[0].bar(x, trend_df["r_alpha"], color=color_palette[neuron_model],
               alpha=0.5, edgecolor='k')
    axs[0].axhline(0, color='k', lw=0.8)
    for s in (-0.8, 0.8):
        axs[0].axhline(s, ls='--', color='gray', lw=0.8, alpha=0.6)
    axs[0].set_ylim(-1.05, 1.05)
    axs[0].set_ylabel(r"Pearson $r$ with $\alpha$")
    axs[0].set_title(r"Dependence of the polynomial coefficients on $\alpha$"
                     + f"\n {neuron_model} transfer function")

    axs[1].bar(x, trend_df[r2_col], color=color_palette[neuron_model],
               alpha=0.5, edgecolor='k')
    axs[1].axhline(1.0, ls='--', color='gray', lw=0.8, alpha=0.6)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel(r"$R^2$ of the trend in $\alpha$")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(trend_df["params_name"], rotation=45, ha='right')
    axs[1].set_xlabel("Polynomial parameters")

    plt.tight_layout()

    return fig

    
# -------------------- #
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

#-------------------------------------------------------------------
def plotting_state_monitor_variables(mon = None, neuron_model = None, pretty_plot = False, V_min=None,
                                     syn = None):

    fig, ax = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

    ax[0].plot(mon.t / b2.ms, mon.V[0] / b2.mV, color=color_palette[neuron_model])
    ax[0].set_ylabel('Membrane potential (mV)')
    if syn == None:
        ax[0].set_title(f'{neuron_model.split("_")[0]} membrane potential response to current injection')
    else:
        ax[0].set_title(f'{neuron_model.split("_")[0]} membrane potential response to synaptic activation')
    
    ax[1].plot(mon.t / b2.ms, mon.Is[0] / b2.nA, color=color_palette['current'])
    ax[1].set_ylabel('Current (nA)')
    ax[1].set_title('Injected current (nA)')
    
    ax[2].plot(mon.t / b2.ms, mon.Ia[0] / b2.nA, color=color_palette['current'])
    ax[2].set_ylabel('Current (nA)')
    ax[2].set_title('Adaptive current (nA)')
    
    ax[3].plot(mon.t / b2.ms, mon.Id[0] / b2.nA, color=color_palette['current'])
    ax[3].set_ylabel('Current (nA)')
    ax[3].set_title('Depolarizing spike-triggered current (nA)')
    
    ax[4].plot(mon.t / b2.ms, mon.Ie[0] / b2.nA, color=color_palette['current'])
    ax[4].set_ylabel('Current (nA)')
    ax[4].set_title('Endogenous current (nA)')
    ax[4].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    #ax[0].legend()

    return ax, fig

# -------------------- #
def plot_violin(neuron_model, poly_z, params_name):
    """Normalised spread of the fitted polynomial coefficients.

    The individual models are overlaid as points: with an ensemble of ~10
    models the violin outline is a kernel density estimate of ten samples
    and must not be read as a distribution on its own.

    Parameters
    ----------
    neuron_model : str
        One of 'FS', 'RS', 'RS_no_adapt'.
    poly_z : ndarray of shape (n_models, n_params)
        Z-scored polynomial coefficients.
    params_name : list of str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 5))
    color = color_palette[neuron_model]

    parts = plt.violinplot(poly_z, showmeans=True)
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

    # Overlay the individual models, jittered horizontally.
    n_models = poly_z.shape[0]
    rng = np.random.default_rng(0)
    for i in range(poly_z.shape[1]):
        x = (i + 1) + rng.uniform(-0.08, 0.08, n_models)
        plt.scatter(x, poly_z[:, i], s=14, color=color,
                    edgecolor='k', linewidth=0.4, zorder=3)

    plt.axhline(0, linestyle="-", color="gray", alpha=0.2)
    plt.ylim(-3, 3)
    plt.xticks(range(1, len(params_name) + 1), params_name, rotation=45, ha='right')
    plt.xlabel("Polynomial parameters")
    plt.ylabel("Z-score")
    plt.title(f"Normalized variability of polynomial parameters "
              f"across the {n_models} retained {neuron_model} fits")
    plt.tight_layout()

    return fig

# -------------------- #
def plot_TF_ensemble_envelope(neuron_model, df_data, fit_cols, unique_inh,
                              exc_col='input_exc', inh_col='input_inh',
                              data_col='avg_f_out', std_col='std_f_out'):
    """Spread of the retained transfer functions against the simulated data.

    Left: the envelope spanned by the retained fits over the input grid,
    with the simulated firing rate and its standard deviation. Right: the
    ensemble spread of the predictions against the simulated standard
    deviation, point by point. If the ensemble spread falls below the
    simulated dispersion, the retained fits are not distinguishable by the
    data and the spread of the coefficients carries no information.

    Parameters
    ----------
    neuron_model : str
    df_data : pandas.DataFrame
        Simulated transfer-function data, with one column per retained fit.
    fit_cols : list of str
        Names of the columns holding the predicted rates.
    unique_inh : ndarray
        Sorted inhibitory input rates to plot.
    exc_col, inh_col, data_col, std_col : str, optional
        Column names in `df_data`.

    Returns
    -------
    matplotlib.figure.Figure
    ensemble_std : ndarray
        Per-grid-point standard deviation of the predictions across fits.
    """
    cmap = neuron_line_cmaps[neuron_model]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    preds_all, sim_std_all = [], []

    for k, inh in enumerate(unique_inh):
        sub = df_data[df_data[inh_col] == inh].sort_values(exc_col)
        color = cmap(k / max(len(unique_inh) - 1, 1))

        preds = sub[fit_cols].to_numpy()          # (n_exc, n_fits)
        preds_all.append(preds)
        sim_std_all.append(sub[std_col].to_numpy())

        axs[0].fill_between(sub[exc_col], preds.min(axis=1), preds.max(axis=1),
                            color=color, alpha=0.35, linewidth=0)
        axs[0].errorbar(sub[exc_col], sub[data_col], yerr=sub[std_col],
                        fmt='o', ms=3.5, color=color, ecolor=color,
                        elinewidth=0.8, capsize=1.5, zorder=3,
                        label=rf"$\nu_i$ = {inh:.1f} Hz")

    axs[0].set_xlabel(r"$\nu_e$ (Hz)")
    axs[0].set_ylabel(r"$\nu_{out}$ (Hz)")
    axs[0].set_title(f"Envelope of the {len(fit_cols)} retained fits \n {neuron_model}")
    axs[0].legend(fontsize=7, ncol=2)

    preds_all = np.vstack(preds_all)
    sim_std_all = np.concatenate(sim_std_all)
    ensemble_std = preds_all.std(axis=1, ddof=1)

    lim = max(np.nanmax(ensemble_std), np.nanmax(sim_std_all)) * 1.05
    axs[1].scatter(sim_std_all, ensemble_std, s=25,
                   color=color_palette[neuron_model], alpha=0.6, edgecolor='k',
                   linewidth=0.3)
    axs[1].plot([0, lim], [0, lim], '--', color='0.4', lw=1,
                label="identity")
    axs[1].set_xlim(0, lim)
    axs[1].set_ylim(0, lim)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlabel(r"simulated std of $\nu_{out}$ (Hz)")
    axs[1].set_ylabel(r"ensemble std across fits (Hz)")
    axs[1].set_title("Are the retained fits distinguishable? \n"
                     "points below the identity: no")
    axs[1].legend()

    plt.tight_layout()

    return fig, ensemble_std

# -------------------- #
def plot_TF_error_map(neuron_model, df_data, fit_col, rel_floor=1.0,
                      exc_col='input_exc', inh_col='input_inh', data_col='avg_f_out'):
    """Where on the input grid does the fitted transfer function fail?

    Left: absolute error, in Hz. Right: error relative to the simulated rate.
    Together these bound the domain over which the fitted transfer function can
    be trusted, which is the quantity a pipeline user needs before deploying it
    in a network.

    Parameters
    ----------
    neuron_model : str
    df_data : pandas.DataFrame
        Simulated data with the predicted rate in `fit_col`.
    fit_col : str
        Column holding the predicted rate, typically that of the best fit.
    rel_floor : float, optional
        Rates below this value (Hz) are masked in the relative-error panel,
        where the ratio is dominated by the denominator. Default 1.0.
    exc_col, inh_col, data_col : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df_data.copy()
    df['_abs_err'] = np.abs(df[fit_col] - df[data_col])
    df['_rel_err'] = np.where(df[data_col] >= rel_floor,
                              df['_abs_err'] / df[data_col], np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, ttl, unit in [
            (axs[0], '_abs_err', "Absolute error", "Hz"),
            (axs[1], '_rel_err', f"Relative error (rates below {rel_floor:g} Hz masked)", "")]:
        grid = df.pivot(index=inh_col, columns=exc_col, values=col)
        im = ax.pcolormesh(grid.columns.to_numpy(), grid.index.to_numpy(),
                           grid.to_numpy(), cmap=neuron_cmaps[neuron_model],
                           shading='nearest')
        plt.colorbar(im, ax=ax, label=f"|fit - data| {unit}".strip())
        ax.set_xlabel(r"$\nu_e$ (Hz)")
        ax.set_ylabel(r"$\nu_i$ (Hz)")
        ax.set_title(f"{ttl}\n{neuron_model}, {fit_col}")

    plt.tight_layout()

    return fig

# -------------------- #
def plot_error_by_inhibition(neuron_model, df_data, fit_col,
                             exc_col='input_exc', inh_col='input_inh',
                             data_col='avg_f_out'):
    """Contribution of each inhibitory input level to the overall fitting error.

    An objective defined as a mean absolute error in Hz is dominated by the
    region where the output rate is largest. This shows how much of the reported
    error comes from each inhibitory level, and whether a few columns of the grid
    are setting the ranking of the retained fits.

    Parameters
    ----------
    neuron_model : str
    df_data : pandas.DataFrame
    fit_col : str
    exc_col, inh_col, data_col : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    contribution : pandas.DataFrame
        Per-inhibition mean error, mean simulated rate, and share of the total.
    """
    df = df_data.copy()
    df['_abs_err'] = np.abs(df[fit_col] - df[data_col])

    g = df.groupby(inh_col).agg(mean_err=('_abs_err', 'mean'),
                                mean_rate=(data_col, 'mean'))
    g['share_of_total'] = g['mean_err'] / g['mean_err'].sum()
    g = g.reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))

    axs[0].bar(g[inh_col], g['mean_err'],
               width=0.8 * np.median(np.diff(np.sort(g[inh_col]))),
               color=color_palette[neuron_model], alpha=0.6, edgecolor='k',
               linewidth=0.4)
    axs[0].axhline(g['mean_err'].mean(), ls='--', color='0.4', lw=1,
                   label='grid mean')
    axs[0].set_xlabel(r"$\nu_i$ (Hz)")
    axs[0].set_ylabel("Mean absolute error (Hz)")
    axs[0].set_title(f"Error by inhibitory level\n{neuron_model}, {fit_col}")
    axs[0].legend(fontsize=8)

    axs[1].scatter(g['mean_rate'], g['mean_err'], s=60,
                   color=color_palette[neuron_model], edgecolor='k', linewidth=0.4)
    axs[1].set_xlabel(r"Mean simulated $\nu_{out}$ (Hz) at that $\nu_i$")
    axs[1].set_ylabel("Mean absolute error (Hz)")
    axs[1].set_title("Is the error simply tracking the rate?\n"
                     "a rising trend means the objective is rate-weighted")

    plt.tight_layout()

    return fig, g

# -------------------- #
def plot_pc_loadings_comparison(loadings_dict, params_name, pc_index=0,
                                expl_var_dict=None, reference_name=None):
    """Compare one principal component's loadings across neuron models.

    Two corrections with respect to a direct comparison. The sign of a principal
    component is arbitrary, so the loadings must be aligned before they can be
    compared; pass components already passed through
    `Analysis_helper.align_pca_signs`. And the bars are offset by model, so that
    three models occupy three positions rather than two.

    Parameters
    ----------
    loadings_dict : dict of {str: ndarray of shape (n_components, n_params)}
        Sign-aligned components per neuron model.
    params_name : list of str
    pc_index : int, optional
        Zero-based index of the component to compare. Default 0 (PC1).
    expl_var_dict : dict of {str: ndarray}, optional
        Explained-variance ratios, added to the legend when given.
    reference_name : str, optional
        Population the components were oriented to, named in the title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(loadings_dict.keys())
    n_m = len(models)
    x = np.arange(len(params_name))
    width = 0.8 / n_m

    fig = plt.figure(figsize=(10, 5))
    for i, m in enumerate(models):
        offset = (i - (n_m - 1) / 2) * width
        lbl = m
        if expl_var_dict is not None:
            lbl += f" ({expl_var_dict[m][pc_index] * 100:.1f}%)"
        plt.bar(x + offset, loadings_dict[m][pc_index], width, label=lbl,
                color=color_palette[m], alpha=0.8, edgecolor='k', linewidth=0.5)

    plt.xticks(x, params_name, rotation=45, ha='right')
    plt.axhline(0, color='k', linewidth=0.8)
    plt.ylabel("Loading weight")
    plt.title(f"PC{pc_index + 1} loadings across neuron models \n"
              f"(oriented to {reference_name})" if reference_name
              else f"PC{pc_index + 1} loadings across neuron models")
    plt.legend()
    plt.tight_layout()

    return fig


# -------------------- #
def plot_pooled_pca_scores(scores_dict, pca, params_name, labels_dict=None):
    """Neuron models projected onto a single, shared PCA basis.

    Scores from independently fitted decompositions live in different bases and
    cannot be overlaid. This plots the scores returned by
    `Analysis_helper.pooled_pca`, which share one basis by construction.

    Parameters
    ----------
    scores_dict : dict of {str: ndarray of shape (n_models, >=2)}
        Scores in the common basis.
    pca : sklearn.decomposition.PCA
        The pooled decomposition, used for the loading arrows and axis labels.
    params_name : list of str
    labels_dict : dict of {str: ndarray}, optional
        Cluster labels per neuron model; when given, clusters are drawn with
        different markers.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    markers = ['o', 's', '^', 'D', 'v']

    all_scores = np.vstack([s[:, :2] for s in scores_dict.values()])

    for m, S in scores_dict.items():
        if labels_dict is None or m not in labels_dict:
            ax.scatter(S[:, 0], S[:, 1], color=color_palette[m], s=95,
                       edgecolor='k', linewidth=0.5, label=m, zorder=3)
        else:
            for j in np.unique(labels_dict[m]):
                k = labels_dict[m] == j
                ax.scatter(S[k, 0], S[k, 1], color=color_palette[m], s=95,
                           marker=markers[j % len(markers)],
                           edgecolor='k', linewidth=0.5,
                           label=f"{m} cluster {j}", zorder=3)

    scale = 0.75 * np.abs(all_scores).max() / np.abs(pca.components_[:2]).max()
    for i in range(pca.components_.shape[1]):
        ax.arrow(0, 0, pca.components_[0, i] * scale, pca.components_[1, i] * scale,
                 head_width=0.02 * scale, color=_ARROW_COLOR, alpha=0.7,
                 length_includes_head=True, zorder=2)
        ax.text(pca.components_[0, i] * scale * 1.08,
                pca.components_[1, i] * scale * 1.08,
                params_name[i], fontsize=8, color=_ARROW_COLOR, ha='center')

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                  color=pc_colors['PC1'], weight='bold')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
                  color=pc_colors['PC2'], weight='bold')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)
    ax.set_title("Neuron models in a common PCA basis\n"
                 "(one decomposition fitted on the pooled ensembles)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    return fig


# -------------------- #
def plot_cv_comparison(stats_dict, params_name):
    """Compare the stiff-to-sloppy ranking across neuron models.

    Parameters
    ----------
    stats_dict : dict of {str: pandas.DataFrame}
        Output of `Analysis_helper.coefficient_stats` per neuron model.
    params_name : list of str

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(stats_dict.keys())
    n_m = len(models)
    x = np.arange(len(params_name))
    width = 0.8 / n_m

    fig = plt.figure(figsize=(10, 5))
    for i, m in enumerate(models):
        s = stats_dict[m].set_index("params_name").loc[params_name]
        offset = (i - (n_m - 1) / 2) * width
        plt.bar(x + offset, 100 * s["cv"], width, label=m,
                color=color_palette[m], alpha=0.8, edgecolor='k', linewidth=0.5)

    plt.axhline(100, ls='--', color='gray', lw=0.8,
                label='CV = 100% (std equals the mean)')
    plt.yscale('log')
    plt.xticks(x, params_name, rotation=45, ha='right')
    plt.ylabel("Coefficient of variation (%)")
    plt.title("Stiff and sloppy coefficients across neuron models\n"
              "low CV: constrained by the data; high CV: unconstrained")
    plt.legend(fontsize=8)
    plt.tight_layout()

    return fig



# -------------------- #
def plot_clusters(neuron_model, alphas, errors, labels, X_pca, diagnostics=None,
                  expl_var=None, panels=('pca', 'alpha_error', 'silhouette'),
                  annotate_counts=True):
    """Detected solution clusters among the retained fits.

    A search may converge to several distinct solutions of comparable quality
    rather than to one solution sampled repeatedly. Where it does, the retained
    fits are not independent draws and the effective sample size is the number
    of clusters.

    Panels can be selected and ordered through `panels`, so that the same
    function serves the diagnostic notebook (all three) and a figure panel
    (`panels=('pca',)`).

    The silhouette panel is annotated with the shape of the curve, which is more
    informative than its value at the selected count: a peak followed by a
    decline indicates genuine groups; a curve still rising at the largest count
    tested indicates repeated solutions; a curve flat near the threshold
    indicates outliers rather than structure.

    Overlapping markers are annotated with the number of fits stacked at that
    position, since repeated solutions are otherwise invisible.

    Parameters
    ----------
    neuron_model : str
    alphas, errors : ndarray of shape (n_models,)
    labels : ndarray of shape (n_models,)
        Cluster index per fit, from `Analysis_helper.detect_clusters`.
    X_pca : ndarray of shape (n_models, >=2)
        PCA scores of this population, in its own basis.
    diagnostics : pandas.DataFrame, optional
        Silhouette table from `detect_clusters`. Required for the silhouette panel.
    expl_var : ndarray, optional
        `pca.explained_variance_ratio_`, added to the PCA axis labels.
    panels : tuple of str, optional
        Any of 'pca', 'alpha_error', 'silhouette', in the order to be drawn.
    annotate_counts : bool, optional
        Annotate stacked markers with their multiplicity. Default True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    uniq = np.unique(labels)
    cmap = neuron_line_cmaps[neuron_model]
    ccolors = {j: cmap(i / max(len(uniq) - 1, 1)) for i, j in enumerate(uniq)}
    markers = ['o', 's', '^', 'D', 'v']

    panels = tuple(p for p in panels if p != 'silhouette' or diagnostics is not None)
    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5),
                            squeeze=False)
    axs = axs[0]

    def _scatter(ax, xs, ys):
        for j in uniq:
            m = labels == j
            n_j = int(m.sum())
            tag = " [singleton]" if n_j == 1 else ""
            ax.scatter(xs[m], ys[m], color=ccolors[j], s=95,
                       marker=markers[j % len(markers)], edgecolor='k',
                       linewidth=0.5, zorder=3,
                       label=f"cluster {j} (n = {n_j}){tag}")
        if annotate_counts:
            pts = np.column_stack([xs, ys])
            span = np.ptp(pts, axis=0)
            span[span == 0] = 1.0
            _, inv, counts = np.unique(np.round(pts / span, 3), axis=0,
                                       return_inverse=True, return_counts=True)
            for j, c in enumerate(counts):
                if c > 1:
                    k = np.where(inv == j)[0][0]
                    ax.annotate(f'x{c}', (xs[k], ys[k]), fontsize=9, weight='bold',
                                xytext=(7, 5), textcoords='offset points', zorder=4)
        ax.legend(fontsize=8)

    for ax, kind in zip(axs, panels):
        if kind == 'pca':
            _scatter(ax, X_pca[:, 0], X_pca[:, 1])
            xl, yl = "PC1", "PC2"
            if expl_var is not None:
                xl += f" ({expl_var[0] * 100:.1f}%)"
                yl += f" ({expl_var[1] * 100:.1f}%)"
            ax.set_xlabel(xl, color=pc_colors['PC1'], weight='bold')
            ax.set_ylabel(yl, color=pc_colors['PC2'], weight='bold')
            ax.axhline(0, color='0.85', lw=0.8, zorder=0)
            ax.axvline(0, color='0.85', lw=0.8, zorder=0)
            ax.set_title(f"Clusters in the PCA plane\n{neuron_model}")

        elif kind == 'alpha_error':
            _scatter(ax, alphas, errors)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel("Mean error")
            ax.set_title(r"Clusters in the $\alpha$-error plane" f"\n{neuron_model}")

        elif kind == 'silhouette':
            s = diagnostics["silhouette"].to_numpy()
            k = diagnostics["n_clusters"].to_numpy()
            ax.plot(k, s, 'o-', color=color_palette[neuron_model], ms=8)
            ax.axhline(0.55, ls='--', color='gray', lw=0.8, label='decision threshold')

            ax.set_xticks(k)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Silhouette")
            ax.set_title("Cluster-count selection\n(read the shape, not the value)")
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()

    return fig


# -------------------- #
def plot_silhouette_comparison(diagnostics_dict, labels_dict=None):
    """Cluster-count selection for every population, on shared axes.

    Parameters
    ----------
    diagnostics_dict : dict of {str: pandas.DataFrame}
        Silhouette tables from `Analysis_helper.detect_clusters`.
    labels_dict : dict of {str: ndarray}, optional
        Cluster labels, used to annotate the legend with the selected count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(5.6, 4.6))

    for m, d in diagnostics_dict.items():
        lbl = m
        if labels_dict is not None and m in labels_dict:
            k = len(np.unique(labels_dict[m]))
            lbl += f" ({k} cluster{'s' if k > 1 else ''})"
        plt.plot(d["n_clusters"], d["silhouette"], 'o-',
                 color=color_palette[m], ms=7, label=lbl)

    plt.axhline(0.5, ls='--', color='gray', lw=0.8, label='decision threshold')
    plt.xticks(list(diagnostics_dict.values())[0]["n_clusters"])
    plt.ylim(0, 1)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette")
    plt.title("Cluster-count selection")
    plt.legend(fontsize=8)
    plt.tight_layout()

    return fig


# -------------------- #
def plot_pc1_similarity(loadings_dict, n_params, pc_index=0, n_null=200000, seed=0):
    """Alignment of the dominant degenerate direction between populations.

    The cosine of the angle between the first principal components of two
    populations measures whether their fits are degenerate along the same
    direction. It is compared against the null of two unrelated directions:
    for uniformly random unit vectors in R^{n_params}, the cosine has density
    proportional to (1 - t^2)^((n_params - 3) / 2), which for n_params = 10 puts
    95% of the mass below |cos| = 0.60. A pair falling inside that band is not
    distinguishable from two unrelated directions, whatever its nominal value.

    Parameters
    ----------
    loadings_dict : dict of {str: ndarray of shape (n_components, n_params)}
        Sign-aligned components per population, from `align_pca_signs`.
    n_params : int
        Dimension of the coefficient space, which sets the null.
    pc_index : int, optional
        Zero-based index of the component compared. Default 0.
    n_null : int, optional
        Random pairs drawn to characterise the null. Default 200000.
    seed : int, optional

    Returns
    -------
    matplotlib.figure.Figure
    table : pandas.DataFrame
        Cosine, angle and null-tail probability for each pair.
    """
    from scipy import integrate

    models = list(loadings_dict.keys())
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]

    def _dens(t):
        return (1 - t ** 2) ** ((n_params - 3) / 2)
    Z, _ = integrate.quad(_dens, -1, 1)

    rows = []
    for a, b in pairs:
        c = float(np.dot(loadings_dict[a][pc_index], loadings_dict[b][pc_index]))
        tail, _ = integrate.quad(_dens, abs(c), 1)
        rows.append([f"{a}\n{b}", a, b, c, np.degrees(np.arccos(np.clip(c, -1, 1))),
                     2 * tail / Z])
    table = pd.DataFrame(rows, columns=["pair", "model_a", "model_b",
                                        "cosine", "angle_deg", "p_null"])

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_null, 2, n_params))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    null = np.abs(np.einsum('ij,ij->i', v[:, 0], v[:, 1]))
    q95 = float(np.percentile(null, 95))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.axhspan(0, q95, color='0.85', zorder=0,
               label=f'unrelated directions (95% of the null, $n_p$ = {n_params})')
    ax.axhline(float(null.mean()), ls=':', color='0.45', lw=1, zorder=1,
               label=f'null mean ({null.mean():.2f})')

    x = np.arange(len(table))
    for i, r in table.iterrows():
        inside = abs(r.cosine) <= q95
        ax.bar(i, abs(r.cosine), 0.55, zorder=2,
               color='0.62' if inside else color_palette[r.model_a],
               edgecolor='k', linewidth=0.6, hatch='//' if inside else None)
        ax.annotate(f"{abs(r.cosine):.3f}\n{r.angle_deg:.0f}"u"\N{DEGREE SIGN}",
                    (i, abs(r.cosine)), ha='center', va='bottom', fontsize=9,
                    xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(table["pair"], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"|cos| between PC{pc_index + 1} directions")
    ax.set_title("Is the dominant degenerate direction shared?\n"
                 "hatched: inside the null, i.e. not distinguishable "
                 "from unrelated")
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()

    return fig, table

# -------------------- #
def tf_grid_from_df(df_data, exc_col='input_exc', inh_col='input_inh',
                    value_col='avg_f_out'):
    """Reshape a long transfer-function table onto the (nu_i, nu_e) grid.

    `plot_surface` needs a 2D array, so the long table has to be pivoted
    first. Duplicated grid points are rejected rather than averaged, and
    missing grid points are reported, since either would silently deform the
    surface.

    Parameters
    ----------
    df_data : pandas.DataFrame
    exc_col, inh_col, value_col : str, optional
        Column names in `df_data`.

    Returns
    -------
    exc_vals : ndarray, shape (n_exc,)
    inh_vals : ndarray, shape (n_inh,)
    values   : ndarray, shape (n_inh, n_exc), matching np.meshgrid(exc, inh)
    """
    dup = df_data.duplicated(subset=[inh_col, exc_col]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated ({inh_col}, {exc_col}) rows; "
                         "the grid is not uniquely defined")

    piv = df_data.pivot(index=inh_col, columns=exc_col, values=value_col)
    values = piv.to_numpy(dtype=float)
    if np.isnan(values).any():
        print(f"[tf_grid_from_df] warning: {int(np.isnan(values).sum())} "
              f"missing grid points in '{value_col}'; "
              "the surface will have holes")

    return piv.columns.to_numpy(), piv.index.to_numpy(), values

# -------------------- #
def panel_grid_map(neuron_model, m, ax, value_col='avg_f_out', diverging=False,
                   vmax=None, cbar=True, cbar_label=None, title=None):
    '''Heatmap of one column of the transfer-function table over the input grid.

    Parameters
    ----------
    value_col : str
        Column to map. 'avg_f_out' for the simulated rate, 'residual_pct' for the
        residual as a percentage of the dynamic range.
    diverging : bool
        Centre the colour scale on zero and use a diverging colormap. Required for
        residuals, wrong for rates.
    vmax : float, optional
        Upper limit of the colour scale, shared across cell types when given. Defaults
        to the symmetric maximum of this cell type alone.
    '''
    exc_vals, inh_vals, grid = tf_grid_from_df(m['df'], value_col=value_col)

    if diverging:
        v = vmax if vmax is not None else np.nanmax(np.abs(grid))
        pcm = ax.pcolormesh(exc_vals, inh_vals, grid, cmap='PRGn',
                            vmin=-v, vmax=v, shading='auto')
    else:
        pcm = ax.pcolormesh(exc_vals, inh_vals, grid,
                            cmap=neuron_cmaps[neuron_model],
                            vmin=0, vmax=vmax, shading='auto')

    ax.set_xlabel(r'$\nu_e$ (Hz)')
    ax.set_ylabel(r'$\nu_i$ (Hz)')
    if title is not None:
        ax.set_title(title)
    if cbar:
        cb = ax.get_figure().colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cb.set_label(cbar_label)
    return pcm

# -------------------- #
def panel_TF_slices(neuron_model, m, ax, n_slices=5, envelope=True,
                    legend=True, title=None):
    '''Output rate against excitatory drive, one curve per inhibitory level.

    The shaded band spans the retained fits, so the width of the band is the spread of
    the ensemble and the offset from the markers is the error of the best fit.
    '''
    df, fit_cols = m['df'], m['fit_cols']
    inh_all = m['inh_vals']
    idx = np.unique(np.linspace(0, len(inh_all) - 1, n_slices).astype(int))
    cmap = neuron_line_cmaps[neuron_model]

    for j in idx:
        inh = inh_all[j]
        sub = df[df['input_inh'] == inh].sort_values('input_exc')
        color = cmap(j / max(len(inh_all) - 1, 1))

        if envelope:
            preds = sub[fit_cols].to_numpy()
            ax.fill_between(sub['input_exc'], preds.min(axis=1), preds.max(axis=1),
                            color=color, alpha=0.30, linewidth=0)
        ax.errorbar(sub['input_exc'], sub['avg_f_out'], yerr=sub['std_f_out'],
                    fmt='o', ms=3.5, color=color, ecolor=color, elinewidth=0.8,
                    capsize=1.5, linestyle='none', zorder=3,
                    label=rf'$\nu_i$ = {inh:.1f} Hz')
        ax.plot(sub['input_exc'], sub['fit_rate'], '-', color=color, lw=1.2, zorder=2)

    ax.set_xlabel(r'$\nu_e$ (Hz)')
    ax.set_ylabel(r'$\nu_{out}$ (Hz)')
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7, frameon=False)    
        
# -------------------- #
def mesh_3d_with_fitting(neuron_model, df_data, mean_error=None,
                         style='surface', elev=22, azim=-135, ax=None,
                         exc_col='input_exc', inh_col='input_inh',
                         data_col='avg_f_out', fit_col='fit_rate'):
    """Simulated input-output surface with the fitted transfer function on top.

    Takes the same table as `plot_TF_fitting` and pivots it internally.

    Parameters
    ----------
    neuron_model : str
    df_data : pandas.DataFrame
        Long transfer-function table holding the simulated and fitted rates
        over the input grid.
    mean_error : float, optional
        Annotated in the upper left corner if given.
    style : {'surface', 'wireframe'}
        'surface' draws the fit as the semi-transparent surface, with its cell
        edges visible, and the simulated data as markers. 'wireframe' inverts
        this, drawing the data as a filled surface and the fit as a black grid;
        note that the grid is then occluded wherever the fit falls below the
        data, which hides underfitting while showing overfitting.
    elev, azim : float, optional
        Viewing angles.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Target axes, which must already carry a 3D projection. A new figure is
        created if omitted.
    exc_col, inh_col, data_col, fit_col : str, optional
        Column names in `df_data`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    exc_vals, inh_vals, data_array = tf_grid_from_df(
        df_data, exc_col, inh_col, data_col)
    _, _, fit_array = tf_grid_from_df(df_data, exc_col, inh_col, fit_col)

    X, Y = np.meshgrid(exc_vals, inh_vals)

    if ax is None:
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    cmap = get_neuron_cmap(neuron_model)

    if style == 'surface':
        ax.plot_surface(X, Y, fit_array, cmap=cmap, alpha=0.5,
                        rstride=1, cstride=1, linewidth=0.25,
                        edgecolor='0.35', antialiased=True, shade=False)
        ax.scatter(X.ravel(), Y.ravel(), data_array.ravel(), s=9,
                   color=get_neuron_color(neuron_model), alpha=0.25, edgecolor=get_neuron_color(neuron_model),
                   linewidth=0.25, depthshade=False)
    elif style == 'wireframe':
        ax.plot_surface(X, Y, data_array, cmap=cmap, alpha=0.5,
                        rstride=1, cstride=1, linewidth=0, antialiased=True,
                        shade=True)
        ax.plot_wireframe(X, Y, fit_array, color='k', linewidth=0.6,
                          rstride=1, cstride=1, alpha=0.75)
    else:
        raise ValueError("style must be 'surface' or 'wireframe'")

    ax.set_xlabel(r'$\nu_e$ (Hz)', labelpad=2)
    ax.set_ylabel(r'$\nu_i$ (Hz)', labelpad=2)
    ax.set_zlabel(r'$\nu_{out}$ (Hz)', labelpad=2)
    ax.set_title(f'{neuron_model}', pad=0)
    ax.view_init(elev=elev, azim=azim)

    if mean_error is not None:
        ax.text2D(0.02, 0.92,
                  rf'$\langle |\epsilon| \rangle$ = {mean_error:.2f} Hz',
                  transform=ax.transAxes, ha='left', fontsize=9)

    return fig
    
# -------------------- #
def make_figure_TF_fitting(M, n_slices=5, residual_vmax=None, figsize=(15, 13)):
    '''Assemble the three by three comparison figure.

    Returns
    -------
    matplotlib.figure.Figure
    axes : dict of {(row, col): axes}
    '''
    names = list(M.keys())
    n = len(names)

    if residual_vmax is None:
        residual_vmax = max(np.nanmax(np.abs(m['df']['residual_pct'])) for m in M.values())

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n, 3, hspace=0.35, wspace=0.33)

    axes, resid_axes = {}, []
    letters = 'ABCDEFGHIJKL'

    for i, name in enumerate(names):
        m = M[name]

        ax3d = fig.add_subplot(gs[i, 0], projection='3d')
        mesh_3d_with_fitting(name, m['df'], mean_error=m['mean_error'], ax=ax3d)

        ax_sl = fig.add_subplot(gs[i, 1])
        panel_TF_slices(name, m, ax_sl, n_slices=n_slices, legend=True,
                        title='data and fit' if i == 0 else None)

        ax_rs = fig.add_subplot(gs[i, 2])
        panel_grid_map(name, m, ax_rs, value_col='residual_pct', diverging=True,
                       vmax=residual_vmax, cbar=False,
                       title='residual' if i == 0 else None)
        resid_axes.append(ax_rs)

        for k, ax in enumerate((ax3d, ax_sl, ax_rs)):
            t = ax.text2D if hasattr(ax, 'text2D') else ax.text
            t(-0.14, 1.06, letters[3 * i + k], transform=ax.transAxes,
              fontsize=13, fontweight='bold', va='top', ha='left')
            axes[(i, k)] = ax

    cb = fig.colorbar(resid_axes[0].collections[0], ax=resid_axes,
                      fraction=0.030, pad=0.02)
    cb.set_label(r'residual (% of $\nu_{out}^{max}$)')

    return fig, axes
    
# -------------------- #
def plot_TF_goodness_of_fit(df_dict, normalise='range', sigma_floor=0.1,
                            ax=None, data_col='avg_f_out',
                            fit_col='fit_rate', std_col='std_f_out'):
    """Fitted against simulated output rates for every population at once.

    Rates are normalised so that populations covering different output ranges
    are placed on a common axis and the residuals are comparable. The
    normalised root-mean-square error is reported for each population.

    Parameters
    ----------
    df_dict : dict of {str: pandas.DataFrame}
        One long transfer-function table per neuron model.
    normalise : {'range', 'sigma', 'none'}
        'range' divides both axes by the largest simulated rate of that
        population, so the residual is read as a fraction of the dynamic
        range. 'sigma' instead expresses the residual in units of the
        simulated dispersion, answering whether the fit sits within the noise
        of the simulation rather than whether it is small; the axes are then
        left in Hz and the scatter is residual against rate. 'none' plots
        rates in Hz.
    sigma_floor : float, optional
        Lower bound on the simulated std, in Hz, used only when
        normalise='sigma' to keep near-silent grid points from dominating.
    ax : matplotlib.axes.Axes, optional
    data_col, fit_col, std_col : str, optional
        Column names in each dataframe.

    Returns
    -------
    matplotlib.figure.Figure
    summary : pandas.DataFrame
        Per-population error metrics, indexed by neuron model.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
    else:
        fig = ax.get_figure()

    rows = []

    for neuron_model, df in df_dict.items():
        sim = df[data_col].to_numpy(dtype=float)
        fit = df[fit_col].to_numpy(dtype=float)
        resid = sim - fit
        scale = np.nanmax(sim)
        color = color_palette[neuron_model]

        rmse = float(np.sqrt(np.nanmean(resid ** 2)))
        rows.append({'neuron_model': neuron_model,
                     'max_rate_Hz': scale,
                     'MAE_Hz': float(np.nanmean(np.abs(resid))),
                     'RMSE_Hz': rmse,
                     'nRMSE_pct': 100.0 * rmse / scale})

        if normalise == 'sigma':
            sigma = np.clip(df[std_col].to_numpy(dtype=float),
                            sigma_floor, None)
            ax.scatter(sim, resid / sigma, s=16, color=color, alpha=0.6,
                       edgecolor='k', linewidth=0.25, label=neuron_model)
        else:
            s = scale if normalise == 'range' else 1.0
            ax.scatter(sim / s, fit / s, s=16, color=color, alpha=0.6,
                       edgecolor='k', linewidth=0.25, label=neuron_model)

    if normalise == 'sigma':
        ax.axhline(0, color='0.4', lw=1)
        for k in (-2, 2):
            ax.axhline(k, color='0.4', lw=0.8, ls=':')
        ax.set_xlabel(r'simulated $\nu_{out}$ (Hz)')
        ax.set_ylabel(r'residual / simulated std')
    else:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0], 0.0)
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], '--', color='0.4', lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        unit = r'/ $\nu_{out}^{max}$' if normalise == 'range' else '(Hz)'
        ax.set_xlabel(rf'simulated $\nu_{{out}}$ {unit}')
        ax.set_ylabel(rf'fitted $\nu_{{out}}$ {unit}')

    summary = pd.DataFrame(rows).set_index('neuron_model')

    ax.text(0.03, 0.97,
            '\n'.join(f"{m}: {r.nRMSE_pct:.1f}%" for m, r in summary.iterrows()),
            transform=ax.transAxes, va='top', ha='left', fontsize=8)
    ax.legend(loc='lower right', fontsize=8, frameon=False)

    return fig, summary

def plot_TF_sliced_by_mossy(df, unique_mossy, unique_inh,
                            exc_col='input_exc', inh_col='input_inh',
                            mossy_col='input_exc_m',
                            data_col='avg_f_out', fit_col='fit_rate',
                            cmap='viridis', ncols=3,
                            title='GoC transfer function (data vs fit), sliced by mossy input'):
    """One panel per mossy value. x = granule excitation, colour = inhibition.
    Markers = numerical data, solid line = analytical fit."""
    n = len(unique_mossy)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    cols = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_inh)))
    norm = plt.Normalize(vmin=float(np.min(unique_inh)), vmax=float(np.max(unique_inh)))
    for k, m_val in enumerate(unique_mossy):
        ax = axes[k // ncols][k % ncols]
        sub = df[df[mossy_col] == m_val]
        for inh_val, c in zip(unique_inh, cols):
            mask = sub[inh_col] == inh_val
            x = sub.loc[mask, exc_col].to_numpy()
            order = np.argsort(x)
            x = x[order]
            y_data = sub.loc[mask, data_col].to_numpy()[order]
            y_fit = sub.loc[mask, fit_col].to_numpy()[order]
            ax.plot(x, y_data, 'o', color=c, alpha=0.5)
            ax.plot(x, y_fit, '-', color=c)
        ax.set_title(f'mossy = {m_val:.1f} Hz')
        ax.set_xlabel('Excitatory input via granule (Hz)')
        ax.set_ylabel('Output rate (Hz)')
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, label='Inhibitory input (Hz)')
    fig.suptitle(title)
    return fig


# =========================================================================== #
# Transfer-function data with N synaptic channels
# =========================================================================== #
# Expected inputs: the .dat written by TF_data_extraction_neuron_<N>_inputs.py
# (columns input_1 ... input_N avg_f_out std_f_out) and its _meta.json.

def tf_channel_info(meta, E_L=None):
    """Per-channel labels, colours and markers from a TF-extraction meta dict.

    A channel is called excitatory when its reversal potential lies above the
    resting potential E_L (read from ``meta['E_L']`` unless given), inhibitory
    otherwise. Excitatory channels use the excitatory-synapse green and circle
    markers, inhibitory ones the inhibitory red and square markers.

    Returns
    -------
    dict : {channel: {"col", "label", "short", "kind", "color", "marker"}}
    """

    if E_L is None:
        E_L = meta.get('E_L')
    if E_L is None:
        raise ValueError("E_L not in meta; pass it explicitly (mV).")
    info = {}
    for k, c in meta['channels'].items():
        kind = 'exc' if c['E'] > E_L else 'inh'
        info[k] = {
            'col': f'input_{k}',
            'kind': kind,
            'short': f'ch {k} ({kind})',
            'label': f"ch {k} rate (Hz) [E = {c['E']:g} mV, Q = {c['Q']:g} nS, τ = {c['tau']:g} ms]",
            'color': syn_colors['E'] if kind == 'exc' else syn_colors['I'],
            'marker': 'o' if kind == 'exc' else 's',
        }
    return info


def tf_grid_values(df_data, channels):
    """Unique input values per channel, and whether the table is a full grid."""
    vals = {k: np.sort(df_data[f'input_{k}'].unique()) for k in channels}
    n_expected = int(np.prod([len(v) for v in vals.values()]))
    n_unique = len(df_data.drop_duplicates(subset=[f'input_{k}' for k in channels]))
    return vals, (n_unique == n_expected == len(df_data))


def tf_slice(df_data, fixed, atol=1e-6):
    """Rows of *df_data* where each channel in *fixed* ({channel: rate}) is held at that rate."""
    mask = np.ones(len(df_data), dtype=bool)
    for k, v in fixed.items():
        mask &= np.isclose(df_data[f'input_{k}'].to_numpy(), v, atol=atol)
    if not mask.any():
        raise ValueError(f"No rows with {fixed}; pick values from the sweep grid.")
    return df_data.loc[mask]


def tf_slice_2d(df_data, x_ch, y_ch, fixed, value_col='avg_f_out'):
    """2D array of *value_col* over (x_ch, y_ch) with the other channels fixed.

    Returns (x_vals, y_vals, values) with values of shape (n_y, n_x), the
    layout used by heatmap_InOut, plot_contours, mesh_3d, plot_gain and
    plot_InOut_relation.
    """
    sub = tf_slice(df_data, fixed)
    return tf_grid_from_df(sub, exc_col=f'input_{x_ch}', inh_col=f'input_{y_ch}', value_col=value_col)


def _fixed_suffix(fixed):
    return '' if not fixed else ' | ' + ', '.join(f'ch {k} = {v:g} Hz' for k, v in fixed.items())


# -------------------- #
def plot_TF_output_summary(df_data, neuron_model, silent_thr=0.1):
    """Distribution of the output rate and its std across all simulated inputs.

    Left: histogram of avg_f_out (silent points, rate < *silent_thr* Hz,
    counted in the title). Right: std_f_out against avg_f_out.
    """
    color = get_neuron_color(neuron_model)
    f = df_data['avg_f_out'].to_numpy()
    s = df_data['std_f_out'].to_numpy()
    frac_silent = np.mean(f < silent_thr)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].hist(f, bins=40, color=color, alpha=0.8)
    axs[0].set_xlabel('Output frequency (Hz)')
    axs[0].set_ylabel('Number of input points')
    axs[0].set_title(f'{neuron_model}: {len(f)} points, {100 * frac_silent:.0f}% below {silent_thr:g} Hz')

    axs[1].plot(f, s, 'o', color=color, alpha=0.5, markersize=4)
    axs[1].set_xlabel('Output frequency (Hz)')
    axs[1].set_ylabel('std of output frequency (Hz)')
    axs[1].set_title('Output variability')
    plt.tight_layout()
    return fig


# -------------------- #
def plot_TF_main_effects(df_data, channel_info, neuron_model, n_bins=10):
    """Output rate against each input channel, all other channels pooled.

    One panel per channel: every simulated point (grey) plus the mean output
    at each input value (grid data) or in *n_bins* bins (non-grid data, e.g.
    LHS), drawn in the channel's synapse colour and marker.
    """
    chans = list(channel_info)
    fig, axs = plt.subplots(1, len(chans), figsize=(4 * len(chans), 4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, chans):
        ci = channel_info[k]
        x = df_data[ci['col']].to_numpy()
        y = df_data['avg_f_out'].to_numpy()
        ax.plot(x, y, '.', color='#bdbdbd', alpha=0.5, markersize=3, zorder=1)

        ux = np.unique(x)
        if len(ux) <= 2 * n_bins:   # grid: average at each grid value
            xm = ux
            ym = np.array([y[x == u].mean() for u in ux])
        else:                       # scattered: average in bins
            edges = np.linspace(x.min(), x.max(), n_bins + 1)
            idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
            xm = 0.5 * (edges[:-1] + edges[1:])
            ym = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
        ax.plot(xm, ym, '-', marker=ci['marker'], color=ci['color'], zorder=2, label='mean over other channels')
        ax.set_xlabel(f"ch {k} rate (Hz)")
        ax.set_title(ci['short'], color=ci['color'])
    axs[0].set_ylabel('Output frequency (Hz)')
    axs[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'{neuron_model}: output rate against each input channel')
    plt.tight_layout()
    return fig


# -------------------- #
def plot_TF_channel_curves(df_data, channel_info, neuron_model, reference, hue_ch):
    """One-channel-at-a-time input–output curves around a reference point.

    Panel k varies channel k while the remaining channels sit at *reference*
    ({channel: rate}); within each panel one curve is drawn per value of
    *hue_ch* (light -> dark for increasing rate). Error bars: std_f_out.
    The panel of *hue_ch* itself shows a single curve.
    """
    chans = list(channel_info)
    hue_vals = np.sort(df_data[f'input_{hue_ch}'].unique())
    cmap = get_neuron_line_cmap(neuron_model)

    fig, axs = plt.subplots(1, len(chans), figsize=(4 * len(chans), 4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, chans):
        ci = channel_info[k]
        curves = [None] if k == hue_ch else list(hue_vals)
        colors = cmap(np.linspace(0, 1, max(len(curves), 2)))
        for c_i, h in enumerate(curves):
            fixed = {j: reference[j] for j in chans if j != k}
            if h is not None:
                fixed[hue_ch] = h
            sub = tf_slice(df_data, fixed).sort_values(ci['col'])
            color = ci['color'] if h is None else colors[c_i]
            ax.errorbar(sub[ci['col']], sub['avg_f_out'], sub['std_f_out'],
                        fmt='-', marker=ci['marker'], color=color, markersize=4, capsize=2,
                        label=None if h is None else f'{h:g}')
        ax.set_xlabel(f'ch {k} rate (Hz)')
        ax.set_title(ci['short'], color=ci['color'])
        if k != hue_ch:
            ax.legend(title=f'ch {hue_ch} (Hz)', fontsize=7, title_fontsize=8)
    axs[0].set_ylabel('Output frequency (Hz)')
    ref_txt = ', '.join(f'ch {j} = {v:g}' for j, v in reference.items() if j != hue_ch)
    fig.suptitle(f'{neuron_model}: one channel at a time (others at {ref_txt} Hz)')
    plt.tight_layout()
    return fig


# -------------------- #
def plot_TF_slice_grid(df_data, channel_info, neuron_model, x_ch, y_ch,
                       row_ch=None, col_ch=None, fixed=None,
                       value_col='avg_f_out', vmax=None,
                       vmin=0, cmap=None, cbar_label='Output frequency (Hz)'):
    """Small-multiple heatmaps of the N-dimensional TF ("dimensional stacking").

    Each panel is a (x_ch, y_ch) heatmap; panels are arranged by the values of
    *row_ch* (bottom -> top, increasing) and *col_ch* (left -> right,
    increasing). Any further channel must be held fixed via *fixed*. All
    panels share one colour scale, so panels can be compared directly.

    For signed quantities (e.g. ``value_col='residual'``) pass a diverging
    *cmap* and symmetric *vmin*/*vmax*.
    """
    fixed = dict(fixed or {})
    row_vals = np.sort(df_data[f'input_{row_ch}'].unique()) if row_ch else [None]
    col_vals = np.sort(df_data[f'input_{col_ch}'].unique()) if col_ch else [None]
    nr, nc = len(row_vals), len(col_vals)

    if vmax is None:
        vmax = float(tf_slice(df_data, fixed)[value_col].max()) if fixed else float(df_data[value_col].max())
    if cmap is None:
        cmap = get_neuron_cmap(neuron_model)

    fig, axs = plt.subplots(nr, nc, figsize=(1.9 * nc + 1.5, 1.7 * nr + 1.0),
                            sharex=True, sharey=True, squeeze=False)
    im = None
    for r, rv in enumerate(row_vals):
        for c, cv in enumerate(col_vals):
            ax = axs[nr - 1 - r, c]   # increasing row value goes up
            fx = dict(fixed)
            if row_ch: fx[row_ch] = rv
            if col_ch: fx[col_ch] = cv
            xv, yv, val = tf_slice_2d(df_data, x_ch, y_ch, fx, value_col=value_col)
            dx = (xv[1] - xv[0]) / 2 if len(xv) > 1 else 0.5
            dy = (yv[1] - yv[0]) / 2 if len(yv) > 1 else 0.5
            im = ax.imshow(val, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[xv[0] - dx, xv[-1] + dx, yv[0] - dy, yv[-1] + dy])
            if r == nr - 1 and col_ch:
                ax.set_title(f'ch {col_ch} = {cv:g} Hz', fontsize=9, color=channel_info[col_ch]['color'])
            if c == nc - 1 and row_ch:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(f'ch {row_ch} = {rv:g} Hz', fontsize=9, rotation=-90, va='bottom',
                              color=channel_info[row_ch]['color'])
    fig.supxlabel(f'ch {x_ch} rate (Hz)', color=channel_info[x_ch]['color'])
    fig.supylabel(f'ch {y_ch} rate (Hz)', color=channel_info[y_ch]['color'])
    fig.suptitle(f'{neuron_model}: {value_col}{_fixed_suffix(fixed)}')
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label=cbar_label)
    return fig

def tf_slice_as_ei(df_data, x_ch, y_ch, fixed):
    """2D slice of N-channel TF data in the layout of the e/i plotting functions.

    Keeps the rows with the channels in *fixed* at the given rates, adds
    ``input_exc`` (= ``input_<x_ch>``) and ``input_inh`` (= ``input_<y_ch>``),
    and sorts by (input_inh, input_exc). The original ``input_1 ... input_N``
    columns are kept, so N-channel TF evaluations (``TF_template``,
    ``res_2_func``) still work on the slice. With it, plot_TF_fitting,
    plot_TF_fitting_viridis, mesh_3d_with_fitting, plots_TF_fitting,
    make_TF_gif, plot_residuals_TF_fitting, plot_abs_residuals_TF_fitting and
    plot_membrane_potential_fluctuations can be used on N-channel data.
    """
    sub = tf_slice(df_data, fixed).copy()
    sub['input_exc'] = sub[f'input_{x_ch}']
    sub['input_inh'] = sub[f'input_{y_ch}']
    return sub.sort_values(['input_inh', 'input_exc']).reset_index(drop=True)

# -------------------- #
def plot_membrane_fluctuations_overview(df_data, mu_V, sig_V, tau_V, neuron_model, sig_V_min=4e-3):
    """(mu_V, sig_V, tau_V) over all input points, coloured by the output rate.

    Left: sig_V against mu_V. Right: tau_V against mu_V. The dashed line marks
    *sig_V_min* (V): below it the erfc (fluctuation-driven) approximation is
    expected to degrade; the fraction of points below it is given in the title.
    """
    f = df_data['avg_f_out'].to_numpy()
    order = np.argsort(f)  # draw active points on top
    mu, sg, tv = 1e3 * np.asarray(mu_V)[order], 1e3 * np.asarray(sig_V)[order], 1e3 * np.asarray(tau_V)[order]
    cmap = get_neuron_cmap(neuron_model) if neuron_model in neuron_cmaps else 'viridis'

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    sc = axs[0].scatter(mu, sg, c=f[order], cmap=cmap, s=12)
    axs[0].axhline(1e3 * sig_V_min, color='k', ls='--', lw=1)
    axs[0].set_xlabel(r'$\mu_V$ (mV)')
    axs[0].set_ylabel(r'$\sigma_V$ (mV)')
    frac = np.mean(np.asarray(sig_V) < sig_V_min)
    axs[0].set_title(fr'{100 * frac:.0f}% of points with $\sigma_V$ < {1e3 * sig_V_min:g} mV')

    axs[1].scatter(mu, tv, c=f[order], cmap=cmap, s=12)
    axs[1].set_xlabel(r'$\mu_V$ (mV)')
    axs[1].set_ylabel(r'$\tau_V$ (ms)')
    axs[1].set_title('Autocorrelation time')
    fig.colorbar(sc, ax=axs, label='Output frequency (Hz)')
    fig.suptitle(f'{neuron_model}: membrane potential fluctuations')
    return fig


# -------------------- #
def plot_TF_fit_overview(df_data, neuron_model, mean_error, fit_col='fit_rate'):
    """Fitted against simulated output rate over all input points.

    Left: fit vs data with the identity line. Right: histogram of residuals
    (data - fit). Works for any number of channels and for non-grid data.
    """
    color = get_neuron_color(neuron_model)
    y = df_data['avg_f_out'].to_numpy()
    yf = df_data[fit_col].to_numpy()
    res = y - yf
    lim = (min(y.min(), yf.min()) - 1, max(y.max(), yf.max()) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(lim, lim, color='#969696', lw=1, zorder=1)
    axs[0].plot(y, yf, 'o', color=color, alpha=0.5, markersize=4, zorder=2)
    axs[0].set_xlim(lim); axs[0].set_ylim(lim)
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('Simulated output rate (Hz)')
    axs[0].set_ylabel('Fitted output rate (Hz)')
    axs[0].set_title(f'{neuron_model}: {len(y)} points, MSE = {mean_error:.3f} Hz$^2$')

    axs[1].hist(res, bins=40, color=color, alpha=0.8)
    axs[1].axvline(0, color='k', lw=1)
    axs[1].set_xlabel('Residual, data - fit (Hz)')
    axs[1].set_ylabel('Number of input points')
    axs[1].set_title(f'max |residual| = {np.max(np.abs(res)):.2f} Hz')
    plt.tight_layout()
    return fig


# -------------------- #
def plot_TF_fit_error_by_channel(df_data, channel_info, neuron_model, fit_col='fit_rate'):
    """Mean squared fit error at each input value of each channel (others pooled).

    One panel per channel, in the channel's synapse colour and marker; shows
    where in the input space the fit degrades. Same quantity as
    ``ntmf.transfer_function.get_mean_error_distribution_general``.
    """
    chans = list(channel_info)
    sq = (df_data['avg_f_out'] - df_data[fit_col]).to_numpy() ** 2
    fig, axs = plt.subplots(1, len(chans), figsize=(4 * len(chans), 4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, chans):
        ci = channel_info[k]
        x = df_data[ci['col']].to_numpy()
        ux = np.unique(x)
        mse = np.array([sq[np.isclose(x, u)].mean() for u in ux])
        ax.plot(ux, mse, '-', marker=ci['marker'], color=ci['color'])
        ax.set_xlabel(f'ch {k} rate (Hz)')
        ax.set_title(ci['short'], color=ci['color'])
    axs[0].set_ylabel('Mean squared error (Hz$^2$)')
    fig.suptitle(f'{neuron_model}: fit error against each input channel (others pooled)')
    plt.tight_layout()
    return fig

# -------------------- #
def plot_MF_single_pop(mf, driving_input, channel_info, neuron_model, nn=None, stats=None):
    """Mean-field run of a single N-channel population, optionally against spiking neurons.

    Top: input rate per channel (synapse colour; line style distinguishes
    channels of the same kind). Middle: mu_V with a +/- sigma_V band.
    Bottom: MF rate, the stationary rate TF(inputs), and, if *nn* is given,
    the binned spiking population rate (with RMSE and Pearson r from *stats*).
    """
    styles = ['-', '--', ':', '-.']
    seen = {'exc': 0, 'inh': 0}
    color = get_neuron_color(neuron_model)
    t = mf['t']

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1.4]})
    for k, ci in channel_info.items():
        ls = styles[seen[ci['kind']] % len(styles)]
        seen[ci['kind']] += 1
        axs[0].plot(t, driving_input[k], ls, color=ci['color'], lw=1.5, label=ci['short'])
    axs[0].set_ylabel('Input rate (Hz)')
    axs[0].legend(loc='upper right', fontsize=8, ncol=len(channel_info))

    mu, sg = 1e3 * np.asarray(mf['mu_V']), 1e3 * np.asarray(mf['sig_V'])
    axs[1].fill_between(t, mu - sg, mu + sg, color=color, alpha=0.2, lw=0, label=r'$\mu_V \pm \sigma_V$')
    axs[1].plot(t, mu, color=color, lw=1.5)
    axs[1].set_ylabel('V (mV)')
    axs[1].legend(loc='upper right', fontsize=8)

    if nn is not None:
        axs[2].step(nn['t'], nn['rate'], where='mid', color='#969696', lw=1.2, label='spiking neurons')
    axs[2].plot(t, mf['TF'], ':', color='k', lw=1, label='stationary TF')
    axs[2].plot(t, mf['nu'], color=color, lw=2, label='mean field')
    axs[2].set_ylabel('Output rate (Hz)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend(loc='upper right', fontsize=8)
    if stats is not None:
        axs[2].set_title(f"RMSE = {stats['rmse']:.2f} Hz, Pearson r = {stats['corr']:.2f}", fontsize=10)

    fig.suptitle(f'{neuron_model}: single-population mean field')
    plt.tight_layout()
    return fig

# -------------------- #
# Networks with any number of populations (config-driven)
# -------------------- #

_EXTERNAL_COLOR = '#969696'   # grey for external Poisson populations
_POP_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']


def population_colors(names):
    """{population name: colour}; names not in ``color_palette`` get neutral grey."""
    return {n: color_palette.get(n, _DEFAULT_NEURON_COLOR) for n in names}


def _channel_sign(neuron_model, ch, model_folder, idx=0):
    """'E' if channel *ch* of *neuron_model* is depolarising (E_k > E_L), else 'I'."""
    with open(os.path.join(model_folder, f'{neuron_model}.json'), 'r') as fh:
        m = json.load(fh)[0][idx]['model']
    return 'E' if m[f'E_{ch}'] > m['E_L'] else 'I'


def _quad_bezier(p0, c, p2, n=200):
    """Points of the quadratic Bezier curve p0 -> p2 with control point c."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2


def _in_box(pts, centre, half_w, half_h, pad=0.0):
    return (np.abs(pts[:, 0] - centre[0]) <= half_w + pad) & (np.abs(pts[:, 1] - centre[1]) <= half_h + pad)


def _network_layout(pops, ext, edges):
    """Position of every node of the network diagram.

    Populations sit on an ellipse, so that a straight arrow between two of them
    never crosses a third. Their order around the ellipse follows the
    feedforward depth (sources on the left, targets on the right; connections
    that close a loop are ignored for this). External inputs form a column on
    the left, ordered by the height of their targets.
    """
    internal = [(s, t) for s, t, _c in edges if s in pops and s != t]
    back, state = set(), {}

    def _dfs(n):
        state[n] = 'open'
        for s, t in internal:
            if s == n:
                if state.get(t) == 'open':
                    back.add((s, t))
                elif t not in state:
                    _dfs(t)
        state[n] = 'done'

    for n in pops:
        if n not in state:
            _dfs(n)
    depth = {n: 0 for n in pops}
    for _ in range(len(pops)):
        for s, t in internal:
            if (s, t) not in back and depth[t] < depth[s] + 1:
                depth[t] = depth[s] + 1

    n_pop = len(pops)
    rx, ry = 1.1 + 0.3 * n_pop, 0.45 + 0.3 * n_pop
    if n_pop == 1:
        slots = [(0.0, 0.0)]
    else:
        ang = np.pi - 2 * np.pi * np.arange(n_pop) / n_pop
        slots = [(rx * np.cos(a), ry * np.sin(a)) for a in ang]
    # slots from left to right (top before bottom); nodes by depth, config order on ties
    slot_order = sorted(range(n_pop), key=lambda k: (round(slots[k][0], 6), -slots[k][1]))
    node_order = sorted(pops, key=lambda n: (depth[n], list(pops).index(n)))
    pos = {n: slots[k] for n, k in zip(node_order, slot_order)}

    targets = {n: [t for s, t, _c in edges if s == n] for n in ext}
    ext_sorted = sorted(ext, key=lambda n: -np.mean([pos[t][1] for t in targets[n]] or [0.0]))
    x_ext = -rx - 1.7
    span = max(len(ext_sorted) - 1, 1)
    for k, n in enumerate(ext_sorted):
        pos[n] = (x_ext, 1.05 * (span / 2 - k) if len(ext_sorted) > 1 else 0.0)
    return pos


def plot_network_diagram(network_config, model_folder, idx=0, ax=None, colors=None):
    """Draw populations, external inputs and connections of a config-driven network.

    External inputs are in a column on the left; populations sit on an ellipse,
    ordered by feedforward depth (see ``_network_layout``). Each connection is an arrow from source to target, green
    for a depolarising and red for a hyperpolarising target channel
    (``syn_colors``), labelled on the arrow with 'ch k, K = ...'
    (K = N_source * conn_prob). Arrows bend to avoid other boxes; the two
    directions of a reciprocal pair bend to opposite sides.
    """
    from matplotlib.path import Path as MplPath

    pops = network_config['populations']
    ext = network_config.get('external_populations', {})
    conns = {**network_config.get('internal_connections', {}), **network_config.get('external_connections', {})}
    edges = [(k.split('->')[0].strip(), k.split('->')[1].strip(), c) for k, c in conns.items()]
    colors = colors or population_colors(pops)
    sizes = {**{k: v['N'] for k, v in ext.items()}, **{k: v['N'] for k, v in pops.items()}}
    pos = _network_layout(pops, ext, edges)
    hw, hh = 0.4, 0.23  # half width and half height of a box

    if ax is None:
        xs0 = [x for x, _ in pos.values()]
        ys0 = [y for _, y in pos.values()]
        fig, ax = plt.subplots(figsize=(1.9 * (max(xs0) - min(xs0) + 1.2), 1.9 * (max(ys0) - min(ys0) + 1.4)))
    else:
        fig = ax.figure
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - hw - 0.1, max(xs) + hw + 0.1)
    ax.set_ylim(min(ys) - hh - 0.45, max(ys) + hh + 0.6)
    ax.axis('off')

    pair = {(s, t) for s, t, _c in edges}
    labels = []  # (text, colour, candidate points along the arrow, index of the arrow)
    arrow_pts = []
    for s, t, c in edges:
        col = syn_colors[_channel_sign(pops[t]['neuron_model'], c['ch'], model_folder, idx)]
        text = f"ch {c['ch']}, K = {sizes[s] * c['conn_prob']:g}"
        p0 = np.array(pos[s], float)
        if s == t:  # self-connection: loop above the box
            loop = patches.FancyArrowPatch((p0[0] - 0.15, p0[1] + hh), (p0[0] + 0.15, p0[1] + hh),
                                           connectionstyle='arc3,rad=-1.4', arrowstyle='-|>',
                                           mutation_scale=11, color=col, lw=1.2, zorder=2)
            ax.add_patch(loop)
            labels.append((text, col, [p0 + [0, hh + 0.33]], None))
            continue
        p2 = np.array(pos[t], float)
        d = p2 - p0
        rads = [0.2, 0.32, 0.45] if (t, s) in pair else [0.0, 0.12, -0.12, 0.25, -0.25, 0.4, -0.4]
        best = None
        for rad in rads:
            ctrl = (p0 + p2) / 2 + rad * np.array([d[1], -d[0]])
            pts = _quad_bezier(p0, ctrl, p2)
            vis = ~_in_box(pts, p0, hw, hh) & ~_in_box(pts, p2, hw, hh)
            i0 = int(np.argmax(vis))
            i1 = len(vis) - 1 - int(np.argmax(vis[::-1]))
            seg = pts[i0:i1 + 1]
            hits = sum(int(_in_box(seg, pos[n], hw, hh, 0.04).any()) for n in pos if n not in (s, t))
            if best is None or hits < best[0]:
                best = (hits, seg)
            if hits == 0:
                break
        seg = best[1]
        ax.add_patch(patches.FancyArrowPatch(path=MplPath(seg), arrowstyle='-|>', mutation_scale=11,
                                             color=col, lw=1.2, zorder=2))
        arc = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        cand = [seg[np.searchsorted(arc, f * arc[-1]).clip(0, len(seg) - 1)]
                for f in (0.7, 0.6, 0.8, 0.5, 0.4, 0.3, 0.2)]
        labels.append((text, col, cand, len(arrow_pts)))
        arrow_pts.append(seg)

    for n, (x, y) in pos.items():
        is_ext = n in ext
        ax.add_patch(patches.FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle='round,pad=0.01',
                                            fc='white' if is_ext else colors[n],
                                            ec=_EXTERNAL_COLOR if is_ext else colors[n], lw=1.5, zorder=3))
        txt = f"{n}\nN = {sizes[n]}" if is_ext else f"{n}\n{pops[n]['neuron_model']}\nN = {sizes[n]}"
        ax.text(x, y, txt, ha='center', va='center', fontsize=7, zorder=4, linespacing=1.3,
                color=_EXTERNAL_COLOR if is_ext else 'white', fontweight='bold')

    # Place each label on its own arrow, at the candidate point where it covers the
    # least: boxes and labels already placed weigh most, other arrows less.
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    taken = [np.array([x - hw, y - hh, x + hw, y + hh]) for x, y in pos.values()]

    def score(b, own):
        area = sum(max(0.0, min(b[2], o[2]) - max(b[0], o[0])) * max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
                   for o in taken)
        cover = sum(int(((q[:, 0] > b[0]) & (q[:, 0] < b[2]) & (q[:, 1] > b[1]) & (q[:, 1] < b[3])).any())
                    for k, q in enumerate(arrow_pts) if k != own)
        return 100.0 * area + cover

    for text, col, cand, own in labels:
        tx = ax.text(*cand[0], text, ha='center', va='center', fontsize=7, color=col, zorder=5,
                     bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))
        best = None
        for pnt in cand:
            tx.set_position(pnt)
            ext_box = tx.get_window_extent(renderer).expanded(1.1, 1.2)
            (x0, y0), (x1, y1) = to_data.transform([[ext_box.x0, ext_box.y0], [ext_box.x1, ext_box.y1]])
            bb = np.array([x0, y0, x1, y1])
            sc = score(bb, own)
            if best is None or sc < best[0]:
                best = (sc, pnt, bb)
            if sc == 0:
                break
        tx.set_position(best[1])
        taken.append(best[2])

    ax.plot([], [], color=syn_colors['E'], label='depolarising channel')
    ax.plot([], [], color=syn_colors['I'], label='hyperpolarising channel')
    ax.legend(frameon=False, fontsize=7, loc='lower left', ncol=2)
    return fig


def plot_in_degree_histograms(built, network_config, ncols=3):
    """Realised in-degree per connection vs the binomial(N_source, conn_prob) expectation."""
    from scipy.stats import binom
    from ntmf.network import realized_in_degrees, all_connections

    deg = realized_in_degrees(built)
    conns = all_connections(network_config)
    sizes = {**{k: v['N'] for k, v in network_config.get('external_populations', {}).items()},
             **{k: v['N'] for k, v in network_config['populations'].items()}}
    colors = population_colors(network_config['populations'])
    n = len(deg)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows), squeeze=False)
    for ax, (key, d) in zip(axs.flat, deg.items()):
        src, tgt = [x.strip() for x in key.split('->')]
        N_src, p = sizes[src], conns[key]['conn_prob']
        if src == tgt:
            N_src -= 1  # no self-connections
        k = np.arange(d.min(), d.max() + 1)
        ax.hist(d, bins=np.arange(d.min() - 0.5, d.max() + 1.5), density=True,
                color=colors[tgt], alpha=0.6)
        ax.plot(k, binom.pmf(k, N_src, p), 'k-', lw=1, label='binomial')
        ax.set_title(f"{key}  (ch {conns[key]['ch']})", fontsize=9)
        ax.text(0.98, 0.95, f"mean {d.mean():.1f}\nexpected {N_src * p:.1f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5)
        ax.set_xlabel('afferents per neuron', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axs.flat[n:]:
        ax.axis('off')
    axs.flat[0].legend(frameon=False, fontsize=7, loc='upper left')
    plt.tight_layout()
    return fig


def plot_external_rates(input_config, network_config, ax=None, dt=0.001):
    """Rate of every external population over the protocol, labelled with its targets."""
    from ntmf.network import external_rate_trace

    T = float(input_config['simulation']['sim_duration'])
    t = np.arange(0.0, T, dt)
    targets = {}
    for key in network_config.get('external_connections', {}):
        s, tg = [x.strip() for x in key.split('->')]
        targets.setdefault(s, []).append(tg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.2))
    else:
        fig = ax.figure
    pop_colors = population_colors(network_config['populations'])
    for k, (name, spec) in enumerate(input_config['external_populations'].items()):
        tg = targets.get(name, [])
        col = pop_colors[tg[0]] if len(tg) == 1 else _EXTERNAL_COLOR
        ax.plot(t, external_rate_trace(spec, t), color=col, lw=1.4,
                ls=['-', '--', ':', '-.'][k % 4], label=f"{name} → {', '.join(tg)}")
    ax.set_ylabel('Input rate (Hz)')
    ax.set_xlabel('Time (s)')
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def _add_epoch_lines(ax, epochs):
    if epochs:
        for t0, _ in epochs[1:]:
            ax.axvline(t0, color='0.85', lw=0.6, zorder=0)


def network_raster_plot_pops(spike_mons, sizes, colors=None, epochs=None, max_neurons=500, markersize=1):
    """Raster of several populations stacked from the bottom; at most *max_neurons* shown per population."""
    colors = colors or population_colors(spike_mons)
    fig, ax = plt.subplots(figsize=(10, 6))
    offset, ticks = 0, [0]
    for name, mon in spike_mons.items():
        n_show = min(sizes[name], max_neurons)
        i = np.asarray(mon.i)
        keep = i < n_show
        ax.plot(np.asarray(mon.t / b2.second)[keep], i[keep] + offset, '.', color=colors[name],
                markersize=markersize, rasterized=True)
        ax.text(1.01, offset + n_show / 2, f'{name}\n({n_show}/{sizes[name]})', color=colors[name],
                transform=ax.get_yaxis_transform(), va='center', fontsize=9)
        offset += n_show
        ticks.append(offset)
    _add_epoch_lines(ax, epochs)
    ax.set_yticks(ticks)
    ax.set_ylim(0, offset)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_network_pop_rates(spike_mons, sizes, sim_duration, colors=None, bin_size=0.02, epochs=None,
                           epoch_rates=None, skip=0.2):
    """One panel per population: binned population rate, optionally with epoch means as horizontal bars."""
    from ntmf.network import population_rates

    colors = colors or population_colors(spike_mons)
    n = len(spike_mons)
    fig, axs = plt.subplots(n, 1, figsize=(10, 1.6 * n + 0.6), sharex=True, squeeze=False)
    for k, (ax, (name, mon)) in enumerate(zip(axs[:, 0], spike_mons.items())):
        t, r = population_rates(mon, sizes[name], sim_duration, bin_size)
        ax.plot(t, r, color=colors[name], lw=1)
        if epoch_rates is not None and epochs is not None:
            for (t0, t1), m in zip(epochs, epoch_rates[name]):
                ax.plot([t0 + skip, t1], [m, m], color='k', lw=1.5)
        _add_epoch_lines(ax, epochs)
        ax.set_ylabel(f'{name}\n(Hz)')
        ax.spines[['top', 'right']].set_visible(False)
    axs[-1, 0].set_xlabel('Time (s)')
    if epoch_rates is not None:
        axs[0, 0].plot([], [], 'k', lw=1.5, label=f'epoch mean (first {skip} s skipped)')
        axs[0, 0].legend(frameon=False, fontsize=7, loc='upper right')
    plt.tight_layout()
    return fig