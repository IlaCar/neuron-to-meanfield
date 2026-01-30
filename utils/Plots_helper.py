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

import brian2 as b2

# -------------------- #
color_palette = {"FS": '#cb181d',
                 "RS": '#238b45',
                 "input": '#9ecae1'}

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
                                     pretty_plot = False):

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
    
    if pretty_plot == False:    
        ax[1].plot(pop2.t/b2.second, pop2.v[0] / b2.mV, color='#00441b')
        ax[1].plot(pop2.t/b2.second, pop2.v[1] / b2.mV, color='#238b45')
        ax[1].plot(pop2.t/b2.second, pop2.v[2] / b2.mV, color='#74c476')
    else:
        ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[0], -50) / b2.mV, '--', color='#00441b')   
        ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[1], -50) / b2.mV, '--', color='#238b45')
        ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[2], -50) / b2.mV, '--', color='#74c476')
    
    ax[1].set_title('Selected RS traces')

    if ext_input_0 != None:
        ax[-1].plot(ext_input_0.t / b2.second, ext_input_0.i, '.', color='k', alpha=0.3, markersize=1)
    ax[-1].plot(ext_input_1.t / b2.second, ext_input_1.i, '.', color=color_palette['FS'], markersize=1)
    ax[-1].plot(ext_input_2.t / b2.second, ext_input_2.i, '.', color=color_palette['RS'], markersize=1)
    ax[-1].plot(ext_input_3.t / b2.second, ext_input_3.i, '.', color=color_palette['FS'], markersize=1)
    ax[-1].plot(ext_input_4.t / b2.second, ext_input_4.i, '.', color=color_palette['RS'], markersize=1)
    
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
                facecolor=color_palette[pop],
                edgecolor=None,
                alpha=alpha,
            )
            ax.add_patch(rect)

            if annotate:
                ax.text(
                    (t0 + t1) / 2,
                    y_pad + box_height / 2,
                    "+",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                )

    # inhibitory inputs
    if inh_intervals is not None:
        for t0, t1, pop in inh_intervals:
            rect = patches.Rectangle(
                (t0, y_pad),
                t1 - t0,
                box_height,
                facecolor=color_palette[pop],
                edgecolor=None,
                alpha=alpha,
            )
            ax.add_patch(rect)

            if annotate:
                ax.text(
                    (t0 + t1) / 2,
                    y_pad + box_height / 2,
                    "−",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
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
                        x_lim=None
                        ):
    
    m_size = 1 if markersize is None else markersize
        
    fig, ax = plt.subplots(figsize=(10, 6))
    offset = 0

    if pop1 is not None:
        ax.plot(
            pop1.t / b2.second,
            pop1.i + offset,
            ',',
            color=color_palette['FS'],
            markersize=m_size
        )
        offset += N_pop1

    if pop2 is not None:
        ax.plot(
            pop2.t / b2.second,
            pop2.i + offset,
            ',',
            color=color_palette['RS'],
            markersize=m_size
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
        transform=ax.get_yaxis_transform(),
        va='center',
        ha='left',
        fontsize=12,
    )

    ax.text(
        0.01,
        N_pop1 + N_pop2 / 2,
        'RS',
        transform=ax.get_yaxis_transform(),
        va='center',
        ha='left',
        fontsize=12,
    )

    ax.set_title('Network Raster Plot')

    # Input boxes 
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
                              inh_intervals = None):

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
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_bins, mean_rate_FS, '.-', label='avg FS freq', color=color_palette['FS'])
    ax.plot(time_bins, mean_rate_RS, '.-', label='avg RS freq', color=color_palette['RS'])
    
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
        color=color_palette['RS'], alpha=0.3, label='± RS std'
    )
    
    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(MultipleLocator(1))    
    ax.set_ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    ax.set_title('Network Population firing rate ± std')

    
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
                        inh_intervals = None):
    
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
                              inh_intervals = None):

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
    plt.plot(time_bins, mean_rate_RS, '.-', label='avg RS freq', color=color_palette['RS'])
    
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
        color=color_palette['RS'], alpha=0.3, label='± RS std'
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
                                     input_interval = None):

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
        
    plt.imshow(data_array, aspect='auto', origin='lower',
               extent=[exc_vals[0], exc_vals[-1], inh_vals[0], inh_vals[-1]],
               cmap=cmap_color)
    
    plt.colorbar(label='Output frequency (Hz)')
    plt.xlabel('Freq Excitatory synapses (Hz)')
    plt.ylabel('Freq Inhibitory synapses (Hz)')
    plt.title(f'{neuron_type} input–output relation')
    
    return fig
    
# -------------------- #
def plot_InOut_relation(exc_vals, inh_vals, data_array, neuron_type):
    
    fig = plt.figure(figsize=(10,6))
    
    if neuron_type == 'FS':
        cmap_color = 'autumn_r'
    if neuron_type == 'RS':
        cmap_color = 'summer'
    
    cmap = cm.get_cmap(cmap_color, len(inh_vals))

    for i in range(len(inh_vals)):
        if i % 5 == 0:
            plt.plot(exc_vals, data_array[i], '-o', color=cmap(i), label=f'{inh_vals[i]}')
        else:
            plt.plot(exc_vals, data_array[i], '-o', color=cmap(i))

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
        
    X, Y = np.meshgrid(exc_vals, inh_vals)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data_array, cmap = cmap_color)
    ax.set_xlabel('Freq Excitatory synapses (Hz)')
    ax.set_ylabel('Freq Inhibitory synapses (Hz)')   
    ax.set_zlabel('Output frequency (Hz)')
    ax.set_title(f'{neuron_type} input–output surface')

    return fig