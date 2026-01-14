import json
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator

import brian2 as b2

# -------------------- #
color_palette = {"FS": '#cb181d',
                 "RS": '#238b45'}

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
                                     ext_input_4 = None):

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(pop1.t/b2.second, pop1.v[0] / b2.mV, color='#67000d')
    ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[0], -50) / b2.mV, '--', color='#67000d')
    ax[0].plot(pop1.t/b2.second, pop1.v[1] / b2.mV, color='#cb181d')
    ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[1], -50) / b2.mV, '--', color='#cb181d')
    ax[0].plot(pop1.t/b2.second, pop1.v[2] / b2.mV, color='#fb6a4a')
    ax[0].plot(pop1.t/b2.second, get_pretty_voltage(pop1.v[2], -50) / b2.mV, '--', color='#fb6a4a')
    ax[0].set_title('Selected FS traces')
    
    ax[1].plot(pop2.t/b2.second, pop2.v[0] / b2.mV, color='#00441b')
    ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[0], -50) / b2.mV, '--', color='#00441b')   
    ax[1].plot(pop2.t/b2.second, pop2.v[1] / b2.mV, color='#238b45')
    ax[1].plot(pop2.t/b2.second, get_pretty_voltage(pop2.v[1], -50) / b2.mV, '--', color='#238b45')
    ax[1].plot(pop2.t/b2.second, pop2.v[2] / b2.mV, color='#74c476')
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
def add_input_boxes(ax,
                    exc_intervals = None,
                    inh_intervals = None,
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
    for t0, t1, pop in intervals:
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

        aligned.append((t0_aligned, t1_aligned, pop))

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