"""Transfer-function data extraction for an AdEx neuron with N synaptic channels.

Instead of an (inh, exc) grid, each synaptic channel k = 1..N of the neuron JSON (E_k, Q_k, tau_k) receives its own Poisson drive, and the sweep runs over the N-dimensional input space defined in the TF sweep config.

For every input sample a population of ``N_cell`` disconnected neurons is simulated; the output rate is averaged over [p_start + delay, p_end).

Output (``<out_folder>/TF_data_<neuron_model>.dat``)
    columns: input_1 ... input_N avg_f_out std_f_out
    Rows are appended as each simulation finishes, so an interrupted sweep can be resumed with ``--resume`` (the input samples are deterministic; the Poisson realisations after the resume point are not identical to an uninterrupted run).

"""

import argparse
import json
import os
import sys
import time

# Adding the project folder to sys.path
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..', '..'))
sys.path.append(parent_dir)

import brian2 as b2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ntmf.config import get_TF_sweep_config, get_syn_info_general
from ntmf.neurons import create_AdEx_neuron, synaptic_channels_from_model
from ntmf.network import extracting_single_pop_freq_and_std_window
from utils.Plots_helper import (disconnected_network_raster_plot_TF,
                                plotting_3_traces,
                                plotting_single_pop_freq_and_std)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
neuron_model = 'neuron_4_inputs'
data_folder = '../../neuron_models/AdEx'
neuron_json = os.path.join(data_folder, neuron_model + '.json')
sweep_json = '../../config/input_config_TF_4_channels.json'
idx = 0  # only one parameter set per cell type

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--neuron-json', default=neuron_json)
parser.add_argument('--sweep-json', default=sweep_json)
parser.add_argument('--out-folder', default=f'TF_{neuron_model}_delay_v1')
parser.add_argument('--plot-every', type=int, default=10000,
                    help='save raster/trace/rate figures every N simulations (default: effectively never)')
parser.add_argument('--max-runs', type=int, default=None,
                    help='stop after this many simulations (for testing)')
parser.add_argument('--resume', action='store_true',
                    help='append to an existing .dat, skipping samples already simulated')
args = parser.parse_args()

simulation_folder = args.out_folder
os.makedirs(simulation_folder, exist_ok=True)

b2.start_scope()
b2.seed(12345)

# Simulation protocol (same as TF_data_extraction_RS.py)
sim_duration = 7 * b2.second
dt = 0.1 * b2.ms
b2.defaultclock.dt = dt
times = b2.arange(0, sim_duration, dt)

p_start = 2 * b2.second   # stimulation onset
p_end = 5 * b2.second     # stimulation offset
delay = 1 * b2.second     # discarded transient (adaptation); 0 s for non-adapting cells
bin_size = 0.25           # seconds
input_interval = [(p_start / b2.second, p_end / b2.second)]

N_cell = 5

# No injected current: zero TimedArray, as in the 2-channel script
current_timed_array = b2.TimedArray(np.zeros(int(sim_duration / b2.ms) + 2) * b2.amp, dt=1. * b2.ms)

# ---------------------------------------------------------------------------
# Channels and sweep
# ---------------------------------------------------------------------------
with open(args.neuron_json, 'r') as fh:
    model_params = json.load(fh)[0][idx]['model']
channel_names = [c['name'] for c in synaptic_channels_from_model(model_params)]
Q = get_syn_info_general(args.neuron_json, idx=idx)   # {"1": Q_1, ...} in nS

sweep = get_TF_sweep_config(args.sweep_json, channel_names)
channels = sweep['channels']
samples = sweep['samples']
num_sim = len(samples)

print(f'Neuron model: {neuron_model} -- channels: {channel_names}')
for k in channel_names:
    c = channels[k]
    print(f"  channel {k}: E = {model_params[f'E_{k}']} mV, Q = {Q[k]}, tau = {model_params[f'tau_{k}']} ms, "
          f"K = {c['N_external'] * c['conn_prob']:.1f}, background = {c['background_freq']} Hz, "
          f"sweep {c['min']}-{c['max']} Hz")
print(f"Sampling: {sweep['sampling'].get('mode', 'grid')} -- number of simulations: {num_sim}")

# ---------------------------------------------------------------------------
# Output file (written incrementally)
# ---------------------------------------------------------------------------
dat_file = os.path.join(simulation_folder, f'TF_data_{neuron_model}.dat')
header = ' '.join([f'input_{k}' for k in channel_names] + ['avg_f_out', 'std_f_out'])

n_done = 0
if args.resume and os.path.exists(dat_file):
    done = np.loadtxt(dat_file, skiprows=1, ndmin=2)
    n_done = len(done)
    if n_done and not np.allclose(done[:, :len(channel_names)], samples[:n_done], atol=1e-4):
        raise RuntimeError(f'{dat_file} does not match the current sweep; refusing to resume.')
    print(f'Resuming: {n_done}/{num_sim} simulations already in {dat_file}')
else:
    with open(dat_file, 'w') as fh:
        fh.write(header + '\n')

# Metadata needed downstream (fitting): channel parameters, K_k, protocol
meta = {
    'neuron_model': neuron_model,
    'neuron_json': os.path.abspath(args.neuron_json),
    'sweep_json': os.path.abspath(args.sweep_json),
    'E_L': model_params['E_L'],
    'channels': {k: {**channels[k],
                     'K': channels[k]['N_external'] * channels[k]['conn_prob'],
                     'E': model_params[f'E_{k}'], 'Q': model_params[f'Q_{k}'],
                     'tau': model_params[f'tau_{k}']} for k in channel_names},
    'sampling': sweep['sampling'],
    'protocol': {'sim_duration_s': float(sim_duration / b2.second),
                 'p_start_s': float(p_start / b2.second), 'p_end_s': float(p_end / b2.second),
                 'delay_s': float(delay / b2.second), 'bin_size_s': bin_size,
                 'dt_ms': float(dt / b2.ms), 'N_cell': N_cell, 'seed': 12345},
    'units': {'E': 'mV', 'Q': 'nS', 'tau': 'ms', 'rates': 'Hz'},
}
with open(os.path.join(simulation_folder, f'TF_data_{neuron_model}_meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=4)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
stim_mask = (times >= p_start) & (times < p_end)
n_stop = num_sim if args.max_runs is None else min(num_sim, n_done + args.max_runs)
t_wall = time.time()

for n_sim in range(n_done + 1, n_stop + 1):
    rates_in = samples[n_sim - 1]
    b2.start_scope()

    G = create_AdEx_neuron(neuron_model=neuron_model,
                           json_file_name=args.neuron_json,
                           idx=idx,
                           N_cell=N_cell,
                           curr_inj=current_timed_array)

    objects = [G]
    namespace = {'current': current_timed_array}

    for k, rate_k in zip(channel_names, rates_in):
        c = channels[k]
        namespace[f'Q_{k}'] = Q[k]

        # Background: constant over the whole simulation, own Poisson group and connectivity
        if c['background_freq'] > 0:
            P_bg = b2.PoissonGroup(c['N_external'], rates=c['background_freq'] * b2.Hz)
            S_bg = b2.Synapses(P_bg, G, on_pre=f'Gsyn{k}_post += Q_{k}')
            S_bg.connect(p=c['conn_prob'])
            objects += [P_bg, S_bg]

        # Stimulation: rate_k inside [p_start, p_end), zero elsewhere
        if rate_k > 0:
            rate_array = b2.zeros(len(times)) * b2.Hz
            rate_array[stim_mask] = rate_k * b2.Hz
            namespace[f'rate_timed_array_{k}'] = b2.TimedArray(rate_array, dt=dt)
            P_stim = b2.PoissonGroup(c['N_external'], rates=f'rate_timed_array_{k}(t)')
            S_stim = b2.Synapses(P_stim, G, on_pre=f'Gsyn{k}_post += Q_{k}')
            S_stim.connect(p=c['conn_prob'])
            objects += [P_stim, S_stim]

    mon_spike = b2.SpikeMonitor(G)
    objects.append(mon_spike)

    make_plots = (n_sim % args.plot_every == 0)
    if make_plots:
        mon_v = b2.StateMonitor(G, ['v'], record=True)
        objects.append(mon_v)

    label = ' '.join(f'{k}: {r:g} Hz' for k, r in zip(channel_names, rates_in))
    print(f'[{n_sim}/{num_sim}] {label}')

    net = b2.Network(*objects)
    net.run(sim_duration, namespace=namespace)

    mean_rate, std_rate = extracting_single_pop_freq_and_std_window(sim_duration=sim_duration,
                                                                    t_start=p_start + delay,
                                                                    t_end=p_end,
                                                                    pop=mon_spike,
                                                                    N_pop=N_cell,
                                                                    bin_size=bin_size)

    with open(dat_file, 'a') as fh:
        fh.write(' '.join(f'{v:.6g}' for v in (*rates_in, mean_rate, std_rate)) + '\n')

    if make_plots:
        tag = '_'.join(f'{k}_{r:g}' for k, r in zip(channel_names, rates_in))

        fig = disconnected_network_raster_plot_TF(neuron_model=neuron_model, pop=mon_spike, N_pop=N_cell,
                                                  input_interval=input_interval,
                                                  x_lim=(0, sim_duration / b2.second))
        fig.savefig(os.path.join(simulation_folder, f'in_{tag}_raster.png'))
        plt.close(fig)

        fig = plotting_3_traces(neuron_model=neuron_model, pop=mon_v, input_interval=input_interval)
        fig.savefig(os.path.join(simulation_folder, f'in_{tag}_sel_traces.png'))
        plt.close(fig)

        fig = plotting_single_pop_freq_and_std(sim_duration=sim_duration, neuron_model=neuron_model,
                                               pop=mon_spike, N_pop=N_cell, bin_size=bin_size,
                                               input_interval=input_interval)
        fig.savefig(os.path.join(simulation_folder, f'in_{tag}.png'))
        plt.close(fig)

    del net, objects

elapsed = time.time() - t_wall
n_ran = n_stop - n_done
if n_ran:
    print(f'{n_ran} simulations in {elapsed:.0f} s ({elapsed / n_ran:.1f} s each)')
print(f'Data saved in {dat_file}')
print('Done!')
