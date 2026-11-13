"""Transfer-function data extraction for one population of a network, driven by the network config.

The population's neuron model, its synaptic channels and the number of inputs per channel are read from the network config (e.g. ``config/network_config_4_pops.json``); the sweep ranges and the sampling come from the TF sweep config (e.g. ``config/input_config_TF_4_pops.json``).

Each channel k the population receives in the network is one sweep dimension. When several sources project onto the same channel, they are merged: the channel's conductance mean and variance depend only on sum_src K_src * nu_src, so the sweep variable is

    nu_eff_k = sum_src K_src * nu_src / K_k,      K_k = sum_src K_src,  K_src = N_src * conn_prob_src.

In the simulation every source keeps its own Poisson group (N_src neurons, connection probability conn_prob_src), and all sources of channel k fire at nu_eff_k. The in-degree of each channel is then distributed as in the network (sum of binomials). There is no separate background: in the network every input of the population is part of nu_eff.

For every input sample a population of ``N_cell`` disconnected neurons is simulated; the output rate is averaged over [p_start + delay, p_end) with ``extracting_single_pop_freq_and_std_window``.

Output (``<out_folder>/TF_data_<neuron_model>.dat``)
    columns: input_<k> for every swept channel, avg_f_out, std_f_out
    Rows are appended as each simulation finishes, so an interrupted sweep can be resumed with ``--resume`` (the input samples are deterministic; the Poisson realisations after the resume point are not identical to an uninterrupted run).

Sources and K per channel come from ``ntmf.network.channel_in_degrees(by_source=True)``; the network config is checked with ``check_network_config`` before the sweep. The ``_meta.json`` written next to it carries K, the sources, E, Q and tau of every channel, read by ``adding_K_params_general`` in the fitting notebook.

Usage (from transfer_function/data_extraction/):
    python TF_data_extraction_network.py --population SST
    python TF_data_extraction_network.py --population RS --network-config ../../config/network_config_4_pops.json
"""

import argparse
import json
import os
import sys
import time

# Adding the project folder to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_dir)

import brian2 as b2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ntmf.config import get_syn_info_general
from ntmf.neurons import create_AdEx_neuron, synaptic_channels_from_model
from ntmf.network import (check_network_config,
                          extracting_single_pop_freq_and_std_window,
                          get_TF_sweep_config_network,
                          load_neuron_model)
from utils.Plots_helper import (disconnected_network_raster_plot_TF,
                                plotting_3_traces,
                                plotting_single_pop_freq_and_std)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--population', required=True,
                    help='population name in the network config (e.g. RS, FS, RS_no_adapt, SST)')
parser.add_argument('--network-config', default=None,
                    help='network config; default: "network_config" entry of the sweep config')
parser.add_argument('--sweep-json', default=os.path.join(project_dir, 'config', 'input_config_TF_4_pops.json'))
parser.add_argument('--neuron-folder', default=os.path.join(project_dir, 'neuron_models', 'AdEx'))
parser.add_argument('--out-folder', default=None,
                    help='default: simulations/TF_<neuron_model> next to this script')
parser.add_argument('--plot-every', type=int, default=10000,
                    help='save raster/trace/rate figures every N simulations (default: effectively never)')
parser.add_argument('--max-runs', type=int, default=None,
                    help='stop after this many simulations (for testing)')
parser.add_argument('--resume', action='store_true',
                    help='append to an existing .dat, skipping samples already simulated')
args = parser.parse_args()

population = args.population
idx = 0  # only one parameter set per cell type

# Neuron model of the population, from the network config named in the sweep config
with open(args.sweep_json, 'r') as fh:
    sweep_cfg = json.load(fh)[0]
network_config_file = args.network_config or os.path.join(os.path.dirname(os.path.abspath(args.sweep_json)),
                                                          sweep_cfg['network_config'])
with open(network_config_file, 'r') as fh:
    net_cfg = json.load(fh)[0]
check_network_config(net_cfg, args.neuron_folder, idx=idx)
if population not in net_cfg['populations']:
    raise ValueError(f"Population '{population}' not in {network_config_file} ({list(net_cfg['populations'])}).")
neuron_model = net_cfg['populations'][population]['neuron_model']
neuron_json = os.path.join(args.neuron_folder, neuron_model + '.json')

model_params = load_neuron_model(neuron_model, args.neuron_folder, idx)['model']
channel_names = [c['name'] for c in synaptic_channels_from_model(model_params)]
Q = get_syn_info_general(neuron_json, idx=idx)   # {"1": Q_1, ...} in nS

sweep = get_TF_sweep_config_network(args.sweep_json, population, channel_names, network_config=net_cfg)
channels = sweep['channels']
swept = sweep['swept']
samples = sweep['samples']
num_sim = len(samples)

simulation_folder = args.out_folder or os.path.join(script_dir, 'simulations', f'TF_{neuron_model}')
os.makedirs(simulation_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation protocol (same as TF_data_extraction_neuron_4_inputs.py, from the sweep config)
# ---------------------------------------------------------------------------
protocol = sweep['protocol']
seed = int(protocol.get('seed', 12345))
b2.start_scope()
b2.seed(seed)

sim_duration = protocol['sim_duration'] * b2.second
dt = protocol['dt'] * b2.ms
b2.defaultclock.dt = dt
times = b2.arange(0, sim_duration, dt)

p_start = protocol['p_start'] * b2.second   # stimulation onset
p_end = protocol['p_end'] * b2.second       # stimulation offset
delay = sweep['delay'] * b2.second          # discarded transient (adaptation)
bin_size = float(protocol['bin_size'])      # seconds
input_interval = [(p_start / b2.second, p_end / b2.second)]
N_cell = int(protocol['N_cell'])

if p_start + delay >= p_end:
    raise ValueError('p_start + delay must be smaller than p_end.')

# No injected current: zero TimedArray, as in the 2-channel script
current_timed_array = b2.TimedArray(np.zeros(int(sim_duration / b2.ms) + 2) * b2.amp, dt=1. * b2.ms)

print(f'Population: {population} -- neuron model: {neuron_model} -- network config: {network_config_file}')
for k in swept:
    c = channels[k]
    src_txt = ' + '.join(f"{s['source']} (K {s['K']:g})" for s in c['sources'])
    print(f"  channel {k}: E = {model_params[f'E_{k}']} mV, Q = {Q[k]}, tau = {model_params[f'tau_{k}']} ms, "
          f"K = {c['K']:g} = {src_txt}, sweep nu_eff {c['min']:g}-{c['max']:g} Hz")
for k in sweep['unused']:
    print(f'  channel {k}: no input in the network, held at 0 Hz (not swept)')
print(f"Sampling: {sweep['sampling'].get('mode', 'grid')} -- number of simulations: {num_sim} -- "
      f"window [{float((p_start + delay) / b2.second):g}, {float(p_end / b2.second):g}) s")

# ---------------------------------------------------------------------------
# Output file (written incrementally)
# ---------------------------------------------------------------------------
dat_file = os.path.join(simulation_folder, f'TF_data_{neuron_model}.dat')
header = ' '.join([f'input_{k}' for k in swept] + ['avg_f_out', 'std_f_out'])

n_done = 0
if args.resume and os.path.exists(dat_file):
    with open(dat_file, 'r') as fh:
        if fh.readline().split() != header.split():
            raise RuntimeError(f'{dat_file} has different columns than the current sweep; refusing to resume.')
    done = np.loadtxt(dat_file, skiprows=1, ndmin=2)
    n_done = len(done)
    if n_done > num_sim or (n_done and not np.allclose(done[:, :len(swept)], samples[:n_done], atol=1e-4)):
        raise RuntimeError(f'{dat_file} does not match the current sweep; refusing to resume.')
    print(f'Resuming: {n_done}/{num_sim} simulations already in {dat_file}')
else:
    with open(dat_file, 'w') as fh:
        fh.write(header + '\n')

# Metadata needed downstream (fitting): channel parameters, K_k and its sources, protocol
meta = {
    'population': population,
    'neuron_model': neuron_model,
    'neuron_json': os.path.abspath(neuron_json),
    'sweep_json': os.path.abspath(args.sweep_json),
    'network_config': os.path.abspath(network_config_file),
    'input_variable': 'nu_eff = sum_src K_src * nu_src / K (Hz), one per channel',
    'E_L': model_params['E_L'],
    'channels': {k: {**channels[k],
                     'E': model_params[f'E_{k}'], 'Q': model_params[f'Q_{k}'],
                     'tau': model_params[f'tau_{k}']} for k in swept},
    'unused_channels': sweep['unused'],
    'sampling': sweep['sampling'],
    'protocol': {'sim_duration_s': float(sim_duration / b2.second),
                 'p_start_s': float(p_start / b2.second), 'p_end_s': float(p_end / b2.second),
                 'delay_s': float(delay / b2.second), 'bin_size_s': bin_size,
                 'dt_ms': float(dt / b2.ms), 'N_cell': N_cell, 'seed': seed},
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
                           json_file_name=neuron_json,
                           idx=idx,
                           N_cell=N_cell,
                           curr_inj=current_timed_array)

    objects = [G]
    namespace = {'current': current_timed_array}

    for k, rate_k in zip(swept, rates_in):
        namespace[f'Q_{k}'] = Q[k]
        if rate_k <= 0:
            continue

        # nu_eff_k inside [p_start, p_end), zero elsewhere; shared by all sources of channel k
        rate_array = b2.zeros(len(times)) * b2.Hz
        rate_array[stim_mask] = rate_k * b2.Hz
        namespace[f'rate_timed_array_{k}'] = b2.TimedArray(rate_array, dt=dt)

        # One Poisson group per source, with the source's own N and conn_prob
        for s in channels[k]['sources']:
            P_stim = b2.PoissonGroup(s['N'], rates=f'rate_timed_array_{k}(t)')
            S_stim = b2.Synapses(P_stim, G, on_pre=f'Gsyn{k}_post += Q_{k}')
            S_stim.connect(p=s['conn_prob'])
            objects += [P_stim, S_stim]

    mon_spike = b2.SpikeMonitor(G)
    objects.append(mon_spike)

    make_plots = (n_sim % args.plot_every == 0)
    if make_plots:
        mon_v = b2.StateMonitor(G, ['v'], record=True)
        objects.append(mon_v)

    label = ' '.join(f'{k}: {r:g} Hz' for k, r in zip(swept, rates_in))
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
        tag = '_'.join(f'{k}_{r:g}' for k, r in zip(swept, rates_in))

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
