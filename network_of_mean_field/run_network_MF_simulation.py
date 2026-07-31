"""Run a network of mean fields (first order) directly from a JSON topology.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import os, sys
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..'))
sys.path.append(parent_dir)

from ntmf.network import generate_ou_process
from ntmf_network_helper import build_network_from_json, simulate_MF_network


def main(
    json_path: str,
    tau_f: float = 15e-3,
    dt: float = 1e-4,
    t_end: float = 3.0,
    signal_targets: tuple[str, ...] = ("RS1",),
    signal_amp: float = 3.0,
    baseline_offset: float = 0.4,
) -> dict:
    time = np.arange(0.0, t_end + 2 * dt, dt)
    n = len(time)

    net = build_network_from_json(json_path, dt=dt)

    # external-projection properties come straight from the network
    ext_number = {p["name"]: p["N"] for p in net["pops"]}
    ext_prob = {name: 0.05 for name in net["names"]}
    ext_receptor = {name: "Glutamate" for name in net["names"]}

    # baseline OU background for all; a square pulse added to the signal targets
    ou = generate_ou_process(time, dt, mu=0.0, tau=5e-3, sigma=1.0, x0=0.0) + baseline_offset
    pulse = np.zeros(n)
    pulse[int(1.0 / dt):int(1.5 / dt)] += signal_amp
    targets = set(signal_targets)
    external_stimulus = {name: (ou + (pulse if name in targets else 0.0)) for name in net["names"]}

    out = simulate_MF_network(
        net, time, external_stimulus,
        external_number=ext_number, external_prob=ext_prob,
        external_receptor=ext_receptor, tau_f=tau_f, order=1,
    )
    out["net"] = net
    return out


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base, "network_3_nodes.json")
    network_name = Path(json_path).stem

    out = main(json_path)
    for name, r in out["rates"].items():
        print(f"{name:6s} final rate = {r[-1]:8.4f} Hz   (min {r.min():.3f}, max {r.max():.3f})")

    try:
        from network_MF_plots import plot_activity, plot_network
        plot_activity(out, out["net"], save_path=os.path.join(base, f"activity_{network_name}.png"))
        print("wrote activity.png")
        try:
            plot_network(out["net"], save_path=os.path.join(base, f"network_{network_name}.png"))
            print("wrote network.png")
        except ImportError:
            print("skipped network.png (networkx not installed)")
    except ImportError:
        print("skipped figures")
