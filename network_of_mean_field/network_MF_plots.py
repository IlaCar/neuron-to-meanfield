"""Plotting for the network-of-mean-fields simulation. """

from __future__ import annotations

from typing import Any

import numpy as np

# pipeline palette (type -> colour)
TYPE_PALETTE = {
    "FS": "#f46d43",           # orange  (inhibitory, warm)
    "RS": "#225ea5",           # blue    (excitatory, cool)
    "RS_no_adapt": "#41b6c4",  # teal    (excitatory, non-adapting)
}
_LINESTYLES = ["-", "--", ":", "-."]


def _group_by_node(net: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in net["pops"]:
        groups.setdefault(p["node"], []).append(p)
    return groups


def plot_activity(
    out: dict[str, Any],
    net: dict[str, Any],
    palette: dict[str, str] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = False,
):
    """Firing-rate traces, one subplot per node, populations overlaid.

    Parameters
    ----------
    out : dict
        Output of ``simulate_MF_network`` (``rates`` + ``time``).
    net : dict
        Network description (for node grouping and population types).
    palette : dict, optional
        Type -> colour. Defaults to the pipeline palette.
    save_path : str, optional
        If given, the figure is written here (PNG).
    show : bool
        Call ``plt.show()`` at the end.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt

    palette = palette or TYPE_PALETTE
    rates = out["rates"]
    time = out["time"]
    groups = _group_by_node(net)
    nodes = list(groups)

    nrows = int(np.ceil(len(nodes) / ncols))
    figsize = figsize or (6.0 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # per-type line-style counter so same-type traces in a node differ
    for ax, node in zip(axes_flat, nodes):
        ls_count: dict[str, int] = {}
        for p in groups[node]:
            name, ptype = p["name"], p["type"]
            colour = palette.get(ptype, "#666666")
            k = ls_count.get(ptype, 0)
            ls = _LINESTYLES[k % len(_LINESTYLES)]
            ls_count[ptype] = k + 1
            ax.plot(time, rates[name], color=colour, linestyle=ls,
                    linewidth=1.2, label=name)
        ax.set_title(f"Node {node}", fontsize=10, fontweight="bold")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("firing rate (Hz)", fontsize=8)
        ax.set_xlim(time[0], time[-1])
        ax.legend(fontsize=8, ncol=2)

    for ax in axes_flat[len(nodes):]:
        fig.delaxes(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_network(
    net: dict[str, Any],
    palette: dict[str, str] | None = None,
    channel_colours: dict[str, str] | None = None,
    save_path: str | None = None,
    show: bool = False,
):
    """Connectivity graph: nodes coloured by type, edges by channel (exc/inh).
 
    Needs networkx. Populations are laid out per node cluster.
 
    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    import networkx as nx
 
    palette = palette or TYPE_PALETTE
    channel_colours = channel_colours or {"e": "#006837", "i": "#a50026"}
 
    names = net["names"]
    pops = net["pops"]
 
    G = nx.DiGraph()
    for p in pops:
        G.add_node(p["name"], type=p["type"], node=p["node"])
    edge_channel = {}
    for p in pops:
        for (pre_idx, _K, _delay, channel) in p["incoming"]:
            src = names[pre_idx]
            G.add_edge(src, p["name"])
            edge_channel[(src, p["name"])] = channel
 
    # cluster layout: one circle of populations per node
    groups = _group_by_node(net)
    cluster_names = list(groups)
    cluster_g = nx.complete_graph(len(cluster_names))
    cluster_pos = nx.spring_layout(cluster_g, scale=6.0, seed=1)
    pos = {}
    for k, node in enumerate(cluster_names):
        sub = G.subgraph([p["name"] for p in groups[node]])
        pos.update(nx.circular_layout(sub, center=cluster_pos[k], scale=2.0))
 
    node_colours = [palette.get(G.nodes[n]["type"], "#666666") for n in G.nodes]
    edge_colours = [channel_colours[edge_channel[e]] for e in G.edges]
 
    node_size = 900
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_colours)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=edge_colours,
        node_size=node_size,          # stop edges at the node boundary
        arrows=True, arrowstyle="-|>", arrowsize=20,
        width=1.6, min_target_margin=12,   # leave the arrowhead clear of the target node
        connectionstyle="arc3,rad=0.16",   # curve reciprocal edges apart
    )
    # legends
    from matplotlib.lines import Line2D
    type_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                           markersize=10, label=t) for t, c in palette.items()
                    if t in {p["type"] for p in pops}]
    chan_handles = [Line2D([0], [0], color=channel_colours["e"], label="excitatory"),
                    Line2D([0], [0], color=channel_colours["i"], label="inhibitory")]
    ax.legend(handles=type_handles + chan_handles, fontsize=8, loc="upper right")
    ax.set_title("Network connectivity", fontsize=11)
    ax.axis("off")
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

