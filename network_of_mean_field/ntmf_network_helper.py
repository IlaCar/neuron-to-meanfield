"""ntmf_network_helper -- glue for running an arbitrary-topology network of
mean fields on top of the existing ntmf pipeline.

Everything here an later be included into the ntmf package (I am still running some tests):

  - :func:`attrdict_to_ntmf_params`  -> convert an embedded cell-parameter block
    into an ntmf-style ``params`` dict. When you move to ntmf JSON neuron
    configs, replace this with ``config.get_params_model_SI``.
  - :func:`build_network_from_json`  -> read a self-contained JSON topology
    (network + edges + cell parameters) into a plain network description.
  - :func:`simulate_MF_network`      -> arbitrary-topology, delay-aware,
    FIRST-ORDER mean-field integrator (generalises ntmf's 2-population
    ``meanfield.simulate_MF_FS_RS``). ``order=1`` is the default and the only
    order implemented for arbitrary topology; the validated second-order
    closure currently lives in ``meanfield.py`` for the 2-population case.

The topology is driven entirely by the JSON file
(no CSV or .npy fit files originally implemented by Pratik are used anymore).

The transfer function itself is NOT re-implemented: every population, every
step, calls ``ntmf.transfer_function.TF_template_sim``. A channel-general TF
(aggregate-input) is reproduced exactly by ntmf's 2-channel TF when the
per-receptor aggregate ``A = sum_pre N_pre * p * strength * nu_pre`` is passed
as ``f_e`` / ``f_i`` with ``K_e = K_i = 1`` in ``params`` (verified to ~1e-14).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from ntmf.transfer_function import (
    TF_template_sim,
    membrane_potential_fluctuations_sim,
)

# Default mapping from receptor name -> ntmf channel. 
# Here more receptor types can be added.
DEFAULT_RECEPTOR_MAP = {"Glutamate": "e", "GABA": "i"}


# ---------------------------------------------------------------------------
# 1. parameter conversion  (cell-parameter block -> ntmf params dict)
# ---------------------------------------------------------------------------

def attrdict_to_ntmf_params(cell_param: Any) -> dict[str, Any]:
    """Convert a cell-parameter object to ntmf conventions.

    Expects the attributes: ``order`` (receptor names), ``Q``, ``T`` (tau_syn
    per channel), ``E``, ``Gl``, ``El``, ``Cm``, the fitted polynomial ``P``,
    the TF prefactor ``alpha``, and (for adaptation) ``Tw``, ``a``, ``b``.
    ``alpha`` defaults to 1.0 if absent.

    Returns
    -------
    dict with keys:
        ``params``      -- ntmf params dict (K_e = K_i = 1; see module docstring)
        ``poly_params`` -- 10 polynomial coefficients
        ``alpha``       -- TF prefactor (from ``cell_param.alpha``, default 1.0)
        ``a``, ``b``, ``tau_w``, ``E_L`` -- adaptation parameters (SI)
        ``has_adapt``   -- True if a != 0 or b != 0
    """
    order = list(cell_param.order)
    try:
        ie = order.index("Glutamate")
        ii = order.index("GABA")
    except ValueError as exc:
        raise ValueError(
            f"attrdict_to_ntmf_params expects 'Glutamate' and 'GABA' in "
            f"cell_param.order, got {order}"
        ) from exc

    T = np.asarray(cell_param.T, dtype=float)
    if not np.isclose(T[ie], T[ii]):
        raise ValueError(
            "ntmf's 2-channel TF uses a single tau_syn; the excitatory and "
            f"inhibitory tau_syn differ ({T[ie]} vs {T[ii]}). Use a "
            "channel-general TF for this case."
        )
    #TODO: expand this
    
    Q = np.asarray(cell_param.Q, dtype=float)
    E = np.asarray(cell_param.E, dtype=float)

    params = {
        "K_e": 1.0,
        "K_i": 1.0,
        "tau_syn": float(T[ie]),
        "Q_e": float(Q[ie]),
        "Q_i": float(Q[ii]),
        "g_L": float(cell_param.Gl),
        "C_m": float(cell_param.Cm),
        "E_e": float(E[ie]),
        "E_i": float(E[ii]),
        "E_L": float(cell_param.El),
    }

    a = float(getattr(cell_param, "a", 0.0))
    b = float(getattr(cell_param, "b", 0.0))
    tau_w = float(getattr(cell_param, "Tw", 1.0))
    alpha = float(getattr(cell_param, "alpha", 1.0))  # explicit TF prefactor

    return {
        "params": params,
        "poly_params": np.asarray(cell_param.P, dtype=float),
        "alpha": alpha,
        "a": a,
        "b": b,
        "tau_w": tau_w,
        "E_L": float(cell_param.El),
        "has_adapt": (a != 0.0) or (b != 0.0),
    }


# ---------------------------------------------------------------------------
# 2. topology loading  (JSON -> plain network description)
# ---------------------------------------------------------------------------
#
# Schema:
#   {
#     "network": [                       # list of populations making up the network
#       {"name": "RS1", "type": "RS", "node": "N1", "N": 8000, "rate": 5},
#       ...
#     ],
#     "edges": [
#       {"pre": "RS1", "post": "FS1", "probability": 0.05,
#        "strength": 1, "delay": 0, "receptor": "Glutamate"},
#       ...
#     ],
#     "cell_parameters": {               # one block per population type
#       "RS": {"P": [...10...], "alpha": 1.0, "order": ["Glutamate", "GABA"],
#              "Q": [1.5e-9, 5e-9], "T": [5e-3, 5e-3], "E": [0, -80e-3],
#              "Cm": 200e-12, "El": -64e-3, "Gl": 10e-9, "Tw": 0.5, "a": 0, "b": 0},
#       "FS": {...}
#     }
#   }
#
# Each population belongs to a node ("node": "N1"/"N2"/...)
# "delay" is in milliseconds. 
#The two receptors are "Glutamate" (excitatory) and "GABA" (inhibitory).

def _assemble_network(
    population_records: list,
    edge_records: list,
    cell_params_by_type: dict,
    dt: float,
    receptor_map: dict | None = None,
) -> dict:
    """Assemble a network description from population/edge records.

    Returns
    -------
    dict
        ``names`` : list[str], population names in order
        ``index`` : {name: row}
        ``pops``  : list of per-population records, each holding the ntmf
        ``params``/``poly``/``alpha``, neuron count, initial rate, adaptation
        parameters, and its incoming edges as tuples
        ``(pre_index, K_syn, delay_steps, channel)`` with
        ``K_syn = N_pre * probability * strength``.
    """
    receptor_map = receptor_map or DEFAULT_RECEPTOR_MAP

    names = [str(r["name"]) for r in population_records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate population name; names must be unique.")
    index = {n: i for i, n in enumerate(names)}

    conv_by_type = {t: attrdict_to_ntmf_params(cp) for t, cp in cell_params_by_type.items()}

    pops = []
    for r in population_records:
        name = str(r["name"])
        ptype = str(r["type"])
        if ptype not in conv_by_type:
            raise KeyError(f"Population '{name}' has type '{ptype}' with no cell parameters.")
        conv = conv_by_type[ptype]
        pops.append({
            "name": name,
            "type": ptype,
            "node": str(r.get("node", "")),
            "N": float(r["N"]),
            "init_rate": float(r["rate"]),
            "params": conv["params"],
            "poly": conv["poly_params"],
            "alpha": conv["alpha"],
            "has_adapt": conv["has_adapt"],
            "a": conv["a"], "b": conv["b"], "tau_w": conv["tau_w"], "E_L": conv["E_L"],
            "incoming": [],
        })

    for e in edge_records:
        pre, post = str(e["pre"]), str(e["post"])
        if pre not in index or post not in index:
            raise KeyError(f"Edge ({pre} -> {post}) references an unknown population.")
        receptor = str(e["receptor"])
        if receptor not in receptor_map:
            raise KeyError(
                f"Receptor '{receptor}' on edge ({pre} -> {post}) is not in "
                f"receptor_map {sorted(receptor_map)}. The 2-channel TF only "
                "supports excitatory/inhibitory; add a channel-general TF to go further."
            )
        channel = receptor_map[receptor]
        prob = float(e["probability"])
        strength = float(e["strength"])
        delay_steps = int((float(e["delay"]) * 1e-3) / dt)
        K_syn = pops[index[pre]]["N"] * prob * strength
        pops[index[post]]["incoming"].append((index[pre], K_syn, delay_steps, channel))

    return {"names": names, "index": index, "pops": pops}


def build_network_from_json(
    json_path: str,
    dt: float,
    cell_params_by_type: dict | None = None,
    receptor_map: dict | None = None,
) -> dict:
    """Read a self-contained JSON topology file into a network description.

    Parameters
    ----------
    json_path : str
        Path to the JSON file (schema documented above).
    dt : float
        Integration step (s); converts edge delays (ms) to step counts.
    cell_params_by_type : dict, optional
        Cell parameters per type. If omitted (the normal case), they are read
        from the file's ``cell_parameters`` block. An explicit argument
        overrides the embedded block.
    receptor_map : dict, optional
        Receptor name -> ntmf channel ("e"/"i").

    """
    with open(json_path, "r") as fh:
        spec = json.load(fh)

    if "network" in spec:
        population_records = spec["network"]
    else:
        raise KeyError("JSON must contain a 'network' key (list of populations).")

    if cell_params_by_type is None:
        if "cell_parameters" not in spec:
            raise ValueError(
                "No cell parameters: include a 'cell_parameters' block in the JSON "
                "or pass cell_params_by_type."
            )
        cell_params_by_type = {
            t: SimpleNamespace(**cp) for t, cp in spec["cell_parameters"].items()
        }

    return _assemble_network(
        population_records, spec["edges"], cell_params_by_type, dt, receptor_map
    )


# ---------------------------------------------------------------------------
# 3. generalized FIRST-ORDER integrator (reuses ntmf's TF)
# ---------------------------------------------------------------------------

def simulate_MF_network(
    net: dict[str, Any],
    time: np.ndarray,
    external_stimulus: dict[str, np.ndarray],
    external_number: dict[str, float],
    external_prob: dict[str, float],
    external_receptor: dict[str, str],
    tau_f: float = 15e-3,
    order: int = 1,
    receptor_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """First-order mean-field simulation over an arbitrary population graph.

    Reuses ``ntmf.transfer_function.TF_template_sim`` for every population and
    step; the only per-population state is the mean rate (and, if the cell type
    has adaptation, the adaptation current).

    Parameters
    ----------
    net : dict
        Output of :func:`build_network_from_json`.
    time : ndarray
        Time vector (s), uniform step. ``dt`` is inferred from it.
    external_stimulus : dict
        ``{pop_name: array}`` giving the external input rate per step
        (length >= len(time)).
    external_number, external_prob, external_receptor : dict
        Per-population external synapse count, connection probability, and
        receptor name (mapped through ``receptor_map``).
    tau_f : float
        Population rate time constant T (s). Default 15 ms.
    order : int
        Only ``1`` is implemented for arbitrary topology (default). ``order=2``
        raises NotImplementedError -- use ``meanfield.simulate_MF_FS_RS(order=2)``
        for the validated 2-population second-order closure.

    Returns
    -------
    dict
        ``rates``       : {pop_name: ndarray}    firing-rate traces (Hz)
        ``adaptation``  : {pop_name: ndarray}    adaptation current (A)
        ``time``        : ndarray
    """
    if order != 1:
        raise NotImplementedError(
            "simulate_MF_network implements order=1 for arbitrary topology. "
            "For second order, use the 2-population closure in "
            "meanfield.simulate_MF_FS_RS(order=2)."
        )
    receptor_map = receptor_map or DEFAULT_RECEPTOR_MAP

    pops = net["pops"]
    index = net["index"]
    n_pop = len(pops)
    n = len(time)
    dt = float(time[1] - time[0])
    T = tau_f

    rates = np.zeros((n_pop, n))
    adapt = np.zeros((n_pop, n))
    for i, p in enumerate(pops):
        rates[i, 0] = p["init_rate"]
        # to match Pratik's initial adaptation: nu0 * b * tau_w
        # TODO: check this
        adapt[i, 0] = p["init_rate"] * p["b"] * p["tau_w"]

    ext_gain = np.array([external_prob[p["name"]] * external_number[p["name"]] for p in pops])
    ext_chan = [receptor_map[external_receptor[p["name"]]] for p in pops]

    for t in range(1, n):
        for i, p in enumerate(pops):
            # ---- aggregate recurrent input per channel ----
            A = {"e": 0.0, "i": 0.0}
            for (pre, K_syn, delay, channel) in p["incoming"]:
                idx = t - 1 - delay
                nu_pre = rates[pre, idx] if idx >= 0 else pops[pre]["init_rate"]
                A[channel] += K_syn * nu_pre
            # ---- external drive ----
            A[ext_chan[i]] += external_stimulus[p["name"]][t - 1] * ext_gain[i]

            w = adapt[i, t - 1]
            F = TF_template_sim(
                f_e=A["e"], f_i=A["i"], params=p["params"],
                poly_params=p["poly"], alpha=p["alpha"], w_ad=w,
            )
            rates[i, t] = rates[i, t - 1] + (dt / T) * (F - rates[i, t - 1])

            # ---- adaptation (only if this cell type adapts) ----
            if p["has_adapt"]:
                mu_V, _, _, _ = membrane_potential_fluctuations_sim(
                    f_e=A["e"], f_i=A["i"], params=p["params"], w_ad=w,
                )
                dW = -w / p["tau_w"] + p["b"] * rates[i, t - 1] + p["a"] * (mu_V - p["E_L"]) / p["tau_w"]
                adapt[i, t] = w + dt * dW
            else:
                adapt[i, t] = w  # stays 0

            if rates[i, t] < 0 or np.isnan(rates[i, t]):
                raise FloatingPointError(
                    f"Invalid rate for '{p['name']}' at step {t}: {rates[i, t]}"
                )

    names = net["names"]
    return {
        "rates": {names[i]: rates[i] for i in range(n_pop)},
        "adaptation": {names[i]: adapt[i] for i in range(n_pop)},
        "time": time,
    }
