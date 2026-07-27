"""
regime_of_validity.py
---------------------
Manuscript 'regime of validity' figure for the RS mean field.

Three panels over the (exc, inh) input grid:
  1. closed-loop  MF / NN          (full MF vs network; what the reader sees)
  2. open-loop    TF@op-point / NN (TF accuracy alone, loop amplification removed)
  3. sigma_V (mV) at the operating point

Good / less-good is defined on the CLOSED-LOOP relative error in tiers:
  <=15%  quantitative agreement   (filled markers)
  <=50%  qualitative agreement    (open markers)
  >50% / runaway  breakdown       (no marker)
The exc = inh diagonal is drawn as the physical E/I-balance reference.

Set the block below for the baked-in case or the wad case.
"""
import os
import sys
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
from ntmf.config import get_params_model_SI, get_network_config, adding_K_params      # noqa: E402
from ntmf.transfer_function import TF_template_sim, membrane_potential_fluctuations_sim  # noqa: E402

# ============================ configure =============================
# baked-in:  TF='10_best_params_TF_RS.json', W_MODE='zero',
#            MF='simulations/val_MF_[0_30_5]', NN='simulations/val_NN_[0_30_5]', TAG='zero'
# wad:       TF='10_best_params_TF_RS_wad.json', W_MODE='measured',
#            MF='simulations/val_MF_wad_[0_30_5]', NN='simulations/val_NN_wad_[0_30_5]', TAG='wad'
TF_FILE = '10_best_params_TF_RS.json'
W_MODE  = 'zero'
MF_dir  = 'simulations/val_MF_[0_30_1]'
NN_dir  = 'simulations/val_NN_[0_30_1]'
TAG     = 'zero'

#TF_FILE = '10_best_params_TF_RS_wad.json'
#W_MODE  = 'measured'
#MF_dir  = 'simulations/val_MF_wad_[0_30_5]'
#NN_dir  = 'simulations/val_NN_wad_[0_30_5]'
#TAG     = 'wad'

TIER_Q  = 0.15     # quantitative-agreement threshold
TIER_L  = 0.50     # qualitative-agreement threshold
cfg     = '../../config/network_config_file_val_heatmap.json'
RS_json = '../../neuron_models/AdEx/RS.json'
grid    = np.arange(0, 31, 2)
# ====================================================================

net = get_network_config(json_file_name=cfg)
params = adding_K_params(get_params_model_SI('RS', RS_json), net)
d = json.load(open(TF_FILE))[0]
poly, alpha = d['polynomial_params'], d['alpha']

n = len(grid)
mf = np.full((n, n), np.nan)
nn = np.full((n, n), np.nan)
tf = np.full((n, n), np.nan)
sig = np.full((n, n), np.nan)


def load_stats(folder, exc, inh):
    fn = os.path.join(folder, f'sim_data_exc_{exc}_inh_{inh}.h5')
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, 'r') as f:
        s = f['stats']
        return {k: float(s[k][()]) for k in s.keys()}


for r, inh in enumerate(grid):
    for c, exc in enumerate(grid):
        nnf = load_stats(NN_dir, exc, inh)
        mff = load_stats(MF_dir, exc, inh)
        if nnf is None or mff is None:
            continue
        nu_FS = nnf['FS_avg_freq']
        nu_RS = nnf['RS_avg_freq']
        w = nnf['RS_avg_w_pA'] * 1e-12 if W_MODE == 'measured' else 0.0
        nn[r, c] = nu_RS
        mf[r, c] = mff['RS_avg_freq']
        f_e, f_i = exc + nu_RS, inh + nu_FS
        tf[r, c] = TF_template_sim(f_e, f_i, params, poly, alpha, w_ad=w)
        _, sV, _, _ = membrane_potential_fluctuations_sim(f_e, f_i, params, w_ad=w)
        sig[r, c] = sV * 1e3

nn_mask = np.where(nn < 0.5, np.nan, nn)
ratio_closed = mf / nn_mask
ratio_open = tf / nn_mask
rel = np.abs(ratio_closed - 1.0)                      # closed-loop relative error

X, Y = np.meshgrid(grid, grid)
quant = np.isfinite(rel) & (rel <= TIER_Q)
qual = np.isfinite(rel) & (rel > TIER_Q) & (rel <= TIER_L)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
panels = [
    (ratio_closed, 'closed-loop  MF / NN', 'PRGn_r', dict(vmin=0, vmax=2)),
    (ratio_open,  'open-loop  TF@op / NN', 'PRGn_r', dict(vmin=0, vmax=2)),
    (sig,         r'$\sigma_V$ at op-point (mV)', 'viridis', dict()),
]
for j, (ax, (M, title, cmap, kw)) in enumerate(zip(axes, panels)):
    pc = ax.pcolormesh(X, Y, M, cmap=cmap, shading='nearest', **kw)
    fig.colorbar(pc, ax=ax, label=title)
    ax.plot([0, 30], [0, 30], color='0.4', ls='--', lw=1)          # exc = inh reference
    if j < 2:  # good / less-good markers on the ratio panels
        ax.scatter(X[quant], Y[quant], s=42, facecolor='white',
                   edgecolor='k', linewidth=1.1, label=f'<={int(TIER_Q*100)}%')
        ax.scatter(X[qual], Y[qual], s=40, marker='o', facecolors='grey',
                   edgecolors='grey', linewidths=1.1, label=f'≤{int(TIER_L * 100)}%', zorder=3)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.6)
    ax.set_xlabel('Input Freq Exc (Hz)')
    ax.set_ylabel('Input Freq Inh (Hz)')
    ax.set_title(title)

fig.suptitle(f'RS mean-field regime of validity ({TAG})  '
             f'-- filled: quantitative (<={int(TIER_Q*100)}%), '
             f'open: qualitative (<={int(TIER_L*100)}%)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs('figures', exist_ok=True)
out = f'figures/regime_of_validity_RS_{TAG}.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
print(f'saved {out} (+ .svg)')

# ---- tier counts + per-cell relative-error table ----
tot = int(np.sum(np.isfinite(rel)))
print(f"\n[RS {TAG}] of {tot} scored points (NN>=0.5 Hz):")
print(f"   quantitative (<={int(TIER_Q*100)}%): {int(np.sum(quant))}")
print(f"   qualitative  (<={int(TIER_L*100)}%): {int(np.sum(quant | qual))}")
print(f"   breakdown    (> {int(TIER_L*100)}%): {tot - int(np.sum(quant | qual))}")
print(f"   median |MF/NN| = {np.nanmedian(ratio_closed):.2f}\n")

print("closed-loop relative error (%)   exc across, inh down   '.'=NN<0.5")
print("exc:  " + "  ".join(f"{e:4d}" for e in grid))
for r, inh in enumerate(grid):
    row = []
    for c in range(n):
        row.append(" .  " if not np.isfinite(rel[r, c]) else f"{rel[r,c]*100:4.0f}")
    print(f"inh{inh:3d}: " + " ".join(row))
