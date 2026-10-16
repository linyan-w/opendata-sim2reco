#!/usr/bin/env python
"""Paper figure: pulls of the marginal bin fractions of the non-calorimetry observables (recoil energy, muon momentum
response, muon angle, prong count) against the open dataset on test events, with the surrogate's own uncertainty
only (epistemic + surrogate sampling + open-dataset statistics; no model-error floor).
Usage: scripts/paper_fig_pulls.py reports/bayes_1A_z OUT.png [--models A] [--split control]
       scripts/paper_fig_pulls.py reports/bayes_1A_z OUT.png --models A B --split heldout   (2p2h test events, one panel per model)"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("test_dir"); ap.add_argument("out"); ap.add_argument("--models", nargs="+", default=["A"]); ap.add_argument("--split", default="control", help="control (non-2p2h test events) or heldout (2p2h test events)"); a = ap.parse_args()
KEYS = {"marg_recoil": "recoil energy", "marg_muP": "muon momentum", "marg_dthx": "muon angle", "marg_nprong": "prong count"}
fig, axs = plots.plt.subplots(1, len(a.models), figsize=(4.4 * len(a.models), 3.3), squeeze=False); b = np.linspace(-5, 5, 26); xx = np.linspace(-5, 5, 300)
for ax, m, lab in zip(axs[0], a.models, "abc"):
    S = json.load(open(pathlib.Path(a.test_dir) / f"bayes_uncertainty_{m}.json"))["binned"][a.split]
    P = {k: np.array(S[k]["pull"], float) for k in KEYS}; P = {k: v[np.isfinite(v)] for k, v in P.items()}; allp = np.concatenate(list(P.values()))
    ax.hist([np.clip(P[k], -4.99, 4.99) for k in KEYS], b, stacked=True, label=list(KEYS.values()), color=[plots.PALETTE["real"], plots.PALETTE["model"], plots.PALETTE["third"], "0.6"], alpha=0.8)
    ax.plot(xx, len(allp) * (b[1] - b[0]) * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1.2, label="unit Gaussian")
    ax.set_xlabel("pull"); ax.set_ylabel("bins"); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title((f"model {m}: " if len(a.models) > 1 else "") + f"RMS {np.sqrt(np.mean(allp ** 2)):.2f}, {100 * np.mean(np.abs(allp) < 2):.0f}% within 2, {len(allp)} bins", fontsize=8.5, loc="right")
    if len(a.models) > 1: ax.text(0.97, 0.95, f"({lab})", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    print(f"{m} {a.split}: RMS {np.sqrt(np.mean(allp**2)):.3f} within2 {np.mean(np.abs(allp)<2):.3f} n {len(allp)}")
plots.save(fig, pathlib.Path(a.out))
