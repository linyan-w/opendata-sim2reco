#!/usr/bin/env python
"""Paper figure: pulls of the marginal bin fractions of the non-calorimetry observables (recoil energy, muon momentum
response, muon angle, prong count) against the open dataset on test events, with the surrogate's own uncertainty
only (epistemic + surrogate sampling + open-dataset statistics; no model-error floor).
Usage: scripts/paper_fig_pulls.py reports/bayes_1A_z OUT.png [--model A] [--draws 8]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("test_dir"); ap.add_argument("out"); ap.add_argument("--model", default="A"); a = ap.parse_args()
KEYS = {"marg_recoil": "recoil energy", "marg_muP": "muon momentum", "marg_dthx": "muon angle", "marg_nprong": "prong count"}
S = json.load(open(pathlib.Path(a.test_dir) / f"bayes_uncertainty_{a.model}.json"))["binned"]["control"]
P = {k: np.array(S[k]["pull"], float) for k in KEYS}; P = {k: v[np.isfinite(v)] for k, v in P.items()}; allp = np.concatenate(list(P.values()))
fig, ax = plots.plt.subplots(figsize=(4.4, 3.3)); b = np.linspace(-5, 5, 26)
ax.hist([np.clip(P[k], -4.99, 4.99) for k in KEYS], b, stacked=True, label=list(KEYS.values()), color=[plots.PALETTE["real"], plots.PALETTE["model"], plots.PALETTE["third"], "0.6"], alpha=0.8)
xx = np.linspace(-5, 5, 300); ax.plot(xx, len(allp) * (b[1] - b[0]) * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1.2, label="unit Gaussian")
ax.set_xlabel("pull"); ax.set_ylabel("bins"); ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title(f"RMS {np.sqrt(np.mean(allp ** 2)):.2f}, {100 * np.mean(np.abs(allp) < 2):.0f}% within 2, {len(allp)} bins", fontsize=8.5, loc="right")
plots.save(fig, pathlib.Path(a.out)); print(f"RMS {np.sqrt(np.mean(allp**2)):.3f} within2 {np.mean(np.abs(allp)<2):.3f} n {len(allp)}")
