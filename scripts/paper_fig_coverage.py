#!/usr/bin/env python
"""Paper figure: is the held-out mode reachable by reweighting?  For each interaction mode, the cumulative fraction
of its events against the density ratio (all other modes) / (mode) in the surrogate's input space, from the
classifier odds of scripts/coverage_by_process.py.  The inverse of the ratio is the weight a reweighting of the
mode-free sample would need at that event.
Usage: scripts/paper_fig_coverage.py reports/coverage_2p2h/by_process_ratios.npz OUT.png"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
npz = np.load(sys.argv[1]); out = pathlib.Path(sys.argv[2])
STY = {"2p2h/MEC": (plots.PALETTE["real"], "-", 2.0, "2p2h (held out)"), "QE": (plots.PALETTE["model"], "--", 1.3, "QE"), "RES": (plots.PALETTE["third"], ":", 1.3, "RES"), "DIS": ("0.5", "-.", 1.3, "DIS")}
fig, ax = plots.plt.subplots(figsize=(4.6, 3.4)); xs = np.logspace(-4, 1, 400)
for k, (c, ls, lw, lab) in STY.items():
    r = np.sort(npz[k]); ax.plot(xs, np.searchsorted(r, xs) / len(r), color=c, ls=ls, lw=lw, label=lab)
prior = float(npz["prior_2p2h/MEC"]); req = prior / (1 - prior)  # below this ratio the mode-free sample has fewer events than the mode itself
ax.axvline(req, color=plots.PALETTE["real"], lw=0.8, alpha=0.6); ax.text(req * 1.15, 0.03, "2p2h-free sample has\nfewer events than 2p2h", fontsize=6.5, color=plots.PALETTE["real"], va="bottom")
for v in (0.1, 0.01): ax.axvline(v, color="0.7", lw=0.6, ls="--")
ax.set_xscale("log"); ax.set_xlim(1e-4, 10); ax.set_ylim(0, 1.02); ax.set_xlabel("density of all other modes relative to the mode, at the event")
ax.set_ylabel("cumulative fraction of the mode's events"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
top = ax.secondary_xaxis("top", functions=(lambda x: 1 / x, lambda x: 1 / x)); top.set_xlabel("weight a reweighting of the mode-free sample would need", fontsize=8)
plots.save(fig, out)
r2 = npz["2p2h/MEC"]; print(f"2p2h: below 0.1 {100*(r2<0.1).mean():.1f}%, below 0.01 {100*(r2<0.01).mean():.1f}%, below equal-count ratio {req:.3f}: {100*(r2<req).mean():.1f}%")
L = {k: npz[k] for k in STY}; share = {k: float(npz[f"prior_{k}"]) for k in STY}
mac = {"nCovTwoPLtTenth": f"{100*(L['2p2h/MEC']<0.1).mean():.0f}\\%", "nCovTwoPLtHundredth": f"{100*(L['2p2h/MEC']<0.01).mean():.0f}\\%", "nCovTwoPLtEqual": f"{100*(L['2p2h/MEC']<req).mean():.0f}\\%",
       "nCovRESLtTenth": f"{100*(L['RES']<0.1).mean():.0f}\\%", "nCovQELtTenth": f"{100*(L['QE']<0.1).mean():.0f}\\%", "nCovDISLtTenth": f"{100*(L['DIS']<0.1).mean():.0f}\\%", "nCovRestOverTwoP": f"{(1-prior)/prior:.0f}"}
import json; mac["nCovTwoPAUC"] = f"{json.load(open(pathlib.Path(sys.argv[1]).parent / 'by_process.json'))['2p2h/MEC']['auc']:.3f}"
open(pathlib.Path(sys.argv[1]).parent / "macros.tex", "w").write("% reachability of the held-out mode by reweighting (scripts/paper_fig_coverage.py)\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
