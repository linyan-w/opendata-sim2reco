#!/usr/bin/env python
"""Paper figures of the ensemble (from scripts/ensemble_spread.py -> reports/ensemble/spread.json):
 (bins)     residual of the ensemble mean against the open dataset and the between-member band, for the selected events of
            the MINERvA 101 selection in bins of the reconstructed muon pT and pL;
 (perevent) per-event between-member spread of the muon momentum (relative) and of the vertex z, in bins of the true muon
            angle and pT: where the training data thin out, the members disagree.
Usage: scripts/paper_fig_ensemble.py [--spread reports/ensemble/spread.json] [--out reports/paper_figs]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--spread", default="reports/ensemble/spread.json"); ap.add_argument("--out", default="reports/paper_figs"); a = ap.parse_args()
S = json.load(open(a.spread)); out = pathlib.Path(a.out)
# ---- binned: residual and band for the selected events by reconstructed pT and pL
fig, axs = plots.plt.subplots(1, 2, figsize=(8.4, 3.3))
for ax, key, lab in zip(axs, ("selrec_by_pT", "selrec_by_pL"), (r"reconstructed muon $p_T$ [GeV]", r"reconstructed muon $p_L$ [GeV]")):
    rows = S["tables"][key]["rows"]; x = np.arange(len(rows)); real = np.array([r["real"] for r in rows]); mean = np.array([r["mean"] for r in rows]); ens = np.array([r["ens"] for r in rows]); st = np.array([r["stat"] for r in rows])
    ax.fill_between(x, -100 * ens / real, 100 * ens / real, color=plots.PALETTE["model"], alpha=0.4, lw=0, label="between-member spread")
    ax.errorbar(x, 100 * (mean - real) / real, 100 * st / real, fmt="o", ms=3.5, color=plots.PALETTE["real"], label="ensemble mean $-$ open dataset")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([r["bin"].strip("[)").replace(",", "–") for r in rows], rotation=60, fontsize=6.5); ax.set_xlabel(lab); ax.set_ylabel("relative difference [%]")
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left"); axs[0].text(0.97, 0.05, "(a)", transform=axs[0].transAxes, ha="right"); axs[1].text(0.97, 0.05, "(b)", transform=axs[1].transAxes, ha="right")
plots.save(fig, out / "ensemble_bins.png")
# ---- per event: spread of the muon momentum and the vertex z against the true angle and pT
fig, axs = plots.plt.subplots(1, 2, figsize=(8.8, 3.3)); fig.subplots_adjust(wspace=0.55)
for ax, name, lab, panel in zip(axs, ("true angle [deg]", "true pT [GeV]"), (r"true muon angle [$^\circ$]", r"true muon $p_T$ [GeV]"), ("(a)", "(b)")):
    rows = S["per_event"][name]; x = np.arange(len(rows))
    ax.plot(x, [100 * r["relP"] for r in rows], "o-", color=plots.PALETTE["real"], ms=4, label="muon momentum: between members (left)"); ax.set_ylabel("muon momentum [%]"); ax.set_ylim(0, None)
    ax2 = ax.twinx(); ax2.plot(x, [r["dz"] for r in rows], "s-", color=plots.PALETTE["model"], ms=4, label="vertex $z$: between members (right)"); ax2.set_ylabel("vertex $z$ [mm]"); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.set_ylim(0, None)
    if "resP" in rows[0]:  # the surrogate's own stochastic resolution per event (two draws of the nominal member): the statistical part, which does not grow
        ax.plot(x, [100 * r["resP"] for r in rows], "o:", color=plots.PALETTE["real"], mfc="white", ms=4, label="muon momentum: resolution of one model (left)")
        ax2.plot(x, [r["resz"] for r in rows], "s:", color=plots.PALETTE["model"], mfc="white", ms=4, label="vertex $z$: resolution of one model (right)")
    ax.set_xticks(x); ax.set_xticklabels([r["bin"].strip("[)").replace(",", "–") for r in rows], rotation=60, fontsize=6.5); ax.set_xlabel(lab); ax.text(0.03, 0.95, panel, transform=ax.transAxes, va="top")
    if name == "true angle [deg]":
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper left")
plots.save(fig, out / "ensemble_perevent.png"); print("written", out / "ensemble_bins.png", out / "ensemble_perevent.png")
# ---- pulls of the marginal bin fractions without and with the between-member spread in the denominator
KEYS = {"marg_recoil": "recoil energy", "marg_muP": "muon momentum", "marg_theta": "muon angle", "marg_vz": "vertex $z$"}; COL = [plots.PALETTE["real"], plots.PALETTE["model"], plots.PALETTE["third"], "0.6"]
fig, ax = plots.plt.subplots(figsize=(4.4, 3.3)); b = np.linspace(-5, 5, 26); xx = np.linspace(-5, 5, 300); P = {}; P0 = {}
for k in KEYS:
    rows = S["tables"][k]["rows"]; res = np.array([r["resid"] for r in rows]); st2 = np.array([r["stat"] for r in rows]) ** 2 + np.array([r["samp"] for r in rows]) ** 2
    P[k] = res / np.sqrt(st2 + np.array([r["ens"] for r in rows]) ** 2); P0[k] = res / np.sqrt(st2)
allp = np.concatenate(list(P.values())); all0 = np.concatenate(list(P0.values()))
ax.hist([np.clip(P[k], -4.99, 4.99) for k in KEYS], b, stacked=True, label=list(KEYS.values()), color=COL, alpha=0.8)
ax.plot(xx, len(allp) * (b[1] - b[0]) * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1.2, label="unit Gaussian")
ax.set_xlabel("pull"); ax.set_ylabel("bins"); ax.set_title(f"RMS {np.sqrt(np.mean(allp ** 2)):.2f}, {100 * np.mean(np.abs(allp) < 2):.0f}% within 2  (without the spread: RMS {np.sqrt(np.mean(all0 ** 2)):.2f})", fontsize=8, loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper left"); plots.save(fig, out / "ensemble_pulls.png"); print("written", out / "ensemble_pulls.png")
