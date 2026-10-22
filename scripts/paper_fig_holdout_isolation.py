#!/usr/bin/env python
"""Holdout figure against isolation: on the held-out 2p2h test events, the full model and the 2p2h-blind model relative to
the open dataset in bins of the event's isolation, the density of all other interaction modes relative to 2p2h at the
event in the surrogate's input space (scripts/holdout_isolation.py; small = the 2p2h-blind model had no training events
nearby). Grey: data statistics and surrogate sampling in quadrature; full model: between-member band; holdout: line.
Usage: scripts/paper_fig_holdout_isolation.py [--cache reports/paper_figs/twop_cache.pkl] [--iso reports/holdout_isolation/events.npz] [--out reports/paper_figs/holdout_isolation.png]"""
import argparse, pathlib, pickle, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="reports/paper_figs/twop_cache.pkl"); ap.add_argument("--iso", default="reports/holdout_isolation/events.npz"); ap.add_argument("--out", default="reports/paper_figs/holdout_isolation.png"); a = ap.parse_args()
C = pickle.load(open(a.cache, "rb")); S = C["S"]; D = S["open dataset"]; A = S["surrogate (full model)"]; B = S["surrogate (2p2h holdout)"]; r = np.load(a.iso)["r"]
RNG = np.random.default_rng(0); EDGES = np.array([1e-6, 1e-3, 1e-2, 10 ** -1.5, 1e-1, 10 ** -0.5, 1, 10])
def val(sel, v, kind):
    if kind == "frac_reco": return v[sel].mean()
    if kind == "frac": return np.isfinite(v[sel]).mean()
    q = v[sel]; q = q[np.isfinite(q)]; return np.median(q) if kind == "median" else q.mean()
def err(sel, v, kind):
    if kind in ("frac_reco", "frac"): p = val(sel, v, kind); return np.sqrt(p * (1 - p) / max(sel.sum(), 1))
    q = v[sel]; q = q[np.isfinite(q)]; f = np.median if kind == "median" else np.mean; return np.std([f(RNG.choice(q, len(q))) for _ in range(200)])
def series(key, kind):
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (r >= lo) & (r < hi)
        if key == "reco": sd = sel; vd = D["reco"].astype(float); sa = [sel] * len(A); va = [d["reco"].astype(float) for d in A]; sb = [sel] * len(B); vb = [d["reco"].astype(float) for d in B]
        else: sd = sel[D["reco"]]; vd = D[key]; sa = [sel[d["reco"]] for d in A]; va = [d[key] for d in A]; sb = [sel[d["reco"]] for d in B]; vb = [d[key] for d in B]
        if sd.sum() < 50: continue
        av = np.array([val(s, v, kind) for s, v in zip(sa, va)]); bv = np.array([val(s, v, kind) for s, v in zip(sb, vb)])
        rows.append((np.sqrt(lo * hi), val(sd, vd, kind), err(sd, vd, kind), av.mean(), av.std(ddof=1), bv.mean(), bv.std(ddof=1), sd.sum()))
    return np.array(rows)
PANELS = [("reco", "frac_reco", "reconstruction probability"), ("ratio", "median", r"median $P_\mathrm{reco}/P_\mathrm{true}$"), ("recoil", "median", "median recoil energy"), ("pP", "frac", "fraction with a proton fit"), ("n", "mean", "mean prong count")]
fig, axs = plots.plt.subplots(1, len(PANELS), figsize=(3.0 * len(PANELS), 3.1))
for ax, (key, kind, yl) in zip(axs, PANELS):
    xs, dv, de, am, asd, bm, bsd, n = series(key, kind).T; ex = np.sqrt(de ** 2 + bsd ** 2)
    pa = (am - dv) / np.sqrt(ex ** 2 + asd ** 2); pb = (bm - dv) / ex; print(f"{key:7s} pull RMS full {np.sqrt(np.mean(pa ** 2)):.2f} (max {np.abs(pa).max():.1f}), holdout {np.sqrt(np.mean(pb ** 2)):.2f} (max {np.abs(pb).max():.1f})")
    ax.fill_between(xs, 1 - ex / dv, 1 + ex / dv, color="0.85", lw=0, label="data statistics $\\oplus$ surrogate sampling")
    ax.fill_between(xs, (am - asd) / dv, (am + asd) / dv, color=plots.PALETTE["model"], alpha=0.4, lw=0); ax.plot(xs, am / dv, "--", color=plots.PALETTE["model"], lw=1.6, label="full model (band: members)")
    ax.plot(xs, bm / dv, ":", color=plots.PALETTE["third"], lw=1.8, label="2p2h holdout")
    ax.axhline(1, color="k", lw=0.6); ax.set_xscale("log"); ax.set_xlabel("density of other modes relative to 2p2h", fontsize=8); ax.set_title(yl, fontsize=8.5, loc="left"); ax.set_ylabel("ratio to the open dataset", fontsize=8)
    lim = max(0.03, 1.5 * np.nanmax(np.abs(np.r_[am / dv - 1, bm / dv - 1, ex / dv]))); ax.set_ylim(1 - lim, 1 + lim)
    ax.annotate("isolated", (0.03, 0.03), xycoords="axes fraction", fontsize=7, color="0.4"); ax.annotate("covered by other modes", (0.97, 0.03), xycoords="axes fraction", fontsize=7, color="0.4", ha="right")
axs[0].legend(frameon=False, fontsize=6.5, loc="upper left"); fig.suptitle("held-out 2p2h test events, by isolation in the surrogate's input space", fontsize=9, x=0.08, ha="left")
plots.save(fig, pathlib.Path(a.out)); print("written", a.out)
