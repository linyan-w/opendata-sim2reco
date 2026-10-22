#!/usr/bin/env python
"""Holdout figure as ratios: on the held-out 2p2h test events, the full model and the 2p2h-blind model relative to the open
dataset, per bin of the true quantity, for the reconstruction probability, the muon momentum response, the recoil energy,
the proton-fit fraction and the prong count. Grey band: the expected spread of the comparison, the open dataset's
statistical error (bootstrap) and the surrogate's sampling error (one draw per event on a sample of the same size, from the
regenerations with different noise) in quadrature; full model: between-member spread (band); 2p2h holdout: line only (no
ensemble). From the cache of scripts/paper_fig_response.py (twop_cache.pkl).
Usage: scripts/paper_fig_holdout_ratio.py [--cache reports/paper_figs/twop_cache.pkl] [--out reports/paper_figs/holdout_ratio.png]"""
import argparse, pathlib, pickle, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="reports/paper_figs/twop_cache.pkl"); ap.add_argument("--out", default="reports/paper_figs/holdout_ratio.png"); a = ap.parse_args()
C = pickle.load(open(a.cache, "rb")); S = C["S"]; Pt, keh, kep, nch = C["Pt"], C["keh"], C["kep"], C["nch"]
D = S["open dataset"]; A = S["surrogate (full model)"]; B = S["surrogate (2p2h holdout)"]
def stat(x, v, lo, hi, kind):
    s = (x >= lo) & (x < hi)
    if kind == "frac_reco": return v[s].mean() if s.sum() >= 30 else np.nan
    q = v[s]; q = q[np.isfinite(q)]
    if len(q) < 30: return np.nan
    if kind == "frac": return np.isfinite(v[s]).mean()
    if kind == "mean": return q.mean()
    return np.median(q)
RNG = np.random.default_rng(0)
def err(x, v, lo, hi, kind):
    """statistical error of the data bin: binomial for fractions, bootstrap (200 resamples) for the mean and the median"""
    s = (x >= lo) & (x < hi); n = max(s.sum(), 1)
    if kind in ("frac_reco", "frac"): p = (v[s].mean() if kind == "frac_reco" else np.isfinite(v[s]).mean()); return np.sqrt(p * (1 - p) / n)
    q = v[s]; q = q[np.isfinite(q)]; f = np.median if kind == "median" else np.mean
    return np.std([f(RNG.choice(q, len(q))) for _ in range(200)])
def series(key, xfull, kind, edges):
    """per bin: data value and error, A mean and spread over members, B mean and spread over regenerations (the surrogate's sampling error on a sample of this size); x is the true quantity of the reconstructed events"""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if key == "reco": xd, xa, xb = [xfull] * 3; vd = D["reco"].astype(float); va = [d["reco"].astype(float) for d in A]; vb = [d["reco"].astype(float) for d in B]
        else: xd = xfull[D["reco"]]; xa = [xfull[d["reco"]] for d in A]; xb = [xfull[d["reco"]] for d in B]; vd = D[key]; va = [d[key] for d in A]; vb = [d[key] for d in B]
        dv = stat(xd, vd, lo, hi, kind); de = err(xd, vd, lo, hi, kind)
        av = np.array([stat(x_, v_, lo, hi, kind) for x_, v_ in (zip(xa, va) if key != "reco" else zip([xfull] * len(va), va))]); bv = np.array([stat(x_, v_, lo, hi, kind) for x_, v_ in (zip(xb, vb) if key != "reco" else zip([xfull] * len(vb), vb))])
        if not np.isfinite(dv) or dv == 0: continue
        rows.append((0.5 * (lo + hi), dv, de, np.nanmean(av), np.nanstd(av, ddof=1), np.nanmean(bv), np.nanstd(bv, ddof=1)))
    return np.array(rows)
PANELS = [("reco", Pt, "frac_reco", np.logspace(np.log10(1.5), np.log10(40), 10), "true muon momentum [GeV]", "reconstruction probability", True),
          ("ratio", Pt, "median", np.logspace(np.log10(1.5), np.log10(40), 10), "true muon momentum [GeV]", r"median $P_\mathrm{reco}/P_\mathrm{true}$", True),
          ("recoil", keh, "median", np.logspace(np.log10(30), np.log10(2000), 9), "true hadronic kinetic energy [MeV]", "median recoil energy", True),
          ("pP", kep, "frac", np.array([50, 80, 120, 180, 270, 400, 600, 900, 1400]), "true leading proton kinetic energy [MeV]", "fraction with a proton fit", True),
          ("n", Pt, "mean", np.logspace(np.log10(1.5), np.log10(40), 8), "true muon momentum [GeV]", "mean prong count", True)]
fig, axs = plots.plt.subplots(1, len(PANELS), figsize=(3.0 * len(PANELS), 3.1))
for ax, (key, x, kind, edges, xl, yl, logx) in zip(axs, PANELS):
    r = series(key, x, kind, edges); xs, dv, de, am, asd, bm, bsd = r.T; ex = np.sqrt(de ** 2 + bsd ** 2)
    pull = (am - dv) / np.sqrt(ex ** 2 + asd ** 2); print(f"{key:7s} pulls of the full model: RMS {np.sqrt(np.mean(pull ** 2)):.2f}, max |{np.abs(pull).max():.2f}|, {np.sum(np.abs(pull) > 2)} of {len(pull)} beyond 2")
    ax.fill_between(xs, 1 - ex / dv, 1 + ex / dv, color="0.85", lw=0, label="data statistics $\\oplus$ surrogate sampling")
    ax.fill_between(xs, (am - asd) / dv, (am + asd) / dv, color=plots.PALETTE["model"], alpha=0.4, lw=0); ax.plot(xs, am / dv, "--", color=plots.PALETTE["model"], lw=1.6, label="full model (band: members)")
    ax.plot(xs, bm / dv, ":", color=plots.PALETTE["third"], lw=1.8, label="2p2h holdout")
    ax.axhline(1, color="k", lw=0.6); ax.set_xscale("log" if logx else "linear"); ax.set_xlabel(xl, fontsize=8); ax.set_title(yl, fontsize=8.5, loc="left"); ax.set_ylabel("ratio to the open dataset", fontsize=8)
    lim = max(0.03, 1.5 * np.nanmax(np.abs(np.r_[am / dv - 1, bm / dv - 1, ex / dv]))); ax.set_ylim(1 - lim, 1 + lim)
axs[0].legend(frameon=False, fontsize=6.5, loc="lower left"); fig.suptitle("held-out 2p2h test events", fontsize=9, x=0.08, ha="left")
plots.save(fig, pathlib.Path(a.out)); print("written", a.out)
