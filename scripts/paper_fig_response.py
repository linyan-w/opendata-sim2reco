#!/usr/bin/env python
"""Paper figures in one style for the full test sample and the held-out 2p2h events:
 (event) muon P_reco/P_true vs true muon P, reconstructed recoil vs true hadronic KE;
 (prong) proton-fit rate and leading proton-hypothesis momentum vs true leading-proton KE, mean prongs vs true charged hadrons.
Open dataset and one or more surrogates. Usage:
 scripts/paper_fig_response.py OUT_PREFIX --models A=reports/m3_1A_z [B=reports/m3_1A_z_no2p2h] [--only-inttype 8] [--n 300000]"""
import argparse, glob, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sim2reco.data.compact import load_compact, CompactDataset, collate
from sim2reco.data.dataset import split_by_subrun
from sim2reco.prep.features import event_features
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev, _pad
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--only-inttype", type=int, nargs="*", default=None); ap.add_argument("--n", type=int, default=300000); a = ap.parse_args()
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet"))); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]
if a.only_inttype: te = te[np.isin(d["intType"][te], a.only_inttype)]
idx = np.random.default_rng(0).permutation(te)[:a.n]
cls, mom, mask = _pad(d, idx); X, names = event_features(cls, mom, mask, d["ctx"][idx])
keh, kep = X[:, names.index("sumKE_had")], X[:, names.index("maxKE_p")]; nch = d["n_charged_true"][idx].astype(int); Pt = np.linalg.norm(d["mu_true"][idx], axis=1) / 1e3
def summarise(reco, y, prongs, offs):
    """per-event quantities for reconstructed events: muon ratio, recoil [GeV], leading proton-fit P [MeV] (nan if none), n prongs, n with kinematics"""
    n = np.diff(offs); seg = np.repeat(np.arange(len(n)), n); lead = np.full(len(n), -1.0)
    np.maximum.at(lead, seg, np.where(prongs[:, 4] > 0.5, prongs[:, 5], -1.0))
    nkin = np.bincount(seg, weights=(prongs[:, 2] > 0.5).astype(float), minlength=len(n))
    return {"reco": reco, "ratio": np.linalg.norm(y[:, :3], axis=1) / 1e3 / Pt[reco], "recoil": y[:, 6] / 1e3, "pP": np.where(lead > 0, lead, np.nan), "n": n.astype(float), "nkin": nkin}
first = None; S = {}
for spec in a.models:
    lab, D = spec.split("="); model, tf, ptf = load_model(f"{D}/model.pt"); ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf)
    if first is None:  # open dataset, decoded with the first model's transforms (identical raw values)
        r = d["reco_exists"][idx] & ds.valid; ir = idx[r]
        offs = np.concatenate([[0], np.cumsum(np.diff(d["p_offsets"])[ir])]); pr = np.concatenate([d["prongs"][d["p_offsets"][e]:d["p_offsets"][e + 1]] for e in ir])
        S["open dataset"] = summarise(r, d["tier1"][ir], pr, offs); first = lab
    torch.manual_seed(0); X1, EX, PR, PM = [], [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
            s = model.sample(to_dev(b, "cuda"), 64); X1.append(s["x1"].cpu()); EX.append(s["exist"].cpu()); PR.append(s["prongs"].cpu()); PM.append(s["pmask"].cpu())
    Np = max(p.shape[1] for p in PR); PR = torch.cat([torch.nn.functional.pad(p, (0, 0, 0, Np - p.shape[1])) for p in PR]).numpy(); PM = torch.cat([torch.nn.functional.pad(p, (0, Np - p.shape[1])) for p in PM]).numpy()
    ex = torch.cat(EX).numpy(); X1 = torch.cat(X1).numpy(); y = tf.inverse(X1[ex], d["mu_true"][idx][ex], d["ctx"][idx][ex])
    pm = PM[ex]; pr = ptf.inverse(PR[ex][pm]); offs = np.concatenate([[0], np.cumsum(pm.sum(1))])
    S[f"surrogate {lab}" if len(a.models) > 1 else "surrogate"] = summarise(ex, y, pr, offs); print(lab, "sampled", flush=True)
STY = {"open dataset": (plots.PALETTE["real"], "o", "-")}; cyc = [(plots.PALETTE["model"], "s", "--"), (plots.PALETTE["third"], "^", ":")]
for i, k in enumerate([k for k in S if k != "open dataset"]): STY[k] = cyc[i]
def prof(ax, xfull, key, edges, band=True, frac=False, logx=True):
    for k, v in S.items():
        x = xfull[v["reco"]]; yv = v[key]; c, mk, ls = STY[k]; xs, m, lo, hi = [], [], [], []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            s = (x >= a_) & (x < b_)
            if s.sum() < 30: continue
            if frac: xs.append(np.sqrt(a_ * b_)); m.append(np.isfinite(yv[s]).mean()); continue
            q = yv[s][np.isfinite(yv[s])]
            if len(q) < 20: continue
            xs.append(np.sqrt(a_ * b_) if logx else 0.5 * (a_ + b_)); p = np.percentile(q, [16, 50, 84]); m.append(p[1]); lo.append(p[0]); hi.append(p[2])
        if band and not frac: ax.fill_between(xs, lo, hi, color=c, alpha=0.13, lw=0)
        ax.plot(xs, m, marker=mk, ls=ls, ms=3.5, color=c, label=k)
    if logx: ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=7)
fig, axs = plots.plt.subplots(1, 2, figsize=(8, 3.3))
prof(axs[0], Pt, "ratio", np.logspace(np.log10(0.5), np.log10(40), 13)); axs[0].set_ylim(0, 2); axs[0].set_xlabel("true muon momentum [GeV]"); axs[0].set_ylabel(r"$P_\mathrm{reco}/P_\mathrm{true}$ of the muon")
prof(axs[1], keh, "recoil", np.logspace(np.log10(30), np.log10(20000), 12)); axs[1].set_yscale("log"); axs[1].set_xlabel("true hadronic kinetic energy [MeV]"); axs[1].set_ylabel("reconstructed recoil energy [GeV]")
plots.save(fig, pathlib.Path(a.out + "_event.png"))
fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.3)); e = np.array([50, 80, 120, 180, 270, 400, 600, 900, 1400, 2500])
prof(axs[0], kep, "pP", e, frac=True); axs[0].set_xlabel("true leading-proton kinetic energy [MeV]"); axs[0].set_ylabel("fraction of events with a proton fit")
prof(axs[1], kep, "pP", e); axs[1].set_yscale("log"); axs[1].set_xlabel("true leading-proton kinetic energy [MeV]"); axs[1].set_ylabel("leading proton-fit momentum [MeV]")
for k, v in S.items():
    c, mk, ls = STY[k]; x = nch[v["reco"]]; ks = [j for j in range(7) if (x == j).sum() >= 30]
    axs[2].plot(ks, [v["n"][x == j].mean() for j in ks], marker=mk, ls=ls, ms=3.5, color=c, label=k)
axs[2].set_xlabel("true charged hadrons after FSI"); axs[2].set_ylabel("mean reconstructed prongs"); axs[2].legend(frameon=False, fontsize=7)
plots.save(fig, pathlib.Path(a.out + "_prong.png")); print("written", a.out)
