#!/usr/bin/env python
"""Paper figures in one style for the full test sample and the held-out 2p2h events:
 (muon) median P_reco/P_true vs true muon P; (recoil) median reconstructed recoil vs true hadronic KE;
 (hadron) per species (proton, charged pion): fraction of events with a fit under that hypothesis (left axis) and the
 median leading fitted momentum (right axis) vs the true leading-hadron KE of that species.
Open dataset and one or more surrogates. Usage:
 scripts/paper_fig_response.py OUT_PREFIX --models A=reports/m3_1A_z [B=reports/m3_1A_z_no2p2h] [--only-inttype 8] [--n 300000] [--replot]"""
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
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--only-inttype", type=int, nargs="*", default=None); ap.add_argument("--n", type=int, default=300000); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--replot", action="store_true", help="re-draw from OUT_PREFIX_cache.pkl"); a = ap.parse_args()
from sim2reco.models.bayes_last import GaussianLastLayer, heads_for, head_layer
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet"))); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]
if a.only_inttype: te = te[np.isin(d["intType"][te], a.only_inttype)]
idx = np.random.default_rng(0).permutation(te)[:a.n]
cls, mom, mask = _pad(d, idx); X, names = event_features(cls, mom, mask, d["ctx"][idx])
keh, kep = X[:, names.index("sumKE_had")], X[:, names.index("maxKE_p")]; kpi = np.maximum(X[:, names.index("maxKE_pip")], X[:, names.index("maxKE_pim")]); nch = d["n_charged_true"][idx].astype(int); Pt = np.linalg.norm(d["mu_true"][idx], axis=1) / 1e3
def summarise(reco, y, prongs, offs):
    """per-event quantities for reconstructed events: muon ratio, recoil [GeV], leading proton-fit and pion-hypothesis P [MeV] (nan if none), n prongs, n with kinematics"""
    n = np.diff(offs); seg = np.repeat(np.arange(len(n)), n); lead = np.full(len(n), -1.0); leadpi = np.full(len(n), -1.0)
    np.maximum.at(lead, seg, np.where(prongs[:, 4] > 0.5, prongs[:, 5], -1.0)); np.maximum.at(leadpi, seg, np.where(prongs[:, 2] > 0.5, prongs[:, 3], -1.0))
    nkin = np.bincount(seg, weights=(prongs[:, 2] > 0.5).astype(float), minlength=len(n))
    return {"reco": reco, "ratio": np.linalg.norm(y[:, :3], axis=1) / 1e3 / Pt[reco], "recoil": y[:, 6] / 1e3, "pP": np.where(lead > 0, lead, np.nan), "piP": np.where(leadpi > 0, leadpi, np.nan), "n": n.astype(float), "nkin": nkin}
import pickle
if not a.replot:
    first = None; S = {}
    for spec in a.models:
        lab, D = spec.split("="); model, tf, ptf = load_model(f"{D}/model.pt"); ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf)
        if first is None:  # open dataset, decoded with the first model's transforms (identical raw values)
            r = d["reco_exists"][idx] & ds.valid; ir = idx[r]
            offs = np.concatenate([[0], np.cumsum(np.diff(d["p_offsets"])[ir])]); pr = np.concatenate([d["prongs"][d["p_offsets"][e]:d["p_offsets"][e + 1]] for e in ir])
            S["open dataset"] = summarise(r, d["tier1"][ir], pr, offs); first = lab
        post = {k: GaussianLastLayer.from_state(v, "cuda") for k, v in torch.load(f"{D}/bayes_last.pt", map_location="cpu", weights_only=False).items()}
        layers = {k: head_layer(model, p) for k, (p, _) in heads_for(model).items()}; W0 = {k: (l.weight.data.clone(), l.bias.data.clone()) for k, l in layers.items()}
        gen = torch.Generator(device="cuda"); gen.manual_seed(123); draws = []
        for kd in range(a.draws):  # posterior weight draw + fresh generation: the spread over draws is epistemic + the surrogate's own sampling uncertainty
            for k in layers:
                if k in post: dW, db = post[k].sample_delta(gen); layers[k].weight.data = W0[k][0] + dW; layers[k].bias.data = W0[k][1] + db
            torch.manual_seed(kd); X1, EX, PR, PM = [], [], [], []
            with torch.no_grad():
                for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
                    s = model.sample(to_dev(b, "cuda"), 64); X1.append(s["x1"].cpu()); EX.append(s["exist"].cpu()); PR.append(s["prongs"].cpu()); PM.append(s["pmask"].cpu())
            Np = max(p.shape[1] for p in PR); PR = torch.cat([torch.nn.functional.pad(p, (0, 0, 0, Np - p.shape[1])) for p in PR]).numpy(); PM = torch.cat([torch.nn.functional.pad(p, (0, Np - p.shape[1])) for p in PM]).numpy()
            ex = torch.cat(EX).numpy(); X1 = torch.cat(X1).numpy(); y = tf.inverse(X1[ex], d["mu_true"][idx][ex], d["ctx"][idx][ex])
            pm = PM[ex]; pr = ptf.inverse(PR[ex][pm]); offs = np.concatenate([[0], np.cumsum(pm.sum(1))]); draws.append(summarise(ex, y, pr, offs))
        for k in layers: layers[k].weight.data, layers[k].bias.data = W0[k]
        S[f"surrogate {lab}" if len(a.models) > 1 else "surrogate"] = draws; print(lab, "sampled", len(draws), "draws", flush=True)
if not a.replot: pickle.dump({"S": S, "Pt": Pt, "keh": keh, "kep": kep, "kpi": kpi, "nch": nch}, open(a.out + "_cache.pkl", "wb"))
else: C_ = pickle.load(open(a.out + "_cache.pkl", "rb")); S, Pt, keh, kep, kpi, nch = (C_[k] for k in ("S", "Pt", "keh", "kep", "kpi", "nch"))
STY = {"open dataset": (plots.PALETTE["real"], "o", "-")}; cyc = [(plots.PALETTE["model"], "s", "--"), (plots.PALETTE["third"], "^", ":")]
for i, k in enumerate([k for k in S if k != "open dataset"]): STY[k] = cyc[i]
def stat(x, yv, a_, b_, kind):
    s = (x >= a_) & (x < b_)
    if kind == "frac": return (np.isfinite(yv[s]).mean(), np.sqrt(np.isfinite(yv[s]).mean() * (1 - np.isfinite(yv[s]).mean()) / max(s.sum(), 1))) if s.sum() >= 30 else (np.nan, np.nan)
    q = yv[s][np.isfinite(yv[s])]
    if len(q) < 30: return np.nan, np.nan
    if kind == "mean": return q.mean(), q.std() / np.sqrt(len(q))
    return np.median(q), 1.2533 * q.std() / np.sqrt(len(q)) if False else (np.median(q), 1.4826 * np.median(np.abs(q - np.median(q))) * 1.2533 / np.sqrt(len(q)))[1]
def prof(ax, xfull, key, edges, kind="median", logx=True, xs=None, q=None, legend=True):
    """q: per-quantity style override {"marker", "ls", "mfc"} for two quantities on one panel (twin axes)"""
    xs = np.sqrt(edges[:-1] * edges[1:]) if xs is None else xs
    ok = np.isfinite(np.array([stat(xfull[S["open dataset"]["reco"]], S["open dataset"][key], a_, b_, kind)[0] for a_, b_ in zip(edges[:-1], edges[1:])]))
    xs = np.asarray(xs, float)[ok]; edges_ok = [(a_, b_) for (a_, b_), o in zip(zip(edges[:-1], edges[1:]), ok) if o]
    for k, v in S.items():
        c, mk, ls = STY[k]
        if q: mk, ls = q["marker"], q["ls"]
        if k == "open dataset":
            m, e = np.array([stat(xfull[v["reco"]], v[key], a_, b_, kind) for a_, b_ in edges_ok]).T
            ax.errorbar(xs, m, e, fmt=mk, ms=3.5, color=c, mfc=(q or {}).get("mfc", c), capsize=0, label=k, zorder=3)
        else:
            M = np.array([[stat(xfull[dv["reco"]], dv[key], a_, b_, kind)[0] for a_, b_ in edges_ok] for dv in v])
            mu, sd = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1)
            ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.35, lw=0); ax.plot(xs, mu, ls=ls, color=c, lw=1.4, label=k)
    if logx: ax.set_xscale("log")
    if legend: ax.legend(frameon=False, fontsize=7)
from matplotlib.lines import Line2D
fig, ax = plots.plt.subplots(figsize=(4.2, 3.3))
prof(ax, Pt, "ratio", np.logspace(np.log10(0.5), np.log10(40), 13)); ax.set_xlabel("true muon momentum [GeV]"); ax.set_ylabel(r"median $P_\mathrm{reco}/P_\mathrm{true}$ of the muon")
plots.save(fig, pathlib.Path(a.out + "_muon.png"))
fig, ax = plots.plt.subplots(figsize=(4.2, 3.3))
prof(ax, keh, "recoil", np.logspace(np.log10(30), np.log10(20000), 12)); ax.set_yscale("log"); ax.set_xlabel("true hadronic kinetic energy [MeV]"); ax.set_ylabel("median recoil energy [GeV]")
plots.save(fig, pathlib.Path(a.out + "_recoil.png"))
QF, QP = {"marker": "o", "ls": "-", "mfc": None}, {"marker": "s", "ls": "--", "mfc": "white"}
e = np.array([50, 80, 120, 180, 270, 400, 600, 900, 1400, 2500])
SPEC = [(x, key, hyp, sp) for x, key, hyp, sp in ((kep, "pP", "proton", "proton"), (kpi, "piP", "pion", "charged-pion")) if (x[S["open dataset"]["reco"]] >= e[0]).sum() >= 300]  # species absent from the sample (no pions in 2p2h) get no panel
fig, axs = plots.plt.subplots(1, len(SPEC), figsize=(4.8 * len(SPEC), 3.4), squeeze=False); axs = axs[0]
for ax, (x, key, hyp, sp) in zip(axs, SPEC):
    QF["mfc"] = None; prof(ax, x, key, e, kind="frac", q=QF, legend=False); ax.set_ylim(0, 1.05)
    ax2 = ax.twinx(); prof(ax2, x, key, e, q=QP, legend=False); ax2.grid(False); ax.set_xlim(e[0], e[-1])
    ax.set_xlabel(f"true leading {sp} kinetic energy [MeV]"); ax.set_ylabel(f"fraction of events with a {hyp} fit"); ax2.set_ylabel(f"median leading {hyp}-fit momentum [MeV]")
    ax2.spines["right"].set_visible(True)
hs = [Line2D([], [], color=STY[k][0], lw=6, alpha=0.6 if k != "open dataset" else 1, label=k) for k in S]
hq = [Line2D([], [], color="0.3", marker="o", ls="-", label="fraction with a fit (left)"), Line2D([], [], color="0.3", marker="s", mfc="white", ls="--", label="median fitted momentum (right)")]
fig.legend(handles=hs + hq, frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(hs) + len(hq) if len(axs) > 1 else 2)
fig.tight_layout(); fig.savefig(a.out + "_hadron.png", bbox_inches="tight"); plots.plt.close(fig); print("written", a.out)
