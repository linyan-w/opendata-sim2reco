#!/usr/bin/env python
"""Paper figure: muon momentum response and recoil energy versus truth in physical units (test split, model A).
Usage: scripts/paper_fig_conditionals.py MODEL_DIR OUT.png [--n 300000]"""
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
ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=300000); a = ap.parse_args()
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet"))); split = split_by_subrun(d["subrun"], seed=0)
idx = np.random.default_rng(0).permutation(np.where(split == 2)[0])[:a.n]; model, tf, ptf = load_model(f"{a.model_dir}/model.pt")
ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); S = []
with torch.no_grad():
    for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
        s = model.sample(to_dev(b, "cuda"), 64, prongs=False); S.append(torch.cat([s["exist"][:, None].float(), s["x1"]], 1).cpu())
S = torch.cat(S).numpy(); ok_r = d["reco_exists"][idx] & ds.valid; ok_s = S[:, 0] > 0
cls, mom, mask = _pad(d, idx); X, names = event_features(cls, mom, mask, d["ctx"][idx]); ke = X[:, names.index("sumKE_had")]; Pt = np.linalg.norm(d["mu_true"][idx], axis=1) / 1e3
yr = tf.inverse(ds.x1[ok_r], d["mu_true"][idx][ok_r], d["ctx"][idx][ok_r]); ys = tf.inverse(S[ok_s, 1:], d["mu_true"][idx][ok_s], d["ctx"][idx][ok_s])
def band(x, y, e):
    q = []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y); q.append(np.percentile(y[m], [16, 50, 84]) if m.sum() >= 50 else [np.nan] * 3)
    return np.sqrt(e[:-1] * e[1:]), np.array(q)
fig, axs = plots.plt.subplots(1, 2, figsize=(8, 3.3))
ep = np.logspace(np.log10(0.5), np.log10(40), 13); ek = np.logspace(np.log10(30), np.log10(20000), 12)
for lab, y, ok, c, mk in (("open dataset", yr, ok_r, plots.PALETTE["real"], "o"), ("surrogate", ys, ok_s, plots.PALETTE["model"], "s")):
    x, q = band(Pt[ok], np.linalg.norm(y[:, :3], axis=1) / 1e3 / Pt[ok], ep); axs[0].fill_between(x, q[:, 0], q[:, 2], color=c, alpha=0.18, lw=0); axs[0].plot(x, q[:, 1], marker=mk, ms=3.5, color=c, label=lab)
    x, q = band(ke[ok], y[:, 6] / 1e3, ek); axs[1].fill_between(x, q[:, 0], q[:, 2], color=c, alpha=0.18, lw=0); axs[1].plot(x, q[:, 1], marker=mk, ms=3.5, color=c, label=lab)
axs[0].set_xscale("log"); axs[0].set_xlabel("true muon momentum [GeV]"); axs[0].set_ylabel(r"$P_\mathrm{reco}/P_\mathrm{true}$ of the muon"); axs[0].set_ylim(0, 2)
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("true hadronic kinetic energy [MeV]"); axs[1].set_ylabel("reconstructed recoil energy [GeV]")
for ax in axs: ax.legend(frameon=False, fontsize=8)
plots.save(fig, pathlib.Path(a.out)); print("written")
