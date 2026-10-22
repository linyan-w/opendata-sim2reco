#!/usr/bin/env python
"""The open dataset's own per-event resolution in the bins of the ensemble spread study: robust width (1.4826 MAD) of
P_reco/P_true of the muon and of z_reco - z_true, on the same CC nu_mu test events as scripts/ensemble_spread.py, by true
muon angle and pT. Compared with the surrogate's two-draw resolution this says whether the resolution growth at large
angle is the detector's. Writes reports/ensemble/data_resolution.json.
Usage: scripts/data_resolution.py [--model reports/m3_1A_all] [--n 300000]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sim2reco.data.compact import load_compact, selection_from_config, subset_mask
from sim2reco.data.dataset import split_by_subrun
from sim2reco.prep.frames import theta_phi_beam
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="reports/m3_1A_all"); ap.add_argument("--n", type=int, default=300000); ap.add_argument("--out", default="reports/ensemble/data_resolution.json"); a = ap.parse_args()
cfg = torch.load(f"{a.model}/model.pt", map_location="cpu", weights_only=False)["config"]
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet")), **selection_from_config(cfg)); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]; te = te[subset_mask(d, "ccnumu")[te]]; idx = np.random.default_rng(0).permutation(te)[:a.n]
t0 = d["tier0"][idx]; ex = t0[:, 0] > 0; yr = d["tier1"][idx]; mu = d["mu_true"][idx]; ctx = d["ctx"][idx]
th = np.degrees(theta_phi_beam(mu[:, 0], mu[:, 1], mu[:, 2])[0]); P = np.linalg.norm(mu, axis=1) / 1e3; pT = P * np.sin(np.radians(th))
rP = np.linalg.norm(yr[:, :3], axis=1) / 1e3 / P; dz = yr[:, 5] - ctx[:, 2]
mad = lambda q: 1.4826 * np.median(np.abs(q - np.median(q)))
PT = [0, .075, .15, .25, .325, .4, .475, .55, .7, .85, 1, 1.25, 1.5, 2.5, 4.5]; R = {}
for name, x, edges in (("true pT [GeV]", pT, PT), ("true angle [deg]", th, [0, 5, 10, 15, 20, 30, 45, 90])):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ex & (x >= lo) & (x < hi)
        if m.sum() < 200: continue
        rows.append({"bin": f"[{lo:g},{hi:g})", "n": int(m.sum()), "resP": float(mad(rP[m])), "resz": float(mad(dz[m])), "resP_core": float(mad(rP[m & (np.abs(dz) < 50)]))})
        print(f"  {name:16s} {rows[-1]['bin']:12s} n {m.sum():7d}  data P resolution {100*rows[-1]['resP']:.2f}%  z {rows[-1]['resz']:.1f} mm")
    R[name] = rows
json.dump(R, open(a.out, "w"), indent=1); print("written", a.out)
