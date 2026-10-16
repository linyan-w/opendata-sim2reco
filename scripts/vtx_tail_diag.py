#!/usr/bin/env python
"""Why are the surrogate's vertex-z tails thin?  On the inclusive GENIE sample (truth with the open dataset's own
reconstruction attached, and the surrogate on the same events), the reconstructed-minus-true vertex z residual
is compared between the open dataset and the surrogate: its tail fractions overall and by true region, nucleus,
flavour/current, muon angle and the surrogate's vertex class; the shape of the far tail (sign and magnitude);
and the contribution of the tail events to the reco-fiducial sample from outside the fiducial volume.
Usage: scripts/vtx_tail_diag.py [--truth data/inclusive/genie_inclusive.truth.parquet --sur data/inclusive/surrogate_Aall_genie_inclusive.root --out reports/vtx_tail]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np, uproot
from sim2reco.prep.volumes import in_volume
from sim2reco.prep.frames import theta_phi_beam
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--truth", default="data/inclusive/genie_inclusive.truth.parquet"); ap.add_argument("--sur", default="data/inclusive/surrogate_Aall_genie_inclusive.root"); ap.add_argument("--out", default="reports/vtx_tail"); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
Z_FID = (5990.0, 8340.0); PLANE = 22.0  # mm; nominal plane pitch of the plane cycle used in the technote
t = ak.from_parquet(a.truth, columns=["mc_run", "mc_subrun", "mc_nthEvtInFile", "mc_vtx", "mc_targetZ", "mc_incoming", "mc_current", "mc_primFSLepton", "truth_vtx_module", "ref_reco_exists", "ref_vtx", "ref_muon_theta"])
key = lambda r, s, n: np.asarray(r, np.int64) * 10 ** 12 + np.asarray(s, np.int64) * 10 ** 7 + np.asarray(n, np.int64)
tk = key(t["mc_run"], t["mc_subrun"], t["mc_nthEvtInFile"]); order = np.argsort(tk)
v = ak.to_numpy(t["mc_vtx"])[:, :3]; Z = ak.to_numpy(t["mc_targetZ"]); inc = ak.to_numpy(t["mc_incoming"]); cur = ak.to_numpy(t["mc_current"]); mu = ak.to_numpy(t["mc_primFSLepton"]); th = np.degrees(theta_phi_beam(mu[:, 0], mu[:, 1], mu[:, 2])[0])
reg = np.full(len(v), 3); reg[v[:, 2] <= Z_FID[0]] = 1; reg[v[:, 2] >= Z_FID[1]] = 2; reg[in_volume(v, "fiducial")] = 0
s = uproot.open(a.sur)["MasterAnaDev"].arrays(["mc_run", "mc_subrun", "mc_nthEvtInFile", "MasterAnaDev_vtx", "surrogate_epi_vtx_z"], library="np")
pos = np.searchsorted(tk[order], key(s["mc_run"], s["mc_subrun"], s["mc_nthEvtInFile"])); srow = order[pos]; svtx = np.stack(s["MasterAnaDev_vtx"])[:, :3]
ex = ak.to_numpy(t["ref_reco_exists"]); rrow = np.where(ex)[0]; rvtx = ak.to_numpy(t["ref_vtx"])[rrow][:, :3]
def dz(row, rv): return rv[:, 2] - v[row, 2]
S = {"ref": {"row": rrow, "dz": dz(rrow, rvtx), "fid": in_volume(rvtx, "fiducial")}, "sur": {"row": srow, "dz": dz(srow, svtx), "fid": in_volume(svtx, "fiducial")}}
R = {}
def tails(m_ref, m_sur, name):
    r = {}
    for k, X in S.items():
        m = (m_ref if k == "ref" else m_sur); d = X["dz"][m]; n = max(len(d), 1)
        r[k] = {"n": int(len(d)), "gt100": float((np.abs(d) > 100).mean()), "gt300": float((np.abs(d) > 300).mean()), "gt1000": float((np.abs(d) > 1000).mean()), "pos100": float((d > 100).mean()), "neg100": float((d < -100).mean()), "p1": float(np.percentile(d, 1)) if len(d) else np.nan, "p99": float(np.percentile(d, 99)) if len(d) else np.nan}
    R[name] = r; print(f"{name:34s} n {r['ref']['n']:>8,} / {r['sur']['n']:>8,}  |dz|>100: {100*r['ref']['gt100']:5.1f}% / {100*r['sur']['gt100']:5.1f}%   >300: {100*r['ref']['gt300']:5.2f}% / {100*r['sur']['gt300']:5.2f}%   >1000: {100*r['ref']['gt1000']:5.2f}% / {100*r['sur']['gt1000']:5.2f}%   dz>100: {100*r['ref']['pos100']:5.1f}% / {100*r['sur']['pos100']:5.1f}%   dz<-100: {100*r['ref']['neg100']:5.1f}% / {100*r['sur']['neg100']:5.1f}%", flush=True)
allr, alls = np.ones(len(rrow), bool), np.ones(len(srow), bool)
print("open dataset / surrogate"); tails(allr, alls, "all reconstructed")
tails(S["ref"]["fid"], S["sur"]["fid"], "reco-fiducial")
for i, nm in enumerate(["true fiducial", "true upstream", "true downstream", "true transverse"]): tails(S["ref"]["fid"] & (reg[rrow] == i), S["sur"]["fid"] & (reg[srow] == i), f"  reco-fid, {nm}")
for z, nm in ((6, "C"), (82, "Pb"), (26, "Fe"), (1, "H"), (8, "O")): tails(S["ref"]["fid"] & (Z[rrow] == z), S["sur"]["fid"] & (Z[srow] == z), f"  reco-fid, {nm}")
for c, nm in ((1, "CC"), (2, "NC")): tails(S["ref"]["fid"] & (cur[rrow] == c), S["sur"]["fid"] & (cur[srow] == c), f"  reco-fid, {nm}")
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 90)): tails(S["ref"]["fid"] & (th[rrow] >= lo) & (th[rrow] < hi), S["sur"]["fid"] & (th[srow] >= lo) & (th[srow] < hi), f"  reco-fid, theta {lo}-{hi} deg")
# true-z bands: where along the detector do the tails live (true fiducial events only, by true z)
ze = [5990, 6300, 6700, 7100, 7500, 7900, 8340]
for lo, hi in zip(ze[:-1], ze[1:]): tails(S["ref"]["fid"] & (v[rrow, 2] >= lo) & (v[rrow, 2] < hi), S["sur"]["fid"] & (v[srow, 2] >= lo) & (v[srow, 2] < hi), f"  reco-fid, true z {lo}-{hi}")
# shape of the far tail: magnitude histogram and the plane-cycle phase of the reconstructed z in the tail
b = np.r_[-np.logspace(np.log10(10), np.log10(2000), 40)[::-1], 0, np.logspace(np.log10(10), np.log10(2000), 40)]
fig, axs = plots.plt.subplots(1, 3, figsize=(14, 3.8))
for k, c, lab in (("ref", plots.PALETTE["real"], "open dataset"), ("sur", plots.PALETTE["model"], "surrogate")):
    d = S[k]["dz"][S[k]["fid"]]; w = np.ones(len(d)) / len(S["ref"]["dz"][S["ref"]["fid"]])
    axs[0].hist(np.clip(d, -1999, 1999), b, weights=w, histtype="step", color=c, label=lab); axs[1].hist(np.clip(d, -1999, 1999), np.linspace(-800, 800, 161), weights=w, histtype="step", color=c, label=lab)
    m = np.abs(d) > 100; ph = np.mod(S[k]["dz"][S[k]["fid"]][m] + PLANE / 2, PLANE) - PLANE / 2
    axs[2].hist(ph, np.linspace(-11, 11, 45), histtype="step", color=c, label=f"{lab}, |dz|>100", density=True)
axs[0].set_xscale("symlog", linthresh=10); axs[0].set_yscale("log"); axs[0].set_xlabel("reco - true vertex z [mm]"); axs[0].set_ylabel("fraction of open-dataset reco-fid events"); axs[0].legend(frameon=False, fontsize=8)
axs[1].set_yscale("log"); axs[1].set_xlabel("reco - true vertex z [mm]"); axs[1].set_ylabel("fraction"); axs[2].set_xlabel("dz modulo the 22 mm plane cycle, tail events"); axs[2].set_ylabel("density"); axs[2].legend(frameon=False, fontsize=8)
fig.suptitle("events with the reconstructed vertex in the fiducial volume", fontsize=10); plots.save(fig, out / "figures" / "dz_tails.png")
# how much of the out-of-FV contribution to the reco-fiducial sample comes from the tails
for k in ("ref", "sur"):
    X = S[k]; m = X["fid"]; outside = reg[X["row"]] != 0; d = np.abs(X["dz"])
    R[f"{k}_outside_by_tail"] = {"outside_frac": float(outside[m].mean()), "outside_with_dz_gt100": float((outside & (d > 100))[m].mean()), "outside_with_dz_gt300": float((outside & (d > 300))[m].mean())}
    print(f"{k}: reco-fid from outside {100*R[f'{k}_outside_by_tail']['outside_frac']:.2f}%, of which with |dz|>100: {100*R[f'{k}_outside_by_tail']['outside_with_dz_gt100']:.2f}%, >300: {100*R[f'{k}_outside_by_tail']['outside_with_dz_gt300']:.2f}%")
json.dump(R, open(out / "summary.json", "w"), indent=1); print("written", out)
