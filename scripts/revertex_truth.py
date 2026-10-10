#!/usr/bin/env python
"""Redraw the vertices of converted generator Truth files (e.g. NuWro, which carries no vertex) from the open
dataset's vertex pool of another volume, per target nucleus, and merge them; nuclei absent from the pool are dropped.
Adds `nuwro_wcomp`, the per-nucleus weight that brings the merged sample's nucleus shares to those of the open
dataset in that volume (generator campaigns are produced per nucleus, so their shares are arbitrary).
Usage: scripts/revertex_truth.py OUT.truth.parquet IN1.truth.parquet [IN2 ...] --volume full [--seed 1]"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np
from sim2reco.prep.volumes import vertex_pool, in_volume
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("inp", nargs="+"); ap.add_argument("--volume", default="full"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
pool, frac = vertex_pool(a.slim_dir, a.volume); rng = np.random.default_rng(a.seed)
t = ak.concatenate([ak.from_parquet(f) for f in a.inp]); Z = ak.to_numpy(t["mc_targetZ"])
key = ak.to_numpy(t["mc_run"]).astype(np.int64) * 10_000_000 + ak.to_numpy(t["mc_nthEvtInFile"]); assert len(np.unique(key)) == len(key), "event keys collide across inputs"
keep = np.isin(Z, list(pool)); vtx = np.empty((len(t), 4)); mod = np.empty(len(t), np.int32)
for z in np.unique(Z[keep]):
    m = Z == z; pick = pool[int(z)][rng.integers(0, len(pool[int(z)]), m.sum())]; vtx[m] = pick[:, :4]; mod[m] = pick[:, 4].astype(np.int32)
missing = sorted(set(pool) - set(np.unique(Z).tolist()))
print(f"pool nuclei absent from the input (share of the {a.volume} volume): " + ", ".join(f"Z={z}: {100*frac[z]:.2f}%" for z in sorted(missing, key=lambda z: -frac[z])))
t, Z, vtx, mod = t[keep], Z[keep], vtx[keep], mod[keep]
zs, cnt = np.unique(Z, return_counts=True); norm = sum(frac[int(z)] for z in zs)  # renormalise over the nuclei present
w = {int(z): frac[int(z)] / norm / (c / len(Z)) for z, c in zip(zs, cnt)}
print("Z: events, sample share -> target share, weight"); [print(f"  {z:3d}: {c:9,d}  {100*c/len(Z):6.2f}% -> {100*frac[z]/norm:6.2f}%  w={w[z]:.3f}") for z, c in zip(zs.tolist(), cnt)]
t = ak.with_field(t, vtx, "mc_vtx"); t = ak.with_field(t, mod, "truth_vtx_module"); t = ak.with_field(t, in_volume(vtx, "fiducial"), "truth_is_fiducial")
t = ak.with_field(t, np.array([w[int(z)] for z in Z]), "nuwro_wcomp")
ak.to_parquet(t, a.out); print(f"wrote {len(t):,} events with {a.volume}-volume vertices -> {a.out}")
