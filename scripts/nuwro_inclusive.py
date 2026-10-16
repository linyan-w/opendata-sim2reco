#!/usr/bin/env python
"""One inclusive NuWro sample (nu_mu, nu_mu-bar, nu_e, nu_e-bar; CC and NC; tracker nuclei + Pb + Fe) over the full
training volume that mimics the open dataset's exposure and composition, to be run through the all-events surrogate
in one pass, and the matching open-dataset (GENIE) sample.

Strata are (flavour, current, nucleus). Fluxes are not used directly: for each flavour, the number of nucleons of each
nucleus seen by that flavour's flux is taken from the open dataset, n_fZ ~ N_open(f, CC, Z) / sigma_NuWro(f, CC, Z),
so the CC events per flavour and nucleus follow the open dataset (flavour rates and nucleus shares of the official
MC), and the NC events are n_fZ * sigma_NuWro(f, NC, Z), i.e. NuWro's own NC/CC ratio for that flavour and nucleus
(exact, since CC and NC of a flavour share its flux). The overall scale is the largest at which every stratum with at
least --min-events generated events can be filled without reuse; each stratum is a random subsample of its campaign
(weight 1), and smaller strata keep all their events with weight `nuwro_wcomp` = target / generated. Vertices are
redrawn from the open dataset's events of the same current and nucleus (all flavours) in the full volume.
The GENIE sample is every open-dataset event in the full volume (all flavours and currents), subsampled at the same
fraction of the open-dataset exposure, with its own reconstruction attached.

Usage: scripts/nuwro_inclusive.py OUT_DIR --truth TRUTH... --runs RUN_JSON... [--min-events 5000]; then
       scripts/nuwro_inclusive.py OUT_DIR --genie-only (separate step: the GENIE build reads the whole open dataset)"""
import argparse, glob, json, pathlib, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np
from sim2reco.prep.volumes import in_volume

ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--truth", nargs="+", default=[]); ap.add_argument("--runs", nargs="+", default=[])
ap.add_argument("--min-events", type=int, default=5000, help="strata with fewer generated events than this do not limit the scale and are weighted instead")
ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--genie-only", action="store_true", help="only (re)write the matching GENIE sample from an existing build")
ap.add_argument("--name", default="nuwro_inclusive", help="output stem (<name>.truth.parquet, <name>.json)"); ap.add_argument("--max-fraction", type=float, default=None, help="cap the exposure fraction (e.g. to reuse the GENIE sample of another build)"); a = ap.parse_args()
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
FLAV = {14: "numu", -14: "numubar", 12: "nue", -12: "nuebar"}


def write_genie(frac, n_open_all):
    n = int(round(frac * n_open_all))
    subprocess.run([sys.executable, "scripts/genie_tracker_truth.py", str(out / "genie_inclusive.truth.parquet"), "--volume", "full", "--population", "all", "--n", str(n)], check=True)


if a.genie_only:
    r = json.load(open(out / f"{a.name}.json")); write_genie(r["exposure_fraction"], r["open_events_all"]); sys.exit()

# flux-averaged cross section per nucleon, per (flavour, current, Z), from the campaign records
ZOF = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Si": 14, "Cl": 17, "Ti": 22, "Fe": 26, "Pb": 82}
sigma = {}
for f in a.runs:
    r = json.load(open(f)); cur = 2 if (r.get("current") or "cc").lower() == "nc" else 1; beam = int(r.get("beam_particle", r.get("pdg", 14)))  # NuWro or GiBUU campaign record
    for lab, s in r["strata"].items(): sigma[(beam, cur, ZOF["".join(c for c in lab if c.isalpha())])] = s.get("sigma_mean_cm2_per_nucleon", s.get("sigma_1e-38cm2_per_nucleon"))  # only ratios within a flavour are used

# open dataset in the full volume: event counts per (flavour, current, Z), vertex pools per (current, Z)
n_open, pools, n_open_all = {}, {}, 0
for f in sorted(glob.glob(f"{a.slim_dir}/*.truth.parquet")):
    t = ak.from_parquet(f, columns=["mc_vtx", "mc_targetZ", "truth_vtx_module", "mc_current", "mc_incoming"])
    v = ak.to_numpy(t["mc_vtx"]); Z = ak.to_numpy(t["mc_targetZ"]); mod = ak.to_numpy(t["truth_vtx_module"]); cur = ak.to_numpy(t["mc_current"]); inc = ak.to_numpy(t["mc_incoming"])
    full = in_volume(v, "full"); n_open_all += int(full.sum())
    for key, cnt in zip(*np.unique(np.stack([inc[full], cur[full], Z[full]], 1), axis=0, return_counts=True)): n_open[tuple(int(x) for x in key)] = n_open.get(tuple(int(x) for x in key), 0) + int(cnt)
    for c in (1, 2):
        for z in np.unique(Z[full & (cur == c)]): m = full & (cur == c) & (Z == z); pools.setdefault((c, int(z)), []).append(np.column_stack([v[m], mod[m]]))
pools = {k: np.concatenate(p) for k, p in pools.items()}

# generated NuWro events per stratum (key columns only; the full files are read one at a time below, to bound memory)
KEYS = [np.stack([ak.to_numpy(t[c]) for c in ("mc_incoming", "mc_current", "mc_targetZ")], 1) for t in (ak.from_parquet(f, columns=["mc_incoming", "mc_current", "mc_targetZ"]) for f in a.truth)]
allkey = np.concatenate(KEYS); fid = np.concatenate([np.full(len(k), i) for i, k in enumerate(KEYS)]); frow = np.concatenate([np.arange(len(k)) for k in KEYS])
avail = {tuple(int(x) for x in k): int(n) for k, n in zip(*np.unique(allkey, axis=0, return_counts=True))}

# target counts per unit scale (scale = total nu_mu CC events)
unit = {}
for (fl, c, z) in avail:
    if (fl, 1, z) not in n_open or (fl, 1, z) not in sigma or (fl, c, z) not in sigma: print(f"  skipped stratum {FLAV[fl]} {'CC' if c == 1 else 'NC'} Z={z}: no open-dataset CC events or cross section"); continue
    unit[(fl, c, z)] = n_open[(fl, 1, z)] * sigma[(fl, c, z)] / sigma[(fl, 1, z)]
norm = sum(n_open[(14, 1, z)] for (fl, c, z) in unit if fl == 14 and c == 1); unit = {k: u / norm for k, u in unit.items()}
main = [k for k in unit if avail[k] >= a.min_events]
scale = min(avail[k] / unit[k] for k in main); limiting = min(main, key=lambda k: avail[k] / unit[k])
if a.max_fraction is not None and scale / norm > a.max_fraction: scale = a.max_fraction * norm; limiting = ("cap", 0, 0); print(f"scale capped at an exposure fraction of {a.max_fraction}")
keep = np.zeros(len(allkey), bool); wt = np.zeros(len(allkey)); rows = []
for k in sorted(unit):
    fl, c, z = k; tgt = unit[k] * scale; idx = np.where(np.all(allkey == np.array(k), axis=1))[0]; n = len(idx)
    if n > tgt: idx = rng.permutation(idx)[:int(round(tgt))]; w = 1.0
    else: w = tgt / n
    keep[idx] = True; wt[idx] = w
    rows.append(dict(flavour=FLAV[fl], current="CC" if c == 1 else "NC", Z=z, generated=n, kept=len(idx), weight=w, open_events=n_open.get(k, 0)))
parts = []
for i, f in enumerate(a.truth):
    m_ = keep & (fid == i)
    if not m_.any(): continue
    t = ak.from_parquet(f)[frow[m_]]; c_ = allkey[m_, 1]; z_ = allkey[m_, 2]; v = np.empty((len(t), 4)); mod = np.empty(len(t), np.int32)
    for c, z in set(zip(c_.tolist(), z_.tolist())):
        mm = (c_ == c) & (z_ == z); p = pools[(c, z)]; pick = p[rng.integers(0, len(p), mm.sum())]; v[mm] = pick[:, :4]; mod[mm] = pick[:, 4].astype(np.int32)
    fields = {f_: t[f_] for f_ in t.fields if f_ not in ("mc_vtx", "truth_vtx_module", "truth_is_fiducial", "nuwro_wcomp")}
    w_ = wt[m_] * (ak.to_numpy(t["gen_relweight"]) if "gen_relweight" in t.fields else 1.0)  # generator per-event weights (GiBUU events above the ceiling) fold into the sample weight
    parts.append(ak.Array({**fields, "mc_vtx": v, "truth_vtx_module": mod, "truth_is_fiducial": in_volume(v, "fiducial"), "nuwro_wcomp": w_})); del t
    print(f"  {pathlib.Path(f).name}: kept {m_.sum():,}", flush=True)
inc = ak.concatenate(parts); ak.to_parquet(inc, out / f"{a.name}.truth.parquet")
frac = scale / norm  # fraction of the open-dataset exposure (norm = open-dataset nu_mu CC events on the nuclei NuWro covers)
W = float(ak.sum(inc["nuwro_wcomp"])); Wo = sum(r_["open_events"] for r_ in rows)
summ = {}
for r_ in rows:
    g = (r_["flavour"], r_["current"]); s_ = summ.setdefault(g, [0.0, 0]); s_[0] += r_["kept"] * r_["weight"]; s_[1] += r_["open_events"]
print(f"inclusive NuWro sample: {len(inc):,} events, {100 * frac:.2f}% of the open-dataset exposure; scale set by "
      f"{'the cap' if limiting[0] == 'cap' else f'{FLAV[limiting[0]]} {chr(67) + chr(67) if limiting[1] == 1 else chr(78) + chr(67)} Z={limiting[2]}'}; {sum(1 for r_ in rows if r_['weight'] > 1.001)} weighted strata")
print(f"{'':16s} {'NuWro':>8s} {'open dataset (GENIE)':>22s}")
for g, (wn, no) in sorted(summ.items()): print(f"{g[0]:8s} {g[1]:3s}    {100 * wn / W:7.2f}%   {100 * no / Wo:9.2f}%")
for r_ in rows:
    if r_["weight"] > 1.001: print(f"  weighted: {r_['flavour']} {r_['current']} Z={r_['Z']}: {r_['generated']} generated, weight {r_['weight']:.2f}")
json.dump({"n_events": len(inc), "exposure_fraction": frac, "scale_numu_cc": scale, "limiting": ["cap", 0, 0] if limiting[0] == "cap" else [FLAV[limiting[0]], limiting[1], limiting[2]],
           "open_events_all": n_open_all, "composition": {f"{g[0]}_{g[1]}": {"nuwro": wn / W, "open": no / Wo} for g, (wn, no) in summ.items()}, "strata": rows},
          open(out / f"{a.name}.json", "w"), indent=1)
