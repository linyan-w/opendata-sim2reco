#!/usr/bin/env python
"""Out-of-fiducial-volume validation: events generated over the surrogate's full training volume (true vertex
4000 <= z <= 8700 mm, tracker and outside it), selected on the RECONSTRUCTED vertex in the tracker fiducial volume.
Three samples: the open dataset's reconstruction of GENIE events (ref_* columns), the surrogate on the same GENIE
events, and the surrogate on NuWro (tracker nuclei + Pb + Fe campaigns, weighted per nucleus to the open dataset's
composition of the volume, `nuwro_wcomp`). Questions: does the surrogate reproduce how many reco-fiducial events come
from outside the fiducial volume, from where, on which nucleus, and what they look like; does it flag them as less
certain. Outputs figures, tables, macros and a JSON in reports/outfv/.
Usage: scripts/fullvolume_compare.py [--genie-truth ... --genie-sur ... --nuwro-truth ... --nuwro-sur ... --out ...]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np, uproot
from sklearn.metrics import roc_auc_score
from sim2reco.prep.frames import theta_phi_beam
from sim2reco.prep.volumes import in_volume, in_hexagon, Z_FID
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); D = "data/fullvolume"
ap.add_argument("--genie-truth", default=f"{D}/genie_fullvol.truth.parquet"); ap.add_argument("--genie-sur", default=f"{D}/surrogate_Az_genie_fullvol.root")
ap.add_argument("--nuwro-truth", default=f"{D}/nuwro_fullvol.truth.parquet"); ap.add_argument("--nuwro-sur", default=f"{D}/surrogate_Az_nuwro_fullvol.root")
ap.add_argument("--out", default="reports/outfv"); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True)
C = {"ref": plots.PALETTE["real"], "genie": plots.PALETTE["model"], "nuwro": plots.PALETTE["third"]}
LAB = {"ref": "open dataset (GENIE events)", "genie": "surrogate on GENIE", "nuwro": "surrogate on NuWro"}
REG = ["fiducial", "upstream", "downstream", "transverse"]
REGLAB = {"fiducial": "fiducial tracker", "upstream": r"upstream ($z<5990$ mm)", "downstream": r"downstream ($z>8340$ mm)", "transverse": "outside the hexagon"}
NUC = [(6, "C"), (82, "Pb"), (26, "Fe"), (1, "H"), (8, "O"), (-1, "other")]


def region(v):
    r = np.full(len(v), 3); z = v[:, 2]
    r[z <= Z_FID[0]] = 1; r[z >= Z_FID[1]] = 2; r[in_volume(v, "fiducial")] = 0
    return r  # 0 fiducial, 1 upstream, 2 downstream, 3 transverse (fiducial z, outside the hexagon)


def key(run, sub, nth): return np.asarray(run, np.int64) * 10 ** 12 + np.asarray(sub, np.int64) * 10 ** 7 + np.asarray(nth, np.int64)


def load(truth_path, sur_path, has_ref):
    t = ak.from_parquet(truth_path); n = len(t)
    v = ak.to_numpy(t["mc_vtx"])[:, :3]; mu = ak.to_numpy(t["mc_primFSLepton"]); th, _ = theta_phi_beam(mu[:, 0], mu[:, 1], mu[:, 2])
    w = ak.to_numpy(t["nuwro_wcomp"]) if "nuwro_wcomp" in t.fields else np.ones(n)
    T = {"n": n, "w": w, "vtx": v, "reg": region(v), "Z": ak.to_numpy(t["mc_targetZ"]), "muP": np.linalg.norm(mu[:, :3], axis=1) / 1e3, "muth": th, "ch": ak.to_numpy(t["mc_intType"])}
    tk = key(t["mc_run"], t["mc_subrun"], t["mc_nthEvtInFile"]); order = np.argsort(tk)
    s = uproot.open(sur_path)["MasterAnaDev"]
    cols = ["mc_run", "mc_subrun", "mc_nthEvtInFile", "MasterAnaDev_vtx", "MasterAnaDev_muon_P", "MasterAnaDev_muon_theta", "MasterAnaDev_recoil_E", "n_prongs",
            "surrogate_epi_logit_reco_exists", "surrogate_epi_flow_flag", "surrogate_epi_multiplicity_logits", "surrogate_epi_vtx_z", "surrogate_epi_p_reco_exists"]
    S = s.arrays(cols, library="np"); sk = key(S["mc_run"], S["mc_subrun"], S["mc_nthEvtInFile"])
    pos = np.searchsorted(tk[order], sk); row = order[pos]; assert np.all(tk[row] == sk) and len(np.unique(row)) == len(row)
    sur = {"row": row, "vtx": np.stack(S["MasterAnaDev_vtx"])[:, :3], "P": S["MasterAnaDev_muon_P"] / 1e3, "th": S["MasterAnaDev_muon_theta"], "recoil": S["MasterAnaDev_recoil_E"], "npr": S["n_prongs"].astype(float),
           "epi_exist": S["surrogate_epi_logit_reco_exists"], "epi_flow": S["surrogate_epi_flow_flag"], "epi_card": S["surrogate_epi_multiplicity_logits"], "epi_vz": S["surrogate_epi_vtx_z"], "epi_preco": S["surrogate_epi_p_reco_exists"]}
    ref = None
    if has_ref:
        ex = ak.to_numpy(t["ref_reco_exists"]); r = np.where(ex)[0]
        ref = {"row": r, "vtx": ak.to_numpy(t["ref_vtx"])[r], "P": ak.to_numpy(t["ref_muon_P"])[r] / 1e3, "th": ak.to_numpy(t["ref_muon_theta"])[r], "recoil": ak.to_numpy(t["ref_recoil_E"])[r], "npr": ak.to_numpy(t["ref_n_prongs"])[r].astype(float) + 1}  # + muon track, as n_prongs in the ntuple
    for X in (sur, ref):
        if X is not None: X["fid"] = in_volume(X["vtx"], "fiducial"); X["w"] = T["w"][X["row"]]; X["reg"] = T["reg"][X["row"]]; X["Z"] = T["Z"][X["row"]]
    return T, sur, ref


print("loading GENIE", flush=True); TG, SG, RG = load(a.genie_truth, a.genie_sur, True)
print("loading NuWro", flush=True); TN, SN, _ = load(a.nuwro_truth, a.nuwro_sur, False)
SAMPLES = [("ref", TG, RG), ("genie", TG, SG), ("nuwro", TN, SN)]
R = {"n_genie": int(TG["n"]), "n_nuwro": int(TN["n"])}


def wsum(w, m): return float(w[m].sum())


# ---------- 1. where do the reco-fiducial events come from (true region, nucleus)
tab = {}
for g, T, S in SAMPLES:
    W = T["w"].sum(); f = S["fid"]; wf = S["w"][f]; Wf = wf.sum(); reg, Z = S["reg"][f], S["Z"][f]
    d = {"recofid_per_truth": Wf / W, "recofid_per_truefid": Wf / wsum(T["w"], T["reg"] == 0),
         "region": {r: float(wf[reg == i].sum() / Wf) for i, r in enumerate(REG)},
         "out_by_nucleus": {}, "eff_truefid": wsum(S["w"], f & (S["reg"] == 0)) / wsum(T["w"], T["reg"] == 0)}
    out_ = reg != 0; Wo = wf[out_].sum(); known = np.zeros(out_.sum(), bool)
    for z, nm in NUC[:-1]: m = Z[out_] == z; known |= m; d["out_by_nucleus"][nm] = float(wf[out_][m].sum() / Wo)
    d["out_by_nucleus"]["other"] = float(wf[out_][~known].sum() / Wo); d["out_frac"] = float(Wo / Wf)
    # stat error on the out-of-FV fraction (weighted binomial)
    p = d["out_frac"]; neff = Wf ** 2 / (wf ** 2).sum(); d["out_frac_err"] = float(np.sqrt(p * (1 - p) / neff)); tab[g] = d
R["origin"] = tab
tex = "\\begin{tabular}{lcccccccc}\n\\toprule\n & reco-FV per true-FV event & true outside FV & upstream & downstream & outside hexagon & of which Pb & Fe & C \\\\\n\\midrule\n"
for g, _, _ in SAMPLES:
    d = tab[g]; tex += f"{LAB[g]} & {d['recofid_per_truefid']:.3f} & {100*d['out_frac']:.1f}\\,$\\pm$\\,{100*d['out_frac_err']:.1f}\\% & {100*d['region']['upstream']:.1f}\\% & {100*d['region']['downstream']:.1f}\\% & {100*d['region']['transverse']:.1f}\\% & {100*d['out_by_nucleus']['Pb']:.0f}\\% & {100*d['out_by_nucleus']['Fe']:.0f}\\% & {100*d['out_by_nucleus']['C']:.0f}\\% \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"; (out / "tables" / "origin.tex").write_text(tex)


# ---------- 2. true-vertex distributions of reco-fiducial events, three-way with ratios
def hist(v, w, e): h = np.histogram(v, e, weights=w)[0]; e2 = np.histogram(v, e, weights=w ** 2)[0]; return h, np.sqrt(e2)


def three_way(specs, sel_fn, fname, title=None, norm="truth"):
    fig, axs = plots.plt.subplots(2, len(specs), figsize=(4 * len(specs), 5.6), gridspec_kw={"height_ratios": (2, 1), "hspace": 0.08}, squeeze=False); res = {}
    for j, (fn, e, xl, logy) in enumerate(specs):
        ax, axr = axs[0, j], axs[1, j]; H = {}
        for g, T, S in SAMPLES:
            m = sel_fn(S); x = fn(T, S)[m]; w = S["w"][m]; ok = np.isfinite(x); h, err = hist(x[ok], w[ok], e)
            N = T["w"].sum() if norm == "truth" else w.sum(); H[g] = (h / N, err / N); c = 0.5 * (e[:-1] + e[1:])
            ax.stairs(H[g][0], e, color=C[g], lw=1.3, label=LAB[g]); ax.errorbar(c, H[g][0], H[g][1], fmt="none", color=C[g], lw=0.8)
        for g, den in (("genie", "ref"), ("nuwro", "genie")):
            num, dn = H[g], H[den]; ok = dn[0] > 0; r = np.where(ok, num[0] / np.where(ok, dn[0], 1), np.nan)
            re = np.where(ok, r * np.sqrt((num[1] / np.maximum(num[0], 1e-15)) ** 2 + (dn[1] / np.where(ok, dn[0], 1)) ** 2), np.nan)
            axr.errorbar(0.5 * (e[:-1] + e[1:]), r, re, fmt="o", ms=2.5, color=C[g], label="surrogate(GENIE) / open dataset" if g == "genie" else "surrogate(NuWro) / surrogate(GENIE)")
        axr.axhline(1, color="k", lw=0.6); axr.set_ylim(0.3, 1.7); axr.set_xlabel(xl); ax.set_xticklabels([])
        if logy: ax.set_yscale("log")
        res[xl] = {g: float(H[g][0].sum()) for g in H}
    axs[0, 0].set_ylabel("per truth event" if norm == "truth" else "normalised"); axs[1, 0].set_ylabel("ratio"); axs[0, 0].legend(frameon=False, fontsize=7); axs[1, 0].legend(frameon=False, fontsize=6.5)
    if title: fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    plots.save(fig, out / "figures" / fname); return res


def hexdist(v):  # transverse "radius" in units of the hexagon apothem (1 = edge of the fiducial hexagon)
    x, y = np.abs(v[:, 0]), np.abs(v[:, 1]); return np.maximum(x, (np.sqrt(3) * y + x) / 2) / 850.0


fid = lambda S: S["fid"]
three_way([(lambda T, S: T["vtx"][S["row"], 2], np.linspace(4000, 8700, 48), "true vertex $z$ [mm]", True),
           (lambda T, S: hexdist(T["vtx"][S["row"]]), np.linspace(0, 1.6, 33), "true transverse position / apothem", True),
           (lambda T, S: S["vtx"][:, 2] - T["vtx"][S["row"], 2], np.linspace(-600, 600, 49), r"reco $-$ true vertex $z$ [mm]", True)],
          fid, "true_vertex_recofid.png", "events with the reconstructed vertex in the fiducial volume")
# ---------- 3. what the migrated events look like (true vertex outside, reco vertex inside)
mig = lambda S: S["fid"] & (S["reg"] != 0)
R["migrated_shapes"] = three_way([(lambda T, S: S["P"], np.linspace(0, 20, 41), "reco muon momentum [GeV]", True), (lambda T, S: S["th"], np.linspace(0, 0.6, 31), "reco muon angle [rad]", False),
                                  (lambda T, S: np.log10(np.clip(S["recoil"], 1, None)), np.linspace(1, 4.3, 34), r"$\log_{10}$ reco recoil $E$ [MeV]", True), (lambda T, S: S["npr"], np.arange(0.5, 9.5, 1), "reco prong count", True)],
                                 mig, "migrated_reco.png", "true vertex outside the fiducial volume, reconstructed inside it")
three_way([(lambda T, S: S["P"], np.linspace(0, 20, 41), "reco muon momentum [GeV]", True), (lambda T, S: S["th"], np.linspace(0, 0.6, 31), "reco muon angle [rad]", False),
           (lambda T, S: np.log10(np.clip(S["recoil"], 1, None)), np.linspace(1, 4.3, 34), r"$\log_{10}$ reco recoil $E$ [MeV]", True), (lambda T, S: S["npr"], np.arange(0.5, 9.5, 1), "reco prong count", True)],
          fid, "recofid_reco.png", "all events with the reconstructed vertex in the fiducial volume")

# ---------- 4. out-of-FV fraction vs reco observables (the contamination a reco-FV analysis carries)
fig, axs = plots.plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (fn, e, xl) in zip(axs, [(lambda S: S["P"], np.linspace(0, 20, 21), "reco muon momentum [GeV]"), (lambda S: np.log10(np.clip(S["recoil"], 1, None)), np.linspace(1, 4.3, 23), r"$\log_{10}$ reco recoil $E$ [MeV]"),
                                 (lambda S: S["vtx"][:, 2], np.linspace(Z_FID[0], Z_FID[1], 24), "reco vertex $z$ [mm]")]):
    for g, T, S in SAMPLES:
        m = S["fid"]; x = fn(S)[m]; w = S["w"][m]; o = (S["reg"][m] != 0)
        a_ = np.histogram(x, e, weights=w)[0]; b_ = np.histogram(x, e, weights=w * o)[0]; n2 = np.histogram(x, e, weights=w ** 2)[0]
        p = np.where(a_ > 0, b_ / np.maximum(a_, 1e-12), np.nan); err = np.sqrt(np.clip(p * (1 - p), 0, None) * n2) / np.maximum(a_, 1e-12)
        ax.errorbar(0.5 * (e[:-1] + e[1:]), p, err, fmt="o-", ms=2.5, lw=1, color=C[g], label=LAB[g])
    ax.set_xlabel(xl); ax.set_ylim(0, None)
axs[0].set_ylabel("fraction with true vertex outside FV"); axs[0].legend(frameon=False, fontsize=7); plots.save(fig, out / "figures" / "outfv_fraction.png")

# ---------- 5. does the model flag migrated events as less certain? (surrogate on GENIE: true-out vs true-in, among reco-FV)
E = {}
for k in ("epi_exist", "epi_card", "epi_flow", "epi_vz"):
    m = SG["fid"]; v = SG[k][m]; o = SG["reg"][m] != 0; ok = np.isfinite(v)
    E[k] = {"median_in": float(np.median(v[ok & ~o])), "median_out": float(np.median(v[ok & o])), "auc_out_vs_in": float(roc_auc_score(o[ok], v[ok]))}
    mn = SN["fid"] & (SN["reg"] != 0); vg, vn = v[ok & o], SN[k][mn]; vn = vn[np.isfinite(vn)]
    E[k]["auc_nuwro_vs_genie_out"] = float(roc_auc_score(np.r_[np.zeros(len(vg)), np.ones(len(vn))], np.r_[vg, vn]))
R["epistemic"] = E
fig, axs = plots.plt.subplots(1, 4, figsize=(16, 3.4))
for ax, (k, xl) in zip(axs, (("epi_exist", "reco-flag logit variance"), ("epi_card", "multiplicity-logit variance"), ("epi_flow", "event-flow epistemic flag"), ("epi_vz", "vertex-$z$ posterior spread [mm]"))):
    m = SG["fid"]; v = SG[k][m]; o = SG["reg"][m] != 0; allv = v[np.isfinite(v) & (v > 0)]; b = np.logspace(np.log10(np.percentile(allv, 0.5)), np.log10(np.percentile(allv, 99.5)), 50)
    for sel, lab, ls in ((~o, "true vertex in FV", "-"), (o, "true vertex outside FV", "--")): x = v[sel]; ax.hist(x[np.isfinite(x) & (x > 0)], b, histtype="step", density=True, color=C["genie"], ls=ls, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_title(f"AUC out vs in {E[k]['auc_out_vs_in']:.2f}", fontsize=9, loc="left")
axs[0].legend(frameon=False, fontsize=7); axs[0].set_ylabel("density (surrogate on GENIE, reco-FV)"); plots.save(fig, out / "figures" / "epistemic_outfv.png")

json.dump(R, open(out / "summary.json", "w"), indent=1, default=float)
M = {"nOFVnGenie": f"{TG['n']:,}".replace(",", "{,}"), "nOFVnNuwro": f"{TN['n']:,}".replace(",", "{,}")}
for g, nm in (("ref", "Ref"), ("genie", "Gen"), ("nuwro", "Nuw")):
    d = tab[g]; M[f"nOFV{nm}OutFrac"] = f"{100*d['out_frac']:.1f}\\%"; M[f"nOFV{nm}PerTrueFid"] = f"{d['recofid_per_truefid']:.3f}"
    for r in REG[1:]: M[f"nOFV{nm}{r.capitalize()}"] = f"{100*d['region'][r]:.1f}\\%"
    for nuc in ("Pb", "Fe", "C"): M[f"nOFV{nm}Out{nuc}"] = f"{100*d['out_by_nucleus'][nuc]:.0f}\\%"
for k, kn in (("epi_exist", "Exist"), ("epi_card", "Card"), ("epi_flow", "Flow"), ("epi_vz", "Vz")): M[f"nOFVEpi{kn}AUC"] = f"{E[k]['auc_out_vs_in']:.2f}"; M[f"nOFVEpi{kn}NuwAUC"] = f"{E[k]['auc_nuwro_vs_genie_out']:.2f}"
(out / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(json.dumps({"origin": tab, "epistemic": E}, indent=1, default=float))
