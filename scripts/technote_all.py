#!/usr/bin/env python
"""Technote inputs for the all-events models (CC and NC, all flavours): population composition, performance of the
all-events model A by subset next to the CC nu_mu model A, the 2p2h holdout of both model pairs, the calibration of A,
and NuWro / paired GENIE through both A models. Writes tables and macros (prefix \\nAE, no digits) to reports/all_events/.
Usage: scripts/technote_all.py [--out reports/all_events]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.data.dataset import split_by_subrun

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/all_events"); ap.add_argument("--cache", default="data/cache_v2_ke10n_all")
ap.add_argument("--cc-model", default="reports/m3_1A_z"); ap.add_argument("--all-model", default="reports/m3_1A_all")
ap.add_argument("--holdout-cc", default="reports/holdout_2p2h_1A_z"); ap.add_argument("--holdout-all", default="reports/holdout_2p2h_1A_all")
ap.add_argument("--cal-cc", default="reports/bayes_1A_z/bayes_uncertainty_A.json"); ap.add_argument("--cal-all", default="reports/bayes_1A_all/bayes_uncertainty_A_ccctl.json"); ap.add_argument("--cal-all-mixed", default="reports/bayes_1A_all/bayes_uncertainty_A.json")
ap.add_argument("--closure-all", default="reports/all_events/closure.json"); ap.add_argument("--closure-cc", default="reports/m3_1A_z/closure_ccinc.json")
ap.add_argument("--nuwro-cc", default="reports/nuwro_compare/summary.json"); ap.add_argument("--nuwro-all", default="reports/nuwro_compare_all/summary.json")
a = ap.parse_args(); out = pathlib.Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(p))
mac = {}
def m(name, v): assert not any(c.isdigit() for c in name), name; mac[name] = v
f3 = lambda x: f"{x:.3f}"; pct = lambda x: f"{100 * x:.1f}\\%"

# ---- population composition (training volume), all splits and the test split ----
cur, inc, rex, sub = [], [], [], []
for f in sorted(glob.glob(f"{a.cache}/*.npz")):
    z = np.load(f); cur.append(z["current"]); inc.append(z["incoming"]); rex.append(z["reco_exists"]); sub.append(z["subrun"])
cur, inc, rex, sub = map(np.concatenate, (cur, inc, rex, sub)); split = split_by_subrun(sub, seed=0)
cats = [("CC $\\nu_\\mu$", (cur == 1) & (inc == 14), "CCnumu"), ("NC (all flavours)", cur == 2, "NC"), ("CC $\\bar\\nu_\\mu$", (cur == 1) & (inc == -14), "CCnumubar"),
        ("CC $\\nu_e$, $\\bar\\nu_e$", (cur == 1) & (np.abs(inc) == 12), "CCnue"), ("other", (cur != 1) & (cur != 2), "Other")]
rows = []
for lab, s, key in cats:
    rows.append(f"{lab} & {s.sum():,} & {100 * s.mean():.2f}\\% & {rex[s].mean():.3f} & {(s & (split == 2)).sum():,} & {(s & (split == 2) & rex).sum():,} \\\\")
    m(f"nAEFrac{key}", f"{100 * s.mean():.1f}\\%"); m(f"nAEReco{key}", f"{rex[s].mean():.2f}"); m(f"nAETestReco{key}", f"{(s & (split == 2) & rex).sum():,}")
rows.append("\\midrule")
rows.append(f"all & {len(cur):,} & 100\\% & {rex.mean():.3f} & {(split == 2).sum():,} & {((split == 2) & rex).sum():,} \\\\")
m("nAEnEvents", f"{len(cur) / 1e6:.1f}\\,M"); m("nAEnTrain", f"{(split == 0).sum() / 1e6:.1f}\\,M"); m("nAEnTest", f"{(split == 2).sum():,}")
m("nAEnRecoNC", f"{(rex & (cur == 2)).sum() / 1e6:.2f}\\,M"); m("nAEFracRecoNC", pct((rex & (cur == 2)).sum() / rex.sum()))
(out / "tables" / "composition.tex").write_text("\\begin{tabular}{lrrrrr}\n\\toprule\n & events & share & reco fraction & test events & test, reconstructed \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rc = out / "reference_categories.json"  # written by the reference-category count over the open dataset (all events, training volume)
if rc.exists():
    r = J(rc); tot = sum(r.values())
    for k, K in (("mu-", "Mu"), ("lepton", "Lepton"), ("hadron", "Hadron"), ("beam", "Beam")): m(f"nAERef{K}", pct(r[k] / tot))

# ---- performance: CC nu_mu model A vs all-events model A by subset ----
cols = [("CC $\\nu_\\mu$ model,\\\\ CC $\\nu_\\mu$ test", a.cc_model), ("all-events model,\\\\ CC $\\nu_\\mu$ test", a.all_model + "_ccnumu"),
        ("all-events model,\\\\ NC test", a.all_model + "_nc"), ("all-events model,\\\\ other CC test", a.all_model + "_other"), ("all-events model,\\\\ full test", a.all_model)]
P = [(J(f"{d}/metrics.json"), J(f"{d}/metrics_tier2.json")) for _, d in cols]
def row(lab, fn, fmt=f3):
    return lab + " & " + " & ".join(fmt(fn(M, T)) for M, T in P) + " \\\\"
perf = [row("test events", lambda M, T: M["n_test"], lambda x: f"{x:,}"),
        row("reconstructed fraction (data)", lambda M, T: M["tier0"]["reco_exists"]["rate"]),
        row("log loss: reconstructed", lambda M, T: M["tier0"]["reco_exists"]["logloss"]),
        row("log loss: MINOS match", lambda M, T: M["tier0"]["minos_ok"]["logloss"]),
        row("log loss: prong multiplicity", lambda M, T: M["multiplicity"]["logloss"]),
        "\\midrule",
        row("closure AUC, reco only", lambda M, T: M["classifier_auc_marginal"]),
        row("closure AUC, truth + reco", lambda M, T: M["classifier_auc_conditional"]),
        row("\\quad muon block", lambda M, T: M["classifier_auc_conditional_by_block"]["muon"]),
        row("\\quad vertex block", lambda M, T: M["classifier_auc_conditional_by_block"]["vertex"]),
        row("\\quad calorimetry block", lambda M, T: M["classifier_auc_conditional_by_block"]["calorimetry"]),
        row("prong closure AUC (summary / + truth)", lambda M, T: (T["closure_prongs"]["event_summary_only"], T["closure_prongs"]["truth_plus_event_summary"]), lambda x: f"{x[0]:.3f} / {x[1]:.3f}"),
        "\\midrule",
        row("$W_1$ $\\log P$ ratio (model space)", lambda M, T: M["tier1_model_space"]["log_P_ratio"]["w1"]),
        row("$W_1$ $\\log E_\\mathrm{recoil}$ (model space)", lambda M, T: M["tier1_model_space"]["log_recoil_E"]["w1"])]
head = "model & CC $\\nu_\\mu$ & \\multicolumn{4}{c}{all events} \\\\\n test sample & " + " & ".join(c.split("\\\\ ")[1].replace(" test", "") for c, _ in cols) + " \\\\"
(out / "tables" / "performance.tex").write_text("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(perf) + "\n\\bottomrule\n\\end{tabular}\n")
for key, (M, T) in zip(["CCModel", "AllCC", "AllNC", "AllOther", "AllFull"], P):
    m(f"nAEAUCmarg{key}", f3(M["classifier_auc_marginal"])); m(f"nAEAUCcond{key}", f3(M["classifier_auc_conditional"]))
    m(f"nAEAUCcalo{key}", f3(M["classifier_auc_conditional_by_block"]["calorimetry"])); m(f"nAELLreco{key}", f3(M["tier0"]["reco_exists"]["logloss"]))
    m(f"nAELLmult{key}", f3(M["multiplicity"]["logloss"])); m(f"nAEProngAUC{key}", f3(T["closure_prongs"]["truth_plus_event_summary"]))
    m(f"nAEnTest{key}", f"{M['n_test']:,}")

# ---- 2p2h holdout: CC nu_mu pair vs all-events pair ----
H = {"cc": J(f"{a.holdout_cc}/summary.json"), "all": J(f"{a.holdout_all}/summary.json")}
U = {"cc": J(f"{a.holdout_cc}/uncertainty.json"), "all": J(f"{a.holdout_all}/uncertainty.json")}
hrows = [("2p2h test events", lambda h: h["B_on_2p2h"]["n_test"] if "n_test" in h["B_on_2p2h"] else float("nan"), lambda x: f"{x:,}"),
         ("log loss: reconstructed, A / B", lambda h: (h["A_on_2p2h"]["reco_exists_logloss"], h["B_on_2p2h"]["reco_exists_logloss"]), lambda x: f"{x[0]:.3f} / {x[1]:.3f}"),
         ("log loss: multiplicity, A / B", lambda h: (h["A_on_2p2h"]["mult_logloss"], h["B_on_2p2h"]["mult_logloss"]), lambda x: f"{x[0]:.3f} / {x[1]:.3f}"),
         ("closure AUC truth + reco, A / B", lambda h: (h["A_on_2p2h"]["auc_conditional"], h["B_on_2p2h"]["auc_conditional"]), lambda x: f"{x[0]:.3f} / {x[1]:.3f}"),
         ("\\quad calorimetry block, A / B", lambda h: (h["A_on_2p2h"]["auc_cond_calo"], h["B_on_2p2h"]["auc_cond_calo"]), lambda x: f"{x[0]:.3f} / {x[1]:.3f}"),
         ("$W_1$ recoil, A / B", lambda h: (h["A_on_2p2h"]["w1_recoil"], h["B_on_2p2h"]["w1_recoil"]), lambda x: f"{x[0]:.3f} / {x[1]:.3f}"),
         ("B on its CC $\\nu_\\mu$ control: closure AUC truth + reco", lambda h: h["B_on_control"]["auc_conditional"], f3)]
ht = [f"{lab} & {fmt(fn(H['cc']))} & {fmt(fn(H['all']))} \\\\" for lab, fn, fmt in hrows]
for k in ("cc", "all"):
    u = U[k]["auc_data_vs_B_cond"]; v = U[k]["auc_data_vs_A_cond"]
    pair = "CC $\\nu_\\mu$" if k == "cc" else "all-events"
    ht.append(f"bootstrap, B vs data (truth + reco) [{pair} pair] & \\multicolumn{{2}}{{c}}{{{u['point']:.3f} [{u['boot_lo']:.3f}, {u['boot_hi']:.3f}]; A: {v['point']:.3f} [{v['boot_lo']:.3f}, {v['boot_hi']:.3f}]}} \\\\")
(out / "tables" / "holdout.tex").write_text("\\begin{tabular}{lcc}\n\\toprule\n & CC $\\nu_\\mu$ models & all-events models \\\\\n\\midrule\n" + "\n".join(ht) + "\n\\bottomrule\n\\end{tabular}\n")
for k, K in (("cc", "CC"), ("all", "All")):
    h = H[k]
    for mm, MM in (("A_on_2p2h", "A"), ("B_on_2p2h", "B")):
        m(f"nAEHo{K}{MM}cond", f3(h[mm]["auc_conditional"])); m(f"nAEHo{K}{MM}calo", f3(h[mm]["auc_cond_calo"])); m(f"nAEHo{K}{MM}Wrecoil", f3(h[mm]["w1_recoil"]))
        m(f"nAEHo{K}{MM}LLmult", f3(h[mm]["mult_logloss"]))
    m(f"nAEHo{K}Ctlcond", f3(h["B_on_control"]["auc_conditional"]))
    u = U[k]["auc_data_vs_B_cond"]; m(f"nAEHo{K}BcondLo", f3(u["boot_lo"])); m(f"nAEHo{K}BcondHi", f3(u["boot_hi"]))
    u = U[k]["auc_data_vs_A_cond"]; m(f"nAEHo{K}AcondLo", f3(u["boot_lo"])); m(f"nAEHo{K}AcondHi", f3(u["boot_hi"]))

# ---- calibration of model A (epistemic pulls, test control) ----
for k, K, p in (("cc", "CC", a.cal_cc), ("all", "All", a.cal_all), ("mixed", "Mixed", a.cal_all_mixed)):
    c = J(p)
    for part, PP in (("calibration_control_marginal", "Marg"), ("calibration_control", "Cond")):
        m(f"nAECal{K}{PP}RMS", f"{c[part]['pull_rms']:.2f}"); m(f"nAECal{K}{PP}Within", pct(c[part]["frac_abs_pull_lt2"]).replace(".0\\%", "\\%"))
        m(f"nAECal{K}{PP}Bins", f"{c[part]['n_bins']}")

cal = {k: J(p) for k, p in (("cc", a.cal_cc), ("all", a.cal_all), ("mixed", a.cal_all_mixed))}
for k, K in (("cc", "CC"), ("all", "All"), ("mixed", "Mixed")):
    c = cal[k]
    for part, PP in (("calibration_heldout_marginal", "TwoPMarg"), ("calibration_heldout", "TwoPCond")):
        m(f"nAECal{K}{PP}RMS", f"{c[part]['pull_rms']:.2f}"); m(f"nAECal{K}{PP}Within", pct(c[part]["frac_abs_pull_lt2"]).replace(".0\\%", "\\%"))
        m(f"nAECal{K}{PP}Bins", f"{c[part]['n_bins']}")
    for f, F in (("exist", "Exist"), ("card", "Card"), ("flow", "Flow")): m(f"nAEFlag{K}{F}AUC", f3(c["flags"][f]["ood_auc"]))
crow = lambda lab, fn: lab + " & " + " & ".join(fn(cal[k]) for k in ("cc", "all", "mixed")) + " \\\\"
pr = lambda part: (lambda c: f"{c[part]['pull_rms']:.2f} ({c[part]['n_bins']}, {100 * c[part]['frac_abs_pull_lt2']:.0f}\\%)")
ctab = [crow("held-out 2p2h events", lambda c: f"{c['n_ood']:,}"), crow("pull RMS, 2p2h, conditional bins", pr("calibration_heldout")),
        crow("pull RMS, 2p2h, marginal bins", pr("calibration_heldout_marginal")), crow("pull RMS, control, conditional bins", pr("calibration_control")),
        crow("pull RMS, control, marginal bins", pr("calibration_control_marginal")), "\\midrule"] + \
       [crow(f"2p2h-vs-control detection AUC, {lab}", lambda c, f=f: f3(c["flags"][f]["ood_auc"])) for f, lab in (("exist", "reconstruction flag"), ("card", "multiplicity"), ("flow", "event flow"))]
(out / "tables" / "calibration.tex").write_text("\\begin{tabular}{lccc}\n\\toprule\n & CC $\\nu_\\mu$ model A & \\multicolumn{2}{c}{all-events model A} \\\\\n & CC $\\nu_\\mu$ control & CC $\\nu_\\mu$ control & all-events control \\\\\n\\midrule\n" + "\n".join(ctab) + "\n\\bottomrule\n\\end{tabular}\n")
for k, K in (("cc", "CC"), ("all", "All")):
    for key, KK in (("auc_A_vs_B_cond", "ABcond"), ("auc_A_vs_B_marg", "ABmarg"), ("auc_data_vs_A_marg", "Amarg"), ("auc_data_vs_B_marg", "Bmarg"), ("auc_data_vs_A_cond", "AcondBoot"), ("auc_data_vs_B_cond", "BcondBoot")):
        u = U[k][key]; m(f"nAEHo{K}{KK}", f3(u["point"])); m(f"nAEHo{K}{KK}Lo", f3(u["boot_lo"])); m(f"nAEHo{K}{KK}Hi", f3(u["boot_hi"]))

# ---- non-calorimetry closure (seven variables), same recipe as the CC nu_mu table ----
if pathlib.Path(a.closure_all).exists():
    C = J(a.closure_all); CC = J(a.closure_cc)
    lab = {"A_full": "all-events A, full test", "A_ccnumu": "all-events A, CC $\\nu_\\mu$ test", "A_nc": "all-events A, NC test", "A_other": "all-events A, other CC test",
           "A_twop": "all-events A, held-out 2p2h", "B_full": "all-events B, full test", "B_ctl": "all-events B, CC $\\nu_\\mu$ non-2p2h control", "B_twop": "all-events B, held-out 2p2h"}
    rows = [f"CC $\\nu_\\mu$ A, CC $\\nu_\\mu$ test & {CC['A_full']['all_marg']:.3f} & {CC['A_full']['all_cond']:.3f} & {CC['A_full']['ccinc_marg']:.3f} & {CC['A_full']['ccinc_cond']:.3f} \\\\",
            f"CC $\\nu_\\mu$ A, held-out 2p2h & {CC['A_twop']['all_marg']:.3f} & {CC['A_twop']['all_cond']:.3f} & {CC['A_twop']['ccinc_marg']:.3f} & {CC['A_twop']['ccinc_cond']:.3f} \\\\",
            f"CC $\\nu_\\mu$ B, held-out 2p2h & {CC['B_twop']['all_marg']:.3f} & {CC['B_twop']['all_cond']:.3f} & {CC['B_twop']['ccinc_marg']:.3f} & {CC['B_twop']['ccinc_cond']:.3f} \\\\", "\\midrule"]
    rows += [f"{lab[k]} & {C[k]['all_marg']:.3f} & {C[k]['all_cond']:.3f} & {C[k]['ccinc_marg']:.3f} & {C[k]['ccinc_cond']:.3f} \\\\" for k in lab if k in C]
    (out / "tables" / "closure.tex").write_text("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{all nine variables} & \\multicolumn{2}{c}{non-calorimetry set} \\\\\n & marginal & + truth & marginal & + truth \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    for k, v in C.items():
        K = {"A_full": "AFull", "A_ccnumu": "ACC", "A_nc": "ANC", "A_other": "AOther", "A_twop": "ATwoP", "B_full": "BFull", "B_ctl": "BCtl", "B_twop": "BTwoP"}[k]
        for q, Q in (("all_marg", "AllMarg"), ("all_cond", "AllCond"), ("ccinc_marg", "CIMarg"), ("ccinc_cond", "CICond")): m(f"nAECL{K}{Q}", f3(v[q]))
        m(f"nAECL{K}N", f"{v['n']:,}")

# ---- NuWro and paired GENIE through both A models ----
N = {"cc": J(a.nuwro_cc), "all": J(a.nuwro_all)}
nrows = []
for k, lab in (("cc", "CC $\\nu_\\mu$ model A"), ("all", "all-events model A")):
    s = N[k]
    for src, sl in (("reco_genie", "surrogate on GENIE"), ("reco_nuwro", "surrogate on NuWro")):
        r = s[src]; nrows.append(f"{lab}, {sl} & {r['reco_frac']:.3f} & {r['minos_frac']:.3f} & {r['mean_recoil']:.0f} & {r['zero_nv100']:.3f} / {r['zero_blobs']:.3f} & {r['mean_npr']:.3f} \\\\")
r = N["cc"]["reco_ref"]; nrows.insert(0, f"open dataset (GENIE events) & {r['reco_frac']:.3f} & -- & {r['mean_recoil']:.0f} & {r['zero_nv100']:.3f} / {r['zero_blobs']:.3f} & {r['mean_npr']:.3f} \\\\"); nrows.insert(1, "\\midrule")
(out / "tables" / "nuwro.tex").write_text("\\begin{tabular}{lccccc}\n\\toprule\n & reco fraction & MINOS-matched & mean recoil $E$ [MeV] & zero non-vtx / blobs & mean prongs \\\\\n\\midrule\n" + "\n".join(nrows) + "\n\\bottomrule\n\\end{tabular}\n")
for k, K in (("cc", "CC"), ("all", "All")):
    s = N[k]; g, n = s["reco_genie"], s["reco_nuwro"]
    m(f"nAENW{K}GenReco", f3(g["reco_frac"])); m(f"nAENW{K}NuwReco", f3(n["reco_frac"])); m(f"nAENW{K}GenRecoil", f"{g['mean_recoil']:.0f}"); m(f"nAENW{K}NuwRecoil", f"{n['mean_recoil']:.0f}")
    m(f"nAENW{K}RecoilShift", f"{100 * (n['mean_recoil'] / g['mean_recoil'] - 1):.1f}\\%"); m(f"nAENW{K}EpiAUC", f"{max(v['auc'] for v in s['epistemic_flags'].values()):.3f}")
m("nAENWRefReco", f3(N["cc"]["reco_ref"]["reco_frac"])); m("nAENWRefRecoil", f"{N['cc']['reco_ref']['mean_recoil']:.0f}")

(out / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
(out / "summary.json").write_text(json.dumps(mac, indent=1))
print(f"{len(mac)} macros, tables in {out / 'tables'}")
