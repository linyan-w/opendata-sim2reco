#!/usr/bin/env python
"""Technote tables and macros of the ensemble section: the members' validation losses and closure, the between-member spread
against the residual on the test split (scripts/ensemble_spread.py -> reports/ensemble/spread.json), the per-event epistemic
columns of the ensemble file against the single-model (posterior-draw) file by region on the inclusive GENIE sample, and the
epistemic flag AUCs of the inclusive comparisons. Writes reports/ensemble/{macros.tex, tables/*.tex}.
Usage: scripts/technote_ensemble.py [--members reports/m3_1A_all reports/m3_1A_all_s1 reports/m3_1A_all_s2 reports/m3_1A_all_s3]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, uproot
from sim2reco.prep.volumes import in_volume
ap = argparse.ArgumentParser(); ap.add_argument("--members", nargs="+", default=["reports/m3_1A_all", "reports/m3_1A_all_s1", "reports/m3_1A_all_s2", "reports/m3_1A_all_s3"])
ap.add_argument("--single", default="data/inclusive/surrogate_Aall_genie_inclusive.root"); ap.add_argument("--ensemble", default="data/inclusive/surrogate_Eall_genie_inclusive.root"); ap.add_argument("--out", default="reports/ensemble"); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True); M = {}
# ---- members
rows = ["\\begin{tabular}{lcccccc}", "\\toprule", "member & val. loss: reco flag & val. loss: event flow & closure AUC marg. & closure AUC cond. & $W_1$ $\\log P$ ratio & $W_1$ $\\log E_\\mathrm{recoil}$ \\\\", "\\midrule"]
for i, md in enumerate(a.members):
    E = json.load(open(f"{md}/metrics.json")); h = json.load(open(f"{md}/history.json"))[-1]["val"]
    rows.append(f"{'deployed (seed 0)' if i == 0 else f'seed {i}'} & {h['exist']:.4f} & {h['flow']:.4f} & {E['classifier_auc_marginal']:.3f} & {E['classifier_auc_conditional']:.3f} & {E['tier1_model_space']['log_P_ratio']['w1']:.4f} & {E['tier1_model_space']['log_recoil_E']['w1']:.4f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]; (out / "tables" / "members.tex").write_text("\n".join(rows) + "\n"); M["EnsMembers"] = str(len(a.members))
# ---- spread vs residual on the test split
S = json.load(open(out / "spread.json")); M["EnsTestN"] = f"{S['n']:,}".replace(",", "{,}")
NAMES = {"marg_muP": "reco muon $P$ (marginal)", "marg_theta": "reco muon angle (marginal)", "marg_recoil": "reco recoil (marginal)", "marg_vz": "reco vertex $z$ (marginal)", "reco_by_pT": "$P(\\mathrm{reco})$ by true $p_T$", "sel_by_pT": "$P(\\mathrm{selected})$ by true $p_T$", "sel_by_pL": "$P(\\mathrm{selected})$ by true $p_L$", "selrec_by_pT": "selected events by reco $p_T$", "selrec_by_pL": "selected events by reco $p_L$"}
rows = ["\\begin{tabular}{lcccccc}", "\\toprule", "distribution & bins & median $|$resid$|$ & median spread & spread / stat & bins with spread $>$ $|$resid$|$ & pull RMS without / with \\\\", "\\midrule"]
for k, lab in NAMES.items():
    T = S["tables"][k]; r = T["rows"]; s = T["summary"]; res = np.array([x["resid"] / x["mean"] for x in r]); ens = np.array([x["ens"] / x["mean"] for x in r])
    rows.append(f"{lab} & {len(r)} & {100 * np.median(np.abs(res)):.2f}\\% & {100 * np.median(ens):.2f}\\% & {s['median_ens_over_stat']:.2f} & {100 * s['frac_ens_gt_resid']:.0f}\\% & {s['pull_rms_no_ens']:.2f} / {s['pull_rms_ens']:.2f} \\\\")
    if k in ("marg_muP", "marg_theta", "marg_recoil", "selrec_by_pT"): M[f"EnsPull{k.replace('_', '')}"] = f"{s['pull_rms_no_ens']:.2f} / {s['pull_rms_ens']:.2f}"; M[f"EnsCover{k.replace('_', '')}"] = f"{100 * s['frac_ens_gt_resid']:.0f}\\%"
rows += ["\\bottomrule", "\\end{tabular}"]; (out / "tables" / "spread.tex").write_text("\n".join(rows) + "\n")
rows = ["\\begin{tabular}{lrccc}", "\\toprule", " & events & muon $P$ & recoil & vertex $z$ [mm] \\\\", "\\midrule"]
for name, lab in (("true pT [GeV]", "true $p_T$ [GeV]"), ("true angle [deg]", "true angle [deg]")):
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{by {lab}}}}} \\\\")
    for r in S["per_event"][name]: rows.append(f"\\quad {r['bin']} & {r['n']:,} & {100 * r['relP']:.2f}\\% & {100 * r['relE']:.2f}\\% & {r['dz']:.1f} \\\\".replace(",", "{,}", 1) if False else f"\\quad {r['bin']} & {r['n']:,} & {100 * r['relP']:.2f}\\% & {100 * r['relE']:.2f}\\% & {r['dz']:.1f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]; (out / "tables" / "per_event_test.tex").write_text("\n".join(rows).replace(",", "{,}") .replace("{,}\\%", ",\\%") + "\n")
pe = S["per_event"]; M["EnsPerEvPbulk"] = f"{100 * pe['true angle [deg]'][0]['relP']:.2f}\\%"; M["EnsPerEvPwide"] = f"{100 * pe['true angle [deg]'][-1]['relP']:.1f}\\%"; M["EnsPerEvZwide"] = f"{pe['true angle [deg]'][-1]['dz']:.0f}"; M["EnsPerEvPhighpT"] = f"{100 * pe['true pT [GeV]'][-1]['relP']:.1f}\\%"
# ---- per-event epistemic columns on the inclusive GENIE file: single model (posterior draws) vs ensemble, by region
cols = ["MasterAnaDev_muon_P", "MasterAnaDev_muon_theta", "MasterAnaDev_recoil_E", "MasterAnaDev_vtx", "mc_vtx", "surrogate_epi_muon_P", "surrogate_epi_recoil_E", "surrogate_epi_vtx_z", "surrogate_epi_p_reco_exists"]
def regions(f):
    a_ = uproot.open(f)["MasterAnaDev"].arrays(cols, library="np"); P = a_["MasterAnaDev_muon_P"]; th = np.degrees(a_["MasterAnaDev_muon_theta"]); pT = P / 1e3 * np.sin(np.radians(th))
    rv = np.stack(a_["MasterAnaDev_vtx"]); tv = np.stack(a_["mc_vtx"]); fid = in_volume(rv, "fiducial"); outside = fid & ~in_volume(tv, "fiducial"); E = np.clip(a_["MasterAnaDev_recoil_E"], 1, None)
    sel = [("true vertex inside the fiducial volume", fid & ~outside), ("true vertex outside (migrated in)", outside), ("muon angle $< 10^\\circ$", fid & (th < 10)), ("muon angle $10$--$20^\\circ$", fid & (th >= 10) & (th < 20)), ("muon angle $> 20^\\circ$", fid & (th >= 20)), ("$p_T < 0.5$ GeV", fid & (pT < 0.5)), ("$p_T$ 1.5--2.5 GeV", fid & (pT >= 1.5) & (pT < 2.5)), ("$p_T > 2.5$ GeV", fid & (pT >= 2.5))]
    return [(lab, int(m.sum()), np.median(a_["surrogate_epi_muon_P"][m] / P[m]), np.median(a_["surrogate_epi_recoil_E"][m] / E[m]), np.median(a_["surrogate_epi_vtx_z"][m]), np.median(a_["surrogate_epi_p_reco_exists"][m])) for lab, m in sel]
Rs, Re = regions(a.single), regions(a.ensemble)
rows = ["\\begin{tabular}{lrcccccccc}", "\\toprule", "reconstructed-fiducial events & events & \\multicolumn{2}{c}{muon $P$} & \\multicolumn{2}{c}{recoil} & \\multicolumn{2}{c}{vertex $z$ [mm]} & \\multicolumn{2}{c}{$P(\\mathrm{reco})$} \\\\",
        " & & posterior & ensemble & posterior & ensemble & posterior & ensemble & posterior & ensemble \\\\", "\\midrule"]
for s, e in zip(Rs, Re): rows.append(f"{s[0]} & {e[1]:,} & {100 * s[2]:.2f}\\% & {100 * e[2]:.2f}\\% & {100 * s[3]:.2f}\\% & {100 * e[3]:.2f}\\% & {s[4]:.1f} & {e[4]:.1f} & {s[5]:.4f} & {e[5]:.4f} \\\\".replace(f"{e[1]:,}", f"{e[1]:,}".replace(",", "{,}")))
rows += ["\\bottomrule", "\\end{tabular}"]; (out / "tables" / "per_event_inclusive.tex").write_text("\n".join(rows) + "\n")
M["EnsIncPinside"] = f"{100 * Re[0][2]:.2f}\\%"; M["EnsIncPoutside"] = f"{100 * Re[1][2]:.1f}\\%"; M["EnsIncZinside"] = f"{Re[0][4]:.1f}"; M["EnsIncZoutside"] = f"{Re[1][4]:.0f}"; M["EnsIncPwide"] = f"{100 * Re[4][2]:.1f}\\%"; M["EnsIncPostPinside"] = f"{100 * Rs[0][2]:.2f}\\%"; M["EnsIncPostPoutside"] = f"{100 * Rs[1][2]:.1f}\\%"
# ---- epistemic flag AUCs of the inclusive comparisons: single model vs ensemble
KN = {"epi_exist": "reconstruction flag", "epi_card": "multiplicity", "epi_flow": "event flow", "epi_vz": "vertex $z$ spread"}
A1, A2 = json.load(open("reports/inclusive_gibuu/summary.json"))["epistemic"], json.load(open("reports/inclusive_gibuu_ens/summary.json"))["epistemic"]
rows = ["\\begin{tabular}{lcccc}", "\\toprule", "epistemic quantity & \\multicolumn{2}{c}{outside vs inside (GENIE)} & \\multicolumn{2}{c}{GiBUU vs GENIE, outside} \\\\", " & posterior & ensemble & posterior & ensemble \\\\", "\\midrule"]
for k, lab in KN.items(): rows.append(f"{lab} & {A1[k]['auc_out_vs_in']:.2f} & {A2[k]['auc_out_vs_in']:.2f} & {A1[k]['auc_nuwro_vs_genie_out']:.2f} & {A2[k]['auc_nuwro_vs_genie_out']:.2f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]; (out / "tables" / "epi_auc.tex").write_text("\n".join(rows) + "\n")
M["EnsAUCflowPost"] = f"{A1['epi_flow']['auc_out_vs_in']:.2f}"; M["EnsAUCflowEns"] = f"{A2['epi_flow']['auc_out_vs_in']:.2f}"; M["EnsAUCvzPost"] = f"{A1['epi_vz']['auc_out_vs_in']:.2f}"; M["EnsAUCvzEns"] = f"{A2['epi_vz']['auc_out_vs_in']:.2f}"
(out / "macros.tex").write_text("".join(f"\\newcommand{{\\n{k}}}{{{v}}}\n" for k, v in M.items())); print("written", out, len(M), "macros")
