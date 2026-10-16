#!/usr/bin/env python
"""Technote tables and macros of the full-dataset, full-volume section: the inclusive NuWro and GiBUU samples
(scripts/nuwro_inclusive.py) against the open dataset, and their reco-fiducial comparison (scripts/fullvolume_compare.py).
Writes reports/inclusive/{macros_inc.tex, tables/composition.tex, tables/origin_both.tex, tables/epistemic_both.tex}.
Usage: scripts/technote_inclusive.py [--nuwro data/inclusive/nuwro_inclusive.json --gibuu data/inclusive_gibuu/gibuu_inclusive.json]"""
import argparse, json, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("--nuwro", default="data/inclusive/nuwro_inclusive.json"); ap.add_argument("--gibuu", default="data/inclusive_gibuu/gibuu_inclusive.json")
ap.add_argument("--nuwro-report", default="reports/inclusive"); ap.add_argument("--gibuu-report", default="reports/inclusive_gibuu"); ap.add_argument("--out", default="reports/inclusive"); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
N, G = json.load(open(a.nuwro)), json.load(open(a.gibuu)); SN, SG = json.load(open(f"{a.nuwro_report}/summary.json")), json.load(open(f"{a.gibuu_report}/summary.json"))
FL = {"numu": r"$\nu_\mu$", "numubar": r"$\bar\nu_\mu$", "nue": r"$\nu_e$", "nuebar": r"$\bar\nu_e$"}
pct = lambda x, d=2: f"{100 * x:.{d}f}\\%"
# composition: flavour x current, open dataset vs NuWro vs GiBUU (weighted shares)
rows = ["\\begin{tabular}{llrrr}", "\\toprule", "Flavour & Current & open dataset & NuWro & GiBUU \\\\", "\\midrule"]
for f in ("numu", "numubar", "nue", "nuebar"):
    for c in ("CC", "NC"):
        k = f"{f}_{c}"; rows.append(f"{FL[f]} & {c} & {pct(N['composition'][k]['open'])} & {pct(N['composition'][k]['nuwro'])} & {pct(G['composition'][k]['nuwro'])} \\\\")
rows += ["\\midrule", f"events & & {N['open_events_all']:,} & {N['n_events']:,} & {G['n_events']:,} \\\\".replace(",", "{,}"), "\\bottomrule", "\\end{tabular}"]
(out / "tables" / "composition.tex").write_text("\n".join(rows) + "\n")
# origin of the reco-fiducial events: reference, surrogate on GENIE, NuWro, GiBUU
def orow(lab, r): return f"{lab} & {r['recofid_per_truefid']:.3f} & {100 * r['recofid_per_truth']:.2f}\\% & {100 * r['out_frac']:.1f}\\,$\\pm$\\,{100 * r['out_frac_err']:.2f}\\% & {100 * r['region']['upstream']:.1f}\\% & {100 * r['region']['downstream']:.1f}\\% & {100 * r['region']['transverse']:.1f}\\% & {100 * r['out_by_nucleus']['Pb']:.0f}\\% & {100 * r['out_by_nucleus']['Fe']:.0f}\\% & {100 * r['out_by_nucleus']['C']:.0f}\\% & {100 * r['vtx_tail_frac']:.1f}\\% \\\\"
rows = ["\\begin{tabular}{lcccccccccc}", "\\toprule", " & reco-FV per true-FV & reco-FV per event & true outside FV & upstream & downstream & outside hexagon & of which Pb & Fe & C & $|\\Delta z| > 100$\\,mm \\\\", "\\midrule",
        orow("open dataset (GENIE events)", SN["origin"]["ref"]), orow("surrogate on GENIE", SN["origin"]["genie"]), orow("surrogate on NuWro", SN["origin"]["nuwro"]), orow("surrogate on GiBUU", SG["origin"]["nuwro"]), "\\bottomrule", "\\end{tabular}"]
(out / "tables" / "origin_both.tex").write_text("\n".join(rows) + "\n")
# epistemic separation: outside vs inside (surrogate on GENIE), and generator vs GENIE among outside events
KN = {"epi_exist": "reconstruction flag", "epi_card": "multiplicity", "epi_flow": "event flow", "epi_vz": "vertex $z$ spread"}
rows = ["\\begin{tabular}{lccc}", "\\toprule", "epistemic quantity & outside vs inside (GENIE) & NuWro vs GENIE, outside & GiBUU vs GENIE, outside \\\\", "\\midrule"]
for k, lab in KN.items(): rows.append(f"{lab} & {SN['epistemic'][k]['auc_out_vs_in']:.2f} & {SN['epistemic'][k]['auc_nuwro_vs_genie_out']:.2f} & {SG['epistemic'][k]['auc_nuwro_vs_genie_out']:.2f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]; (out / "tables" / "epistemic_both.tex").write_text("\n".join(rows) + "\n")
M = {"IncFrac": pct(N["exposure_fraction"], 1), "IncOpenAll": f"{N['open_events_all']:,}".replace(",", "{,}"), "IncNuwN": f"{N['n_events']:,}".replace(",", "{,}"), "IncGibN": f"{G['n_events']:,}".replace(",", "{,}"),
     "IncNuwLimit": f"{FL.get(N['limiting'][0], N['limiting'][0])} {'CC' if N['limiting'][1] == 1 else 'NC'} on Z={N['limiting'][2]}", "IncNuwWeighted": str(sum(1 for r in N["strata"] if r["weight"] > 1.001)), "IncGibWeighted": str(sum(1 for r in G["strata"] if r["weight"] > 1.001)),
     "IncNuwMaxW": f"{max(r['weight'] for r in N['strata']):.0f}", "IncGibMaxW": f"{max(r['weight'] for r in G['strata']):.0f}", "IncNuwNC": pct(sum(v['nuwro'] for k, v in N['composition'].items() if k.endswith('NC')), 1), "IncGibNC": pct(sum(v['nuwro'] for k, v in G['composition'].items() if k.endswith('NC')), 1), "IncOpenNC": pct(sum(v['open'] for k, v in N['composition'].items() if k.endswith('NC')), 1),
     "IncNuwRecoFrac": pct(SN["origin"]["nuwro"]["recofid_per_truth"]), "IncGibRecoFrac": pct(SG["origin"]["nuwro"]["recofid_per_truth"]), "IncGenRecoFrac": pct(SN["origin"]["genie"]["recofid_per_truth"]), "IncRefRecoFrac": pct(SN["origin"]["ref"]["recofid_per_truth"])}
(out / "macros_inc.tex").write_text("".join(f"\\newcommand{{\\n{k}}}{{{v}}}\n" for k, v in M.items())); print("written", out, M)
