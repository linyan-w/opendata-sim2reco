#!/usr/bin/env python
"""Macros of the all-events model's seven-variable closure (scripts/closure_subset_all.py -> reports/all_events/closure.json)
for the paper's closure table: \nAEClo<Subset><Var><Marg|Cond>, Subset in {CC (A, CC nu_mu test events), All (A, all test events), ATwoP, BTwoP (2p2h test events), BCtl (B on its control), BAll},
Var in {CI (seven variables), Mu, Vtx, Recoil}. Writes reports/all_events/closure_macros.tex."""
import json, pathlib
c = json.load(open("reports/all_events/closure.json")); out = []
for sub, key in (("CC", "A_ccnumu"), ("All", "A_full"), ("ATwoP", "A_twop"), ("BTwoP", "B_twop"), ("BCtl", "B_ctl"), ("BAll", "B_full")):
    r = c[key]
    for var, k in (("CI", "ccinc"), ("Mu", "mu"), ("Vtx", "vtx"), ("Recoil", "recoil")):
        for kind in ("marg", "cond"): out.append(f"\\newcommand{{\\nAEClo{sub}{var}{kind.capitalize()}}}{{{r[f'{k}_{kind}']:.3f}}}")
    out.append(f"\\newcommand{{\\nAEClo{sub}N}}{{{r['n']:,}}}".replace(",", "{,}"))
# the technote's closure-subset table and \nCL* macros (as scripts/closure_subset.py wrote them for the CC model) for the CC nu_mu subset evaluation directory
cc = pathlib.Path("reports/m3_1A_all_ccnumu"); (cc / "tables").mkdir(parents=True, exist_ok=True); f3 = lambda x: f"{x:.3f}"
rows = [("model A, CC $\\nu_\\mu$ test events", "A_ccnumu"), ("model A, full test split (all events)", "A_full"), ("model B (2p2h blind), CC $\\nu_\\mu$ non-2p2h control", "B_ctl"), ("model A, held-out 2p2h events", "A_twop"), ("model B, held-out 2p2h events", "B_twop")]
(cc / "tables" / "closure_ccinc.tex").write_text("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{all nine variables} & \\multicolumn{2}{c}{non-calorimetry set (muon, vertex, recoil)} \\\\\n & marginal & + truth & marginal & + truth \\\\\n\\midrule\n"
    + "".join(f"{lab} & {f3(c[k]['all_marg'])} & {f3(c[k]['all_cond'])} & {f3(c[k]['ccinc_marg'])} & {f3(c[k]['ccinc_cond'])} \\\\\n" for lab, k in rows) + "\\bottomrule\n\\end{tabular}\n")
mac = []
for sub, key in (("Full", "A_ccnumu"), ("TwoP", "A_twop")):
    for var, k in (("All", "all"), ("CI", "ccinc"), ("Mu", "mu"), ("Vtx", "vtx"), ("Recoil", "recoil")):
        for kind in ("marg", "cond"): mac.append(f"\\newcommand{{\\nCLA{sub}{var}{kind.capitalize()}}}{{{c[key][f'{k}_{kind}']:.3f}}}")
for sub, key in (("Full", "B_full"), ("TwoP", "B_twop"), ("Ctl", "B_ctl")):
    for var, k in (("All", "all"), ("CI", "ccinc"), ("Mu", "mu"), ("Vtx", "vtx"), ("Recoil", "recoil")):
        for kind in ("marg", "cond"): mac.append(f"\\newcommand{{\\nCLB{sub}{var}{kind.capitalize()}}}{{{c[key][f'{k}_{kind}']:.3f}}}")
(cc / "closure_ccinc_macros.tex").write_text("% all-events models, closure by subset (scripts/paper_closure_macros.py); A 'Full' = CC nu_mu test events\n" + "\n".join(mac) + "\n")
pathlib.Path("reports/all_events/closure_macros.tex").write_text("% all-events model, seven-variable closure by subset (scripts/paper_closure_macros.py)\n" + "\n".join(out) + "\n"); print("\n".join(out))
