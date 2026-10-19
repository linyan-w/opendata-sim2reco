#!/usr/bin/env python
"""Macros of the all-events model's seven-variable closure (scripts/closure_subset_all.py -> reports/all_events/closure.json)
for the paper's closure table: \nAEClo<Subset><Var><Marg|Cond>, Subset in {CC (CC nu_mu test events), All (all test events)},
Var in {CI (seven variables), Mu, Vtx, Recoil}. Writes reports/all_events/closure_macros.tex."""
import json, pathlib
c = json.load(open("reports/all_events/closure.json")); out = []
for sub, key in (("CC", "A_ccnumu"), ("All", "A_full")):
    r = c[key]
    for var, k in (("CI", "ccinc"), ("Mu", "mu"), ("Vtx", "vtx"), ("Recoil", "recoil")):
        for kind in ("marg", "cond"): out.append(f"\\newcommand{{\\nAEClo{sub}{var}{kind.capitalize()}}}{{{r[f'{k}_{kind}']:.3f}}}")
    out.append(f"\\newcommand{{\\nAEClo{sub}N}}{{{r['n']:,}}}".replace(",", "{,}"))
pathlib.Path("reports/all_events/closure_macros.tex").write_text("% all-events model, seven-variable closure by subset (scripts/paper_closure_macros.py)\n" + "\n".join(out) + "\n"); print("\n".join(out))
