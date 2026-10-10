#!/usr/bin/env python
"""Share of all-events (training volume) events by the reference the reconstructed muon is expressed against:
leading mu-, other charged lepton, leading charged hadron above 10 MeV, or the 1 GeV beam direction.
Writes reports/all_events/reference_categories.json. Usage: PYTHONPATH=. scripts/reference_categories.py"""
import glob, numpy as np, awkward as ak, json
from sim2reco.prep.particles import in_population, CHARGED_LEPTONS, CHARGED_HADRONS, _isin
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet"))
cnt = np.zeros(4, np.int64); ccn = 0
for s in stems:
    t = ak.from_parquet(s + ".truth.parquet", columns=["mc_vtx", "mc_FSPartPDG", "mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_current", "mc_incoming"])
    t = t[in_population(t, "all")]; pdg = t["mc_FSPartPDG"]; apdg = abs(pdg)
    P = np.sqrt(t["mc_FSPartPx"] ** 2 + t["mc_FSPartPy"] ** 2 + t["mc_FSPartPz"] ** 2)
    mass = ak.where(apdg == 2212, 938.272, ak.where(apdg == 211, 139.570, ak.where(apdg == 321, 493.677, 0.0)))
    ke = t["mc_FSPartE"] - mass
    h0 = ak.to_numpy(ak.any(pdg == 13, axis=1)); h1 = ak.to_numpy(ak.any(_isin(apdg, CHARGED_LEPTONS) & (P > 0), axis=1))
    h2 = ak.to_numpy(ak.any(_isin(apdg, CHARGED_HADRONS) & (ke >= 10) & (P > 0), axis=1))
    c = np.where(h0, 0, np.where(h1, 1, np.where(h2, 2, 3))); cnt += np.bincount(c, minlength=4)
    print(s, cnt / cnt.sum(), flush=True)
json.dump({"mu-": int(cnt[0]), "lepton": int(cnt[1]), "hadron": int(cnt[2]), "beam": int(cnt[3])}, open("reports/all_events/reference_categories.json", "w"))
