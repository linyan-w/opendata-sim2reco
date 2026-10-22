#!/usr/bin/env python
"""Where does the 2p2h holdout differ from the full model?  For the held-out 2p2h test events of the paper figure cache
(scripts/paper_fig_response.py --only-inttype 8 --subset ccnumu, same event order), the isolation of each event in the
surrogate's input space: the density of all other modes relative to 2p2h at the event, from a 2p2h-against-rest classifier
trained on the training split of the all-events population (the holdout model's training set is the 'rest').  Also the
physical final state: nucleon counts above 50 MeV, pion presence, hadronic kinetic energy, the momentum share and opening
angle of the two leading nucleons, the true muon pT and angle.  Writes reports/holdout_isolation/events.npz.
Usage: scripts/holdout_isolation.py [--model reports/m3_1A_all] [--ntrain 1500000]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sim2reco.data.compact import load_compact, selection_from_config, subset_mask
from sim2reco.data.dataset import split_by_subrun
from sim2reco.prep.features import event_features
from sim2reco.train.m2 import _pad, _theta
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="reports/m3_1A_all"); ap.add_argument("--ntrain", type=int, default=1500000); ap.add_argument("--out", default="reports/holdout_isolation/events.npz"); a = ap.parse_args()
cfg = torch.load(f"{a.model}/model.pt", map_location="cpu", weights_only=False)["config"]
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet")), **selection_from_config(cfg)); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]; te = te[d["intType"][te] == 8]; te = te[subset_mask(d, "ccnumu")[te]]; idx = np.random.default_rng(0).permutation(te)[:300000]  # exactly the cache's events
rng = np.random.default_rng(1); tr = rng.permutation(np.where(split == 0)[0])[:a.ntrain]
def feats(ii):
    cls, mom, mask = _pad(d, ii); X, names = event_features(cls, mom, mask, d["ctx"][ii])
    P = np.linalg.norm(mom, axis=-1); Pp = np.where(mask & (cls == 4), P, -1.0); Ps = -np.sort(-Pp, axis=1)
    return np.column_stack([X, Ps[:, 0], Ps[:, 1]]), (cls, mom, mask, P, names)
Xtr, _ = feats(tr); ytr = d["intType"][tr] == 8; prior = ytr.mean(); print(f"train {len(tr)} events, 2p2h share {100*prior:.2f}%", flush=True)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0).fit(Xtr, ytr)
Xte, (cls, mom, mask, P, names) = feats(idx); p = np.clip(clf.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
r = (1 - p) / p * prior / (1 - prior)  # density of all other modes relative to 2p2h at the event
# sanity: AUC on a held-out mixed sample of the test split
chk = rng.permutation(np.where(split == 2)[0])[:300000]; Xc, _ = feats(chk); auc = roc_auc_score(d["intType"][chk] == 8, clf.predict_proba(Xc)[:, 1]); print(f"classifier AUC on mixed test events {auc:.3f}", flush=True)
M = {4: 938.272, 12: 939.565}; KE = np.zeros_like(P)
for c, m in M.items(): s = mask & (cls == c); KE[s] = np.sqrt(P[s] ** 2 + m ** 2) - m
nuc = mask & ((cls == 4) | (cls == 12)); Pn = np.where(nuc & (KE > 50), P, -1.0); o = np.argsort(-Pn, axis=1); i0, i1 = o[:, 0], o[:, 1]; ar = np.arange(len(idx))
p0, p1 = mom[ar, i0], mom[ar, i1]; has2 = (Pn[ar, i1] > 0)
cosang = np.where(has2, (p0 * p1).sum(1) / np.maximum(np.linalg.norm(p0, axis=1) * np.linalg.norm(p1, axis=1), 1e-9), np.nan)
share = np.where(has2, Pn[ar, i0] / np.maximum(Pn[ar, i0] + Pn[ar, i1], 1e-9), np.nan)
mu = d["mu_true"][idx]; pT = np.linalg.norm(mu[:, :2], axis=1) / 1e3
out = dict(r=r.astype(np.float32), p2p2h=p.astype(np.float32), n_p50=(mask & (cls == 4) & (KE > 50)).sum(1), n_n50=(mask & (cls == 12) & (KE > 50)).sum(1), has_pi=(mask & np.isin(cls, [5, 6, 7])).any(1),
           sumKE_had=Xte[:, names.index("sumKE_had")], cos_nn=cosang.astype(np.float32), share_nn=share.astype(np.float32), mu_pT=pT.astype(np.float32), mu_theta=np.degrees(_theta(mu)).astype(np.float32), idx=idx, prior=prior, auc=auc)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out, **out)
print(f"2p2h test events {len(idx)}: isolation r<0.1 {100*(r<0.1).mean():.1f}%, r<0.01 {100*(r<0.01).mean():.2f}%, median {np.median(r):.2f}; written {a.out}")
