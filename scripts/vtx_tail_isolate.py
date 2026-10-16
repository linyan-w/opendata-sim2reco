#!/usr/bin/env python
"""Where do the surrogate's vertex-z tails get lost: in the event flow's continuous z, or at decode, where the
vertex class (sampled independently of the flow's z given the conditions) overrides it?  On test events: the
reco-minus-true z tail fractions of the open dataset, of the flow's continuous z before decoding, and of the
decoded z; the vertex-class marginals; and, among the open dataset's tail events, the share that is snapped far
(class 8) against unsnapped (class 0).
Usage: scripts/vtx_tail_isolate.py [--model reports/m3_1A_z] [--n 300000]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sim2reco.data.compact import load_compact, CompactDataset, collate, selection_from_config
from sim2reco.data.dataset import split_by_subrun
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="reports/m3_1A_z"); ap.add_argument("--n", type=int, default=300000); ap.add_argument("--out", default="reports/vtx_tail/isolate.json"); a = ap.parse_args()
cfg = torch.load(f"{a.model}/model.pt", map_location="cpu", weights_only=False)["config"]
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet")), **selection_from_config(cfg)); split = split_by_subrun(d["subrun"], seed=0)
idx = np.random.default_rng(0).permutation(np.where(split == 2)[0])[:a.n]
model, tf, ptf = load_model(f"{a.model}/model.pt"); ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); torch.manual_seed(0)
X1, EX, VC = [], [], []
with torch.no_grad():
    for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
        s = model.sample(to_dev(b, "cuda"), 64, prongs=False); X1.append(s["x1"].cpu()); EX.append(s["exist"].cpu()); VC.append(s["vclass"].cpu())
x1 = torch.cat(X1).numpy(); ex = torch.cat(EX).numpy(); vc = torch.cat(VC).numpy(); zt = d["ctx"][idx, 2]
y = tf.inverse(x1, d["mu_true"][idx], d["ctx"][idx]); z_flow = y[:, 5]
z_dec = np.where(vc > 0, tf.planes.z_for_class(vc, zt, z_flow), z_flow)
r = d["reco_exists"][idx] & ds.valid; z_real = d["tier1"][idx, 5]; vc_real = tf.planes.classify(np.where(r, z_real, zt), zt)
def tails(dz): return {"gt100": float((np.abs(dz) > 100).mean()), "gt300": float((np.abs(dz) > 300).mean()), "gt1000": float((np.abs(dz) > 1000).mean()), "pos100": float((dz > 100).mean()), "neg100": float((dz < -100).mean())}
R = {"n_real": int(r.sum()), "n_sur": int(ex.sum()), "real": tails((z_real - zt)[r]), "flow_continuous": tails((z_flow - zt)[ex]), "decoded": tails((z_dec - zt)[ex]),
     "class_marginal_real": np.bincount(vc_real[r], minlength=9).tolist(), "class_marginal_sur": np.bincount(vc[ex], minlength=9).tolist()}
far_real = np.abs(z_real - zt) > 100; R["real_tail_by_class"] = {"frac_class8": float((vc_real[r & far_real] == 8).mean()), "frac_class0": float((vc_real[r & far_real] == 0).mean()), "frac_near": float(((vc_real[r & far_real] > 0) & (vc_real[r & far_real] < 8)).mean())}
far_flow = np.abs(z_flow - zt) > 100; R["sur_flow_tail_by_class"] = {"frac_class8": float((vc[ex & far_flow] == 8).mean()), "frac_class0": float((vc[ex & far_flow] == 0).mean()), "frac_near": float(((vc[ex & far_flow] > 0) & (vc[ex & far_flow] < 8)).mean())}
R["sur_class8_with_flow_far"] = float(far_flow[ex & (vc == 8)].mean()); R["real_class8_far"] = float(far_real[r & (vc_real == 8)].mean())
R["real_class0_far"] = float(far_real[r & (vc_real == 0)].mean()); R["sur_class0_flow_far"] = float(far_flow[ex & (vc == 0)].mean())
for k, v in R.items(): print(k, v)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(R, open(a.out, "w"), indent=1)
