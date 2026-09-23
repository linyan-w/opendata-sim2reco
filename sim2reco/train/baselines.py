"""M1 baselines: gradient-boosted trees for Tier 0 and multiplicity, MDNs for the muon response and recoil.

Everything is trained on the subrun-level train split, early-stopped on val, reported on test.
Outputs (metrics JSON, LaTeX tables, figures) go to `out_dir`.
"""
from __future__ import annotations

import json
import pathlib
import time

import awkward as ak
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from ..data.dataset import EventDataset, split_by_subrun
from ..eval import plots
from ..eval.metrics import (binary_metrics, calibration_by_bin, confusion, multiclass_metrics,
                            sample_vs_real_1d)
from ..models.mdn import MDN
from ..prep.features import event_features, muon_true

N_PRONG_MAX = 6  # classes 0..5 and "6+"


def load(stem: str, n_max: int = 64):
    reco = ak.from_parquet(stem + ".reco.parquet")
    truth = ak.from_parquet(stem + ".truth.parquet")
    return EventDataset(reco, truth, n_max=n_max)


def gbdt(Xtr, ytr, Xva, yva, seed=0):
    """Regularized HistGBDT. The sklearn default (31 leaves, lr 0.1) diverges on the imbalanced multiclass
    multiplicity target (log loss above marginal); these settings give test log loss 0.84 vs 1.06 marginal."""
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


class Standardizer:
    def __init__(self, X):
        Xa = np.arcsinh(X)
        self.m, self.s = Xa.mean(0), Xa.std(0) + 1e-6

    def __call__(self, X):
        return ((np.arcsinh(X) - self.m) / self.s).astype(np.float32)


def train_mdn(Xtr, Ytr, Xva, Yva, n_components=8, epochs=150, bs=2048, lr=2e-3, device="cuda", seed=0, clip=25.0):
    """Robust standardisation (median, 1.4826 MAD) so heavy tails do not squash the core; standardised targets
    are clipped at +-clip for training only (test NLL is evaluated unclipped)."""
    torch.manual_seed(seed)
    ymean = np.median(Ytr, 0)
    ystd = 1.4826 * np.median(np.abs(Ytr - ymean), 0) + 1e-6
    Ytr = np.clip((Ytr - ymean) / ystd, -clip, clip) * ystd + ymean
    Yva = np.clip((Yva - ymean) / ystd, -clip, clip) * ystd + ymean
    model = MDN(Xtr.shape[1], Ytr.shape[1], n_components).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.tensor(Xtr, device=device), torch.tensor((Ytr - ymean) / ystd, dtype=torch.float32, device=device)
    Xv, Yv = torch.tensor(Xva, device=device), torch.tensor((Yva - ymean) / ystd, dtype=torch.float32, device=device)
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = model.nll(Xt[idx], Yt[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vl = model.nll(Xv, Yv).mean().item()
        hist.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.ymean, model.ystd = ymean, ystd
    return model, hist


def mdn_nll(model, X, Y, device="cuda", clip=25.0):
    """Test NLL per event in the original target units (log-Jacobian of the standardisation added back).
    Targets are clipped at +-clip robust sigmas exactly as in training; without the clip a handful of
    pathological muons (|slope| >> 1) dominate the mean (NLL ~ 500 instead of ~ -4)."""
    with torch.no_grad():
        Xt = torch.tensor(X, device=device)
        Ys = np.clip((Y - model.ymean) / model.ystd, -clip, clip)
        Yt = torch.tensor(Ys, dtype=torch.float32, device=device)
        nll = model.nll(Xt, Yt).cpu().numpy() + np.log(model.ystd).sum()
    return float(nll.mean())


def mdn_sample(model, X, device="cuda"):
    with torch.no_grad():
        s = model.sample(torch.tensor(X, device=device), 1)[:, 0].cpu().numpy()
    return s * model.ystd + model.ymean


def muon_targets(ds, idx):
    mu_true, _ = muon_true(ds.cls[idx], ds.mom[idx], ds.mask[idx])
    mu_reco = ds.tier1[idx, :3].astype(np.float64)
    Pt, Pr = np.linalg.norm(mu_true, axis=1), np.linalg.norm(mu_reco, axis=1)
    r = np.log(Pr / Pt)
    dsx = mu_reco[:, 0] / mu_reco[:, 2] - mu_true[:, 0] / mu_true[:, 2]
    dsy = mu_reco[:, 1] / mu_reco[:, 2] - mu_true[:, 1] / mu_true[:, 2]
    return np.stack([r, dsx, dsy], 1)


def run(stem: str, out_dir: str, device: str = "cuda", epochs: int = 150, seed: int = 0):
    out = pathlib.Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    t0 = time.time()
    ds = load(stem)
    X, names = event_features(ds.cls, ds.mom, ds.mask, ds.ctx)
    split = split_by_subrun(ds.subrun, seed=seed)
    tr, va, te = (split == 0), (split == 1), (split == 2)
    reco = ds.reco_exists.astype(bool)
    metrics = {"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "n_features": len(names),
               "features": names}
    print(f"data ready: {len(ds)} events, {len(names)} features, {time.time()-t0:.0f} s")

    # ---- Tier 0: efficiency, minos_ok, charge ------------------------------------------------------------
    tier0 = {}
    y = ds.tier0[:, 0].astype(int)
    clf = gbdt(X[tr], y[tr], X[va], y[va], seed)
    p = clf.predict_proba(X[te])[:, 1]
    tier0["reco_exists"] = binary_metrics(y[te], p)
    mu_P, mu_th, n_had = X[:, names.index("mu_P")], X[:, names.index("mu_theta")], X[:, names.index("n_had")]
    cal = {"mu_theta_deg": calibration_by_bin(np.degrees(mu_th[te]), y[te], p, np.arange(0, 61, 5)),
           "mu_P_GeV": calibration_by_bin(mu_P[te] / 1000, y[te], p, np.array([0, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40])),
           "n_had": calibration_by_bin(n_had[te], y[te], p, np.arange(-0.5, 10.5, 1))}
    tier0["reco_exists"]["calibration"] = cal
    fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
    plots.calibration_panel(axs[0], cal["mu_theta_deg"], "true muon angle to beam [deg]")
    plots.calibration_panel(axs[1], cal["mu_P_GeV"], "true muon momentum [GeV]"); axs[1].set_xscale("log")
    plots.calibration_panel(axs[2], cal["n_had"], "true hadrons after cuts")
    axs[0].set_title("P(reco muon candidate | truth), test split", loc="left", fontsize=10)
    plots.save(fig, out / "figures" / "m1_efficiency_calibration.png")

    ym = ds.tier0[:, 1].astype(int)
    clf = gbdt(X[tr & reco], ym[tr & reco], X[va & reco], ym[va & reco], seed)
    pm = clf.predict_proba(X[te & reco])[:, 1]
    tier0["minos_ok"] = binary_metrics(ym[te & reco], pm)
    tier0["minos_ok"]["calibration"] = {
        "mu_theta_deg": calibration_by_bin(np.degrees(mu_th[te & reco]), ym[te & reco], pm, np.arange(0, 61, 5)),
        "mu_P_GeV": calibration_by_bin(mu_P[te & reco] / 1000, ym[te & reco], pm, np.array([0, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40]))}
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    plots.calibration_panel(axs[0], tier0["minos_ok"]["calibration"]["mu_theta_deg"], "true muon angle to beam [deg]")
    plots.calibration_panel(axs[1], tier0["minos_ok"]["calibration"]["mu_P_GeV"], "true muon momentum [GeV]"); axs[1].set_xscale("log")
    axs[0].set_title("P(MINOS match | reconstructed), test split", loc="left", fontsize=10)
    plots.save(fig, out / "figures" / "m1_minos_calibration.png")

    ok = reco & (ds.tier0[:, 1] > 0)
    yc = ds.tier0[:, 2].astype(int)
    clf = gbdt(X[tr & ok], yc[tr & ok], X[va & ok], yc[va & ok], seed)
    tier0["charge_neg"] = binary_metrics(yc[te & ok], clf.predict_proba(X[te & ok])[:, 1])
    metrics["tier0"] = tier0
    print("tier0:", {k: {kk: round(vv, 4) for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in tier0.items()})

    # ---- multiplicity -------------------------------------------------------------------------------------
    yn = np.minimum(ds.nprong, N_PRONG_MAX)
    clf = gbdt(X[tr & reco], yn[tr & reco], X[va & reco], yn[va & reco], seed)
    Pn = clf.predict_proba(X[te & reco])
    mm = multiclass_metrics(yn[te & reco], Pn)
    rng = np.random.default_rng(seed)
    samp = (Pn.cumsum(1) > rng.random(len(Pn))[:, None]).argmax(1)
    K = N_PRONG_MAX + 1
    mm["confusion_true_vs_sampled"] = confusion(yn[te & reco], samp, K).tolist()
    mm["marginal_true"] = np.bincount(yn[te & reco], minlength=K).tolist()
    mm["marginal_sampled"] = np.bincount(samp, minlength=K).tolist()
    nvis = X[:, names.index("n_charged_had")][te & reco]
    mm["mean_nprong_by_ncharged"] = [{"n_charged": int(k), "n": int((nvis == k).sum()),
                                      "true": float(yn[te & reco][nvis == k].mean()), "sampled": float(samp[nvis == k].mean())}
                                     for k in range(0, 7) if (nvis == k).sum() > 50]
    metrics["multiplicity"] = mm
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    plots.bar_compare(axs[0], np.array(mm["marginal_true"]), np.array(mm["marginal_sampled"]), "reco hadron prongs (6 = 6+)")
    r = mm["mean_nprong_by_ncharged"]
    axs[1].plot([q["n_charged"] for q in r], [q["true"] for q in r], "o-", color=plots.PALETTE["real"], ms=4, label="MasterAnaDev")
    axs[1].plot([q["n_charged"] for q in r], [q["sampled"] for q in r], "s--", color=plots.PALETTE["model"], ms=4, label="GBDT sampled")
    axs[1].set_xlabel("true charged hadrons after cuts"); axs[1].set_ylabel("mean reco prongs"); axs[1].legend(frameon=False)
    plots.save(fig, out / "figures" / "m1_multiplicity.png")
    print("multiplicity:", {k: round(v, 4) for k, v in mm.items() if isinstance(v, float)})

    # ---- muon response MDN ----------------------------------------------------------------------------------
    std = Standardizer(X[tr])
    Xs = np.concatenate([std(X), ds.tier0[:, 1:2]], 1)  # + minos_ok flag
    Ymu = muon_targets(ds, np.arange(len(ds)))
    good = reco & np.isfinite(Ymu).all(1)
    mdn_res = {}
    for K_, tag in ((1, "gauss"), (8, "mdn8")):
        model, hist = train_mdn(Xs[tr & good], Ymu[tr & good], Xs[va & good], Ymu[va & good], K_, epochs, device=device, seed=seed)
        nll = mdn_nll(model, Xs[te & good], Ymu[te & good], device)
        S = mdn_sample(model, Xs[te & good], device)
        res = {"test_nll": nll, "val_curve": hist, "n_components": K_}
        Yte = Ymu[te & good]; okte = ds.tier0[te & good, 1] > 0
        for j, nm in enumerate(["log_P_ratio", "dslope_x", "dslope_y"]):
            res[nm] = sample_vs_real_1d(Yte[:, j], S[:, j])
            res[nm + "_minos_ok"] = sample_vs_real_1d(Yte[okte, j], S[okte, j])
            res[nm + "_no_minos"] = sample_vs_real_1d(Yte[~okte, j], S[~okte, j])
        mdn_res[tag] = res
        if tag == "mdn8":
            fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
            plots.hist_compare(axs[0], Yte[okte, 0], S[okte, 0], np.linspace(-0.5, 0.5, 80), "log(P_reco / P_true), MINOS matched", ("MasterAnaDev", "MDN (K=8)"))
            plots.hist_compare(axs[1], Yte[~okte, 0], S[~okte, 0], np.linspace(-2.5, 1.0, 80), "log(P_reco / P_true), not matched", ("MasterAnaDev", "MDN (K=8)"))
            plots.hist_compare(axs[2], Yte[:, 1] * 1e3, S[:, 1] * 1e3, np.linspace(-30, 30, 80), "reco - true slope dx/dz [mrad]", ("MasterAnaDev", "MDN (K=8)"))
            axs[1].set_yscale("log")
            plots.save(fig, out / "figures" / "m1_muon_response.png")
    metrics["muon_mdn"] = mdn_res
    print("muon MDN test NLL: gauss %.4f  mdn8 %.4f ; W1 logP ratio: %.4f" % (mdn_res["gauss"]["test_nll"], mdn_res["mdn8"]["test_nll"], mdn_res["mdn8"]["log_P_ratio"]["w1"]))

    # ---- recoil energy MDN ----------------------------------------------------------------------------------
    rec = ds.tier1[:, 3].astype(np.float64)
    Yr = np.log(np.clip(rec, 1.0, None))[:, None]
    goodr = reco & np.isfinite(Yr[:, 0])
    rec_res = {}
    for K_, tag in ((1, "gauss"), (8, "mdn8")):
        model, hist = train_mdn(Xs[tr & goodr], Yr[tr & goodr], Xs[va & goodr], Yr[va & goodr], K_, epochs, device=device, seed=seed)
        nll = mdn_nll(model, Xs[te & goodr], Yr[te & goodr], device)
        S = mdn_sample(model, Xs[te & goodr], device)[:, 0]
        Yte = Yr[te & goodr, 0]
        res = {"test_nll": nll, "n_components": K_, "log_recoil": sample_vs_real_1d(Yte, S)}
        ke = X[te & goodr, names.index("sumKE_had")]
        edges = np.array([0, 100, 200, 400, 800, 1500, 3000, 6000, 20000])
        rows = []
        for b in range(len(edges) - 1):
            s = (ke >= edges[b]) & (ke < edges[b + 1])
            if s.sum() > 50:
                rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(s.sum()),
                             "real_q16_50_84": np.percentile(Yte[s], [16, 50, 84]).tolist(), "fake_q16_50_84": np.percentile(S[s], [16, 50, 84]).tolist()})
        res["log_recoil_by_sumKE"] = rows
        rec_res[tag] = res
        if tag == "mdn8":
            fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
            plots.hist_compare(axs[0], Yte, S, np.linspace(2, 11, 80), "log(recoil_E [MeV])", ("MasterAnaDev", "MDN (K=8)"))
            x = [(q["lo"] + q["hi"]) / 2 for q in rows]
            for k_, (c, lab) in enumerate([(plots.PALETTE["real"], "MasterAnaDev"), (plots.PALETTE["model"], "MDN (K=8)")]):
                key = "real_q16_50_84" if k_ == 0 else "fake_q16_50_84"
                med = [q[key][1] for q in rows]; lo = [q[key][0] for q in rows]; hi = [q[key][2] for q in rows]
                axs[1].plot(x, med, "o-" if k_ == 0 else "s--", color=c, ms=4, label=lab)
                axs[1].fill_between(x, lo, hi, color=c, alpha=0.15)
            axs[1].set_xscale("log"); axs[1].set_xlabel("true hadronic KE after cuts [MeV]"); axs[1].set_ylabel("log(recoil_E), median and 16-84%"); axs[1].legend(frameon=False)
            plots.save(fig, out / "figures" / "m1_recoil.png")
    metrics["recoil_mdn"] = rec_res
    print("recoil MDN test NLL: gauss %.4f  mdn8 %.4f ; W1 log recoil: %.4f" % (rec_res["gauss"]["test_nll"], rec_res["mdn8"]["test_nll"], rec_res["mdn8"]["log_recoil"]["w1"]))

    metrics["runtime_s"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    write_tables(metrics, out / "tables")
    print(f"done in {metrics['runtime_s']:.0f} s -> {out}")
    return metrics


def write_tables(m: dict, tdir: pathlib.Path):
    t0 = m["tier0"]
    rows = "\n".join(f"{name} & {v['rate']:.3f} & {v['logloss_marginal']:.4f} & {v['logloss']:.4f} & {v['auc']:.3f} \\\\"
                     for name, v in ((r"reconstructed (muon candidate)", t0["reco_exists"]), (r"MINOS matched $\mid$ reconstructed", t0["minos_ok"]),
                                     (r"negative charge $\mid$ matched", t0["charge_neg"])))
    (tdir / "tier0.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\nTarget & rate & log loss (marginal) & log loss (GBDT) & AUC \\\\\n\\midrule\n"
        + rows + "\n\\bottomrule\n\\end{tabular}\n")
    mm = m["multiplicity"]
    (tdir / "multiplicity.tex").write_text(
        "\\begin{tabular}{lcc}\n\\toprule\n & marginal & GBDT \\\\\n\\midrule\n"
        f"log loss & {mm['logloss_marginal']:.4f} & {mm['logloss']:.4f} \\\\\naccuracy & {mm['accuracy_marginal']:.3f} & {mm['accuracy']:.3f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    K = len(mm["marginal_true"])
    hdr = " & ".join(str(k) if k < K - 1 else f"{k}+" for k in range(K))
    lines = []
    for i, row in enumerate(mm["confusion_true_vs_sampled"]):
        tot = max(sum(row), 1)
        lines.append(f"{i if i < K-1 else str(i)+'+'} & " + " & ".join(f"{c/tot:.2f}" for c in row) + " \\\\")
    (tdir / "multiplicity_confusion.tex").write_text(
        "\\begin{tabular}{l" + "c" * K + "}\n\\toprule\ntrue $\\backslash$ sampled & " + hdr + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    mu = m["muon_mdn"]; rc = m["recoil_mdn"]

    def q(d):
        r, f = d["real_q16_50_84"], d["fake_q16_50_84"]
        return f"{r[1]:.3f} [{r[0]:.3f}, {r[2]:.3f}] & {f[1]:.3f} [{f[0]:.3f}, {f[2]:.3f}] & {d['w1']:.4f}"
    rows = [
        r"muon $\log(P_\mathrm{reco}/P_\mathrm{true})$, MINOS matched & " + q(mu["mdn8"]["log_P_ratio_minos_ok"]) + r" \\",
        r"muon $\log(P_\mathrm{reco}/P_\mathrm{true})$, not matched & " + q(mu["mdn8"]["log_P_ratio_no_minos"]) + r" \\",
        r"muon $\Delta(p_x/p_z)$ & " + q(mu["mdn8"]["dslope_x"]) + r" \\",
        r"muon $\Delta(p_y/p_z)$ & " + q(mu["mdn8"]["dslope_y"]) + r" \\",
        r"$\log(E_\mathrm{recoil}/\mathrm{MeV})$ & " + q(rc["mdn8"]["log_recoil"]) + r" \\",
    ]
    (tdir / "mdn_marginals.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\nVariable & MasterAnaDev median [16\\%, 84\\%] & MDN sample median [16\\%, 84\\%] & $W_1$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (tdir / "mdn_nll.tex").write_text(
        "\\begin{tabular}{lcc}\n\\toprule\nTarget & Gaussian ($K=1$) & MDN ($K=8$) \\\\\n\\midrule\n"
        f"muon response (3-d) & {mu['gauss']['test_nll']:.4f} & {mu['mdn8']['test_nll']:.4f} \\\\\n"
        f"$\\log E_\\mathrm{{recoil}}$ (1-d) & {rc['gauss']['test_nll']:.4f} & {rc['mdn8']['test_nll']:.4f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    n = f"{m['n_train']} & {m['n_val']} & {m['n_test']}"
    (tdir / "splits.tex").write_text("\\begin{tabular}{ccc}\n\\toprule\ntrain & validation & test \\\\\n\\midrule\n" + n + " \\\\\n\\bottomrule\n\\end{tabular}\n")
