"""M3: Tier 2 prong set flow + vertex-plane head on top of the M2 surrogate (warm start)."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.compact import MODEL_NAMES, CompactDataset, Tier1Transform, collate, load_compact
from ..data.dataset import split_by_subrun
from ..data.prongs_tf import ProngTransform, VertexPlaneTable
from ..eval import plots
from ..eval.metrics import multiclass_metrics, sample_vs_real_1d
from ..models.surrogate import Surrogate
from ..prep.features import event_features
from . import m2

PRONG_FIELD_NAMES = ["theta", "phi", "has_kin", "pi_P", "has_p", "p_P", "p_score1", "is_primary", "is_exiting"]


def make_loaders(d, split, tf, ptf, bs, seed, workers=4):
    idx = {k: np.where(split == v)[0] for k, v in (("train", 0), ("val", 1), ("test", 2))}
    ds = {k: CompactDataset(d, v, tf, seed, prong_tf=ptf) for k, v in idx.items()}
    ld = {k: DataLoader(ds[k], batch_size=bs, shuffle=(k == "train"), collate_fn=collate, num_workers=workers,
                        drop_last=(k == "train"), persistent_workers=workers > 0) for k in ds}
    return idx, ds, ld


def fit_transforms(d, split, seed):
    ok_tr = (split == 0) & d["reco_exists"]
    planes = VertexPlaneTable().fit(d["tier1"][ok_tr, 5])
    tf = Tier1Transform(); tf.planes = planes
    tf.fit(d["tier1"][ok_tr], d["mu_true"][ok_tr], d["ctx"][ok_tr], seed)
    # prongs of training reco events
    tr_ev = np.where(ok_tr)[0]
    sel = np.concatenate([np.arange(d["p_offsets"][e], d["p_offsets"][e + 1]) for e in tr_ev[:200000]])
    ptf = ProngTransform().fit(d["prongs"][sel], seed)
    return tf, ptf


def train(stems, out_dir, init_from="reports/m2/model.pt", epochs=12, bs=1024, lr=2e-4, seed=0, device="cuda",
          prong_layers=3, log_every=500):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
    d = load_compact(stems); split = split_by_subrun(d["subrun"], seed=seed)
    tf, ptf = fit_transforms(d, split, seed)
    idx, ds, ld = make_loaders(d, split, tf, ptf, bs, seed)
    print(f"data: {len(split)} events, {d['p_offsets'][-1]} prongs, {len(tf.planes.z)} vertex planes, {time.time()-t0:.0f} s", flush=True)
    ck = torch.load(init_from, map_location=device, weights_only=False); c = ck["config"]
    model = Surrogate(c["d_model"], 4, c["n_layers"], c["flow_hidden"], c["flow_layers"], tier2=True, prong_layers=prong_layers).to(device)
    own = model.state_dict()
    sd = {k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape}  # skip re-shaped heads
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"warm start from {init_from}: {len(missing)} new tensors (vtx head + prong flow), {len(unexpected)} unexpected", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(ld["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    hist, best, step = [], np.inf, 0
    cfg = dict(c, tier2=True, prong_layers=prong_layers)
    for ep in range(epochs):
        model.train(); agg = {}
        for b in ld["train"]:
            b = m2.to_dev(b, device); L = model.losses(b); loss = sum(L.values())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            for k, v in L.items(): agg[k] = agg.get(k, 0) + v.item()
            if step % log_every == 0: print(f"  step {step} " + " ".join(f"{k} {v.item():.4f}" for k, v in L.items()), flush=True)
        model.eval(); va = {}
        with torch.no_grad():
            for b in ld["val"]:
                L = model.losses(m2.to_dev(b, device))
                for k, v in L.items(): va[k] = va.get(k, 0) + v.item() / len(ld["val"])
        tot = sum(va.values()); hist.append({"epoch": ep, "train": {k: v / len(ld["train"]) for k, v in agg.items()}, "val": va, "time": time.time() - t0})
        print(f"epoch {ep}: val " + " ".join(f"{k} {v:.4f}" for k, v in va.items()) + f"  total {tot:.4f}  [{time.time()-t0:.0f} s]", flush=True)
        if np.isfinite(tot) and tot < best:
            best = tot
            torch.save({"model": model.state_dict(), "transform": tf.state(), "prong_transform": ptf.state(), "config": cfg}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    return d, split, tf, ptf, idx, ld


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False); c = ck["config"]
    m = Surrogate(c["d_model"], 4, c["n_layers"], c["flow_hidden"], c["flow_layers"], tier2=True, prong_layers=c.get("prong_layers", 3)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, Tier1Transform.from_state(ck["transform"]), ProngTransform.from_state(ck["prong_transform"])


def _event_prong_summary(theta, kin, piP, pf, pP, sc, ex, offsets):
    """Per-event summaries from flat prong arrays: n, n_kin, n_pfit, n_exit, lead pi_P, lead theta, sum pi_P, max score."""
    n_ev = len(offsets) - 1; out = np.zeros((n_ev, 8), np.float64)
    seg = np.repeat(np.arange(n_ev), np.diff(offsets))
    out[:, 0] = np.diff(offsets)
    np.add.at(out[:, 1], seg, kin); np.add.at(out[:, 2], seg, pf); np.add.at(out[:, 3], seg, ex)
    lead = np.full(n_ev, -1.0); np.maximum.at(lead, seg, np.where(kin, piP, -1.0)); out[:, 4] = lead
    # theta of leading prong
    order = np.lexsort((-np.where(kin, piP, -1.0), seg)); first = np.r_[True, seg[order][1:] != seg[order][:-1]]
    lt = np.full(n_ev, -9.0); lt[seg[order][first]] = np.where(kin, theta, -9.0)[order][first]; out[:, 5] = lt
    np.add.at(out[:, 6], seg, np.where(kin, piP, 0.0)); ms = np.full(n_ev, -1.0); np.maximum.at(ms, seg, np.where(pf, sc, -1.0)); out[:, 7] = ms
    return out


@torch.no_grad()
def evaluate_tier2(model, tf, ptf, d, idx_test, ld_test, out_dir, device="cuda", n_steps=64, seed=0):
    out = pathlib.Path(out_dir); (out / "figures").mkdir(exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    S_pr, S_pm, S_v, S_pv, S_x, S_n, S_ex = [], [], [], [], [], [], []
    for b in ld_test:
        b = m2.to_dev(b, device); s = model.sample(b, n_steps)
        S_pr.append(s["prongs"].cpu()); S_pm.append(s["pmask"].cpu()); S_v.append(s["vclass"].cpu()); S_pv.append(s["pv"].cpu())
        S_x.append(s["x1"].cpu()); S_n.append(s["nprong"].cpu()); S_ex.append(s["exist"].cpu())
    Npad = max(p.shape[1] for p in S_pr)
    pad = lambda p: torch.nn.functional.pad(p, (0, 0, 0, Npad - p.shape[1])) if p.dim() == 3 else torch.nn.functional.pad(p, (0, Npad - p.shape[1]))
    PR = torch.cat([pad(p) for p in S_pr]).numpy(); PM = torch.cat([pad(p) for p in S_pm]).numpy()
    VC = torch.cat(S_v).numpy(); PV = torch.cat(S_pv).numpy(); X1 = torch.cat(S_x).numpy(); NP = torch.cat(S_n).numpy()
    t0 = d["tier0"][idx_test]; reco = t0[:, 0] > 0; ir = idx_test[reco]
    M = {}
    # ---- vertex class ----
    vc_real = tf.planes.classify(d["tier1"][ir, 5], d["ctx"][ir, 2])
    M["vertex_class"] = multiclass_metrics(vc_real, PV[reco]); M["vertex_class"]["snapped_frac_real"] = float((vc_real > 0).mean()); M["vertex_class"]["snapped_frac_sampled"] = float((VC[reco] > 0).mean())
    M["vertex_class"]["marginal_real"] = np.bincount(vc_real, minlength=9).tolist(); M["vertex_class"]["marginal_sampled"] = np.bincount(VC[reco], minlength=9).tolist()
    # decoded reco z: snapped -> plane, else flow residual
    y_fake = tf.inverse(X1[reco], d["mu_true"][ir], d["ctx"][ir])
    zsnap = tf.planes.z_for_class(VC[reco], d["ctx"][ir, 2], y_fake[:, 5]); z_fake = np.where(VC[reco] > 0, zsnap, y_fake[:, 5]); z_real = d["tier1"][ir, 5]
    off_real = z_real - tf.planes.z[tf.planes.nearest(z_real)]; off_fake = z_fake - tf.planes.z[tf.planes.nearest(z_fake)]
    M["vertex_offset_to_nearest_plane"] = {"real_frac_within_0.1mm": float((np.abs(off_real) < 0.1).mean()), "fake_frac_within_0.1mm": float((np.abs(off_fake) < 0.1).mean()),
                                           "w1_mm": sample_vs_real_1d(off_real, off_fake, -15, 15)["w1"]}
    # ---- prongs: flat real vs flat fake over reconstructed test events ----
    real_sel = np.concatenate([np.arange(d["p_offsets"][e], d["p_offsets"][e + 1]) for e in ir])
    Preal = d["prongs"][real_sel]; off_real_p = np.concatenate([[0], np.cumsum(np.diff(d["p_offsets"])[ir])])
    pm = PM[reco]; Pfake = ptf.inverse(PR[reco][pm]); off_fake_p = np.concatenate([[0], np.cumsum(pm.sum(1))])
    kin_r, kin_f = Preal[:, 2] > 0.5, Pfake[:, 2] > 0.5; pf_r, pf_f = Preal[:, 4] > 0.5, Pfake[:, 4] > 0.5
    M["prong_counts"] = {"n_real": int(len(Preal)), "n_fake": int(len(Pfake)), "has_kin": [float(kin_r.mean()), float(kin_f.mean())],
                         "has_p": [float(pf_r.mean()), float(pf_f.mean())], "is_exiting": [float((Preal[:, 8] > 0.5).mean()), float((Pfake[:, 8] > 0.5).mean())]}
    M["prong_marginals"] = {"theta": sample_vs_real_1d(Preal[kin_r, 0], Pfake[kin_f, 0]), "log_pi_P": sample_vs_real_1d(np.log(Preal[kin_r & (Preal[:, 3] > 0), 3]), np.log(Pfake[kin_f, 3])),
                            "log_p_P": sample_vs_real_1d(np.log(Preal[pf_r, 5]), np.log(Pfake[pf_f, 5])), "p_score1": sample_vs_real_1d(Preal[pf_r, 6], Pfake[pf_f, 6], 0, 1)}
    # event-level summaries + classifier tests
    Er = _event_prong_summary(Preal[:, 0], kin_r, Preal[:, 3], pf_r, Preal[:, 5], Preal[:, 6], Preal[:, 8] > 0.5, off_real_p)
    Ef = _event_prong_summary(Pfake[:, 0], kin_f, Pfake[:, 3], pf_f, Pfake[:, 5], Pfake[:, 6], Pfake[:, 8] > 0.5, off_fake_p)
    cls_p, mom_p, mask_p = m2._pad(d, ir); X, names = event_features(cls_p, mom_p, mask_p, d["ctx"][ir])
    yc = np.r_[np.ones(len(Er)), np.zeros(len(Ef))]; perm = rng.permutation(len(yc)); half = len(perm) // 2
    def _auc(A):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=True, random_state=seed).fit(A[perm[:half]], yc[perm[:half]])
        return float(roc_auc_score(yc[perm[half:]], c.predict_proba(A[perm[half:]])[:, 1]))
    M["closure_prongs"] = {"event_summary_only": _auc(np.r_[Er, Ef]), "truth_plus_event_summary": _auc(np.r_[np.c_[X, Er], np.c_[X, Ef]]),
                           "truth_plus_vertex_z": _auc(np.r_[np.c_[X, z_real - d["ctx"][ir, 2]], np.c_[X, z_fake - d["ctx"][ir, 2]]])}
    # conditional: leading proton-hypothesis P vs true leading proton KE
    ke = X[:, names.index("maxKE_p")]
    lead_pP_r = np.full(len(ir), np.nan); lead_pP_f = np.full(len(ir), np.nan)
    seg_r = np.repeat(np.arange(len(ir)), np.diff(off_real_p)); seg_f = np.repeat(np.arange(len(ir)), np.diff(off_fake_p))
    lr_, lf_ = np.full(len(ir), -1.0), np.full(len(ir), -1.0)
    np.maximum.at(lr_, seg_r, np.where(pf_r, Preal[:, 5], -1.0)); np.maximum.at(lf_, seg_f, np.where(pf_f, Pfake[:, 5], -1.0))
    lead_pP_r = np.where(lr_ > 0, lr_, np.nan); lead_pP_f = np.where(lf_ > 0, lf_, np.nan)
    rows = []
    for lo, hi in zip([50, 100, 200, 400, 700, 1000, 1500], [100, 200, 400, 700, 1000, 1500, 3000]):
        s = (ke >= lo) & (ke < hi); r_ = lead_pP_r[s]; f_ = lead_pP_f[s]
        if s.sum() > 100: rows.append({"lo": lo, "hi": hi, "n": int(s.sum()), "frac_with_pfit": [float(np.isfinite(r_).mean()), float(np.isfinite(f_).mean())],
                                       "real_q16_50_84": np.nanpercentile(r_, [16, 50, 84]).tolist() if np.isfinite(r_).any() else None, "fake_q16_50_84": np.nanpercentile(f_, [16, 50, 84]).tolist() if np.isfinite(f_).any() else None})
    M["lead_proton_P_by_true_KE"] = rows
    # physical-units validation against the truth (proton-fit rate, reco proton P, prong count, PID score)
    lead_cls = X[:, names.index("lead_had_cls")].astype(int)
    nch = d["n_charged_true"][ir].astype(int) if "n_charged_true" in d else None
    M["validation"] = _prong_validation(ke, lead_pP_r, lead_pP_f, Er, Ef, nch, lead_cls, Preal, Pfake, pf_r, pf_f, off_real_p, off_fake_p, out / "figures")
    (out / "metrics_tier2.json").write_text(json.dumps(M, indent=1, default=float))
    _figures(M, Preal, Pfake, kin_r, kin_f, pf_r, pf_f, Er, Ef, off_real, off_fake, out / "figures")
    _tables(M, out / "tables")
    return M


def _profile(ax, x, yr, yf, edges, ylabel, xlabel, log=True, frac=False):
    xs, mr, mf, lr, hr, lf, hf = [], [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() < 100: continue
        xs.append(np.sqrt(lo * hi) if log else 0.5 * (lo + hi))
        if frac:
            mr.append(np.isfinite(yr[s]).mean()); mf.append(np.isfinite(yf[s]).mean())
        else:
            a = yr[s][np.isfinite(yr[s])]; b = yf[s][np.isfinite(yf[s])]
            qa = np.percentile(a, [16, 50, 84]) if len(a) > 20 else [np.nan] * 3; qb = np.percentile(b, [16, 50, 84]) if len(b) > 20 else [np.nan] * 3
            mr.append(qa[1]); lr.append(qa[0]); hr.append(qa[2]); mf.append(qb[1]); lf.append(qb[0]); hf.append(qb[2])
    ax.plot(xs, mr, "o-", color=plots.PALETTE["real"], ms=4, label="MasterAnaDev"); ax.plot(xs, mf, "s--", color=plots.PALETTE["model"], ms=4, label="surrogate")
    if not frac:
        ax.fill_between(xs, lr, hr, color=plots.PALETTE["real"], alpha=0.15); ax.fill_between(xs, lf, hf, color=plots.PALETTE["model"], alpha=0.15)
    if log: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False)


def _prong_validation(ke, lead_pP_r, lead_pP_f, Er, Ef, nch, lead_cls, Pr, Pf, pf_r, pf_f, offr, offf, fdir):
    """Truth-conditional hadron validation in physical units. Returns a dict of the plotted numbers."""
    out = {}
    edges = np.array([50, 80, 120, 180, 270, 400, 600, 900, 1400, 2500])
    fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
    _profile(axs[0], ke, lead_pP_r, lead_pP_f, edges, "fraction of events with a proton fit", "true leading proton KE [MeV]", frac=True)
    _profile(axs[1], ke, lead_pP_r, lead_pP_f, edges, "leading proton-hypothesis P [MeV]", "true leading proton KE [MeV]")
    axs[1].set_yscale("log")
    if nch is not None:
        rows = []
        for k in range(0, 7):
            s = nch == k
            if s.sum() > 100: rows.append((k, Er[s, 1].mean(), Ef[s, 1].mean(), Er[s, 0].mean(), Ef[s, 0].mean(), int(s.sum())))
        out["prongs_vs_true_charged"] = rows
        axs[2].plot([r[0] for r in rows], [r[3] for r in rows], "o-", color=plots.PALETTE["real"], ms=4, label="MasterAnaDev, all prongs")
        axs[2].plot([r[0] for r in rows], [r[4] for r in rows], "s--", color=plots.PALETTE["model"], ms=4, label="surrogate, all prongs")
        axs[2].plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=plots.PALETTE["third"], ms=4, label="MasterAnaDev, with kinematics")
        axs[2].plot([r[0] for r in rows], [r[2] for r in rows], "s--", color=plots.PALETTE["fourth"], ms=4, label="surrogate, with kinematics")
        axs[2].set_xlabel("true charged hadrons (p, pi, K) after FSI"); axs[2].set_ylabel("mean reco prongs per event"); axs[2].legend(frameon=False, fontsize=7)
    plots.save(fig, fdir / "m3_prong_conditionals.png")
    # PID: proton score of prongs in events whose leading true hadron is a proton vs a charged pion; P in MeV
    seg_r = np.repeat(np.arange(len(offr) - 1), np.diff(offr)); seg_f = np.repeat(np.arange(len(offf) - 1), np.diff(offf))
    is_p_r, is_p_f = lead_cls[seg_r] == 4, lead_cls[seg_f] == 4
    is_pi_r, is_pi_f = np.isin(lead_cls[seg_r], [5, 6]), np.isin(lead_cls[seg_f], [5, 6])
    fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
    b = np.linspace(0, 1, 40)
    axs[0].hist(Pr[pf_r & is_p_r, 6], bins=b, histtype="step", color=plots.PALETTE["real"], density=True, label="MasterAnaDev, leading true p")
    axs[0].hist(Pf[pf_f & is_p_f, 6], bins=b, histtype="step", color=plots.PALETTE["model"], density=True, ls="--", label="surrogate, leading true p")
    axs[0].hist(Pr[pf_r & is_pi_r, 6], bins=b, histtype="step", color=plots.PALETTE["third"], density=True, label=r"MasterAnaDev, leading true $\pi^\pm$")
    axs[0].hist(Pf[pf_f & is_pi_f, 6], bins=b, histtype="step", color=plots.PALETTE["fourth"], density=True, ls="--", label=r"surrogate, leading true $\pi^\pm$")
    axs[0].set_xlabel("proton score1 of prongs with a proton fit"); axs[0].set_ylabel("density"); axs[0].legend(frameon=False, fontsize=7)
    kin_r, kin_f = Pr[:, 2] > 0.5, Pf[:, 2] > 0.5
    plots.hist_compare(axs[1], Pr[kin_r & (Pr[:, 3] > 0), 3], Pf[kin_f, 3], np.linspace(0, 1500, 60), "pion-hypothesis P [MeV]", ("MasterAnaDev", "surrogate"))
    plots.hist_compare(axs[2], np.degrees(Pr[kin_r, 0]), np.degrees(Pf[kin_f, 0]), np.linspace(0, 180, 60), "prong angle to beam [deg]", ("MasterAnaDev", "surrogate")); axs[2].set_yscale("log")
    plots.save(fig, fdir / "m3_prong_pid.png")
    out["score_medians"] = {"lead_p_real": float(np.median(Pr[pf_r & is_p_r, 6])), "lead_p_fake": float(np.median(Pf[pf_f & is_p_f, 6])),
                            "lead_pi_real": float(np.median(Pr[pf_r & is_pi_r, 6])), "lead_pi_fake": float(np.median(Pf[pf_f & is_pi_f, 6]))}
    return out


def _figures(M, Pr, Pf, kr, kf, pr, pf, Er, Ef, offr, offf, fdir):
    fig, axs = plots.plt.subplots(1, 4, figsize=(14, 3.2))
    plots.hist_compare(axs[0], Pr[kr, 0], Pf[kf, 0], np.linspace(0, 3.2, 64), "prong theta (beam frame) [rad]", ("MasterAnaDev", "surrogate"))
    plots.hist_compare(axs[1], np.log(Pr[kr & (Pr[:, 3] > 0), 3]), np.log(Pf[kf, 3]), np.linspace(3.5, 8.5, 64), "log pion-hypothesis P [MeV]", ("MasterAnaDev", "surrogate"))
    plots.hist_compare(axs[2], np.log(Pr[pr, 5]), np.log(Pf[pf, 5]), np.linspace(5, 8.5, 64), "log proton-hypothesis P [MeV]", ("MasterAnaDev", "surrogate"))
    plots.hist_compare(axs[3], Pr[pr, 6], Pf[pf, 6], np.linspace(0, 1, 50), "proton score1", ("MasterAnaDev", "surrogate"))
    plots.save(fig, fdir / "m3_prongs.png")
    fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, j, nm in ((axs[0], 1, "prongs with kinematics per event"), (axs[1], 2, "proton fits per event"), (axs[2], 3, "exiting prongs per event")):
        plots.bar_compare(ax, np.bincount(Er[:, j].astype(int), minlength=6)[:6], np.bincount(Ef[:, j].astype(int), minlength=6)[:6], nm, ("MasterAnaDev", "surrogate"))
    plots.save(fig, fdir / "m3_prong_event.png")
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    plots.hist_compare(axs[0], offr, offf, np.linspace(-15, 15, 121), "reco vertex z - nearest plane [mm]", ("MasterAnaDev", "surrogate")); axs[0].set_yscale("log")
    plots.bar_compare(axs[1], np.array(M["vertex_class"]["marginal_real"]), np.array(M["vertex_class"]["marginal_sampled"]), "vertex class (0 unsnapped, 4 nearest plane, 8 far plane)", ("MasterAnaDev", "surrogate"))
    plots.save(fig, fdir / "m3_vertex.png")


def _tables(M, tdir):
    pm = M["prong_marginals"]; pc = M["prong_counts"]
    rows = [f"fraction with kinematics & {pc['has_kin'][0]:.3f} & {pc['has_kin'][1]:.3f} & -- \\\\", f"fraction with proton fit & {pc['has_p'][0]:.3f} & {pc['has_p'][1]:.3f} & -- \\\\", f"fraction exiting & {pc['is_exiting'][0]:.3f} & {pc['is_exiting'][1]:.3f} & -- \\\\"]
    for k, lab in (("theta", r"$\theta$ [rad]"), ("log_pi_P", r"$\log P_\pi$"), ("log_p_P", r"$\log P_p$"), ("p_score1", "proton score")):
        v = pm[k]; r, f = v["real_q16_50_84"], v["fake_q16_50_84"]
        rows.append(f"{lab} & {r[1]:.3f} [{r[0]:.3f}, {r[2]:.3f}] & {f[1]:.3f} [{f[0]:.3f}, {f[2]:.3f}] & {v['w1']:.4f} \\\\")
    (tdir / "prongs.tex").write_text("\\begin{tabular}{lccc}\n\\toprule\nProng variable & MasterAnaDev & surrogate & $W_1$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    v = M["vertex_class"]; o = M["vertex_offset_to_nearest_plane"]; c = M["closure_prongs"]
    (tdir / "tier2_closure.tex").write_text("\\begin{tabular}{lc}\n\\toprule\nTest & value \\\\\n\\midrule\n"
        f"vertex class log loss (marginal / head) & {v['logloss_marginal']:.4f} / {v['logloss']:.4f} \\\\\nsnapped fraction (MasterAnaDev / surrogate) & {v['snapped_frac_real']:.3f} / {v['snapped_frac_sampled']:.3f} \\\\\n"
        f"reco $z$ within 0.1\\,mm of a plane (MasterAnaDev / surrogate) & {o['real_frac_within_0.1mm']:.3f} / {o['fake_frac_within_0.1mm']:.3f} \\\\\n\\midrule\n"
        f"AUC: event prong summary only & {c['event_summary_only']:.3f} \\\\\nAUC: truth features + prong summary & {c['truth_plus_event_summary']:.3f} \\\\\nAUC: truth features + reco vertex $z$ & {c['truth_plus_vertex_z']:.3f} \\\\\n" + "\\bottomrule\n\\end{tabular}\n")
