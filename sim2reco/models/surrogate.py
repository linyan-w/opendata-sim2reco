"""The M2 surrogate: set encoder -> Tier 0 heads, cardinality head, Tier 1 conditional flow matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SetEncoder
from .flow import FlowMatcher
from .prongflow import ProngFlow

N_PRONG_CLASSES = 9  # 0..8
TIER1_DIM = 9  # MODEL_COLS of sim2reco.data.compact
PRONG_DIM = 10  # sim2reco.data.prongs_tf.PRONG_DIM
N_VTX_CLASSES = 9  # sim2reco.data.prongs_tf.VertexPlaneTable.N_CLASSES
PRONG_DIM_CAP = 8  # sim2reco.data.compact.N_PRONG_CAP


class Surrogate(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, flow_hidden=512, flow_layers=4, tier2=False,
                 prong_layers=3):
        super().__init__()
        self.enc = SetEncoder(d_model, n_heads, n_layers)
        self.tier0 = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3))
        self.card = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, N_PRONG_CLASSES))
        self.flag_emb = nn.Linear(2, 32)
        self.n_emb = nn.Embedding(N_PRONG_CLASSES, 32)
        self.flow = FlowMatcher(TIER1_DIM, d_model + 64, flow_hidden, flow_layers)
        self.tier2 = tier2
        if tier2:
            # vertex-plane class head (unsnapped / snapped to nearest plane + delta) on z, flags and N
            # input: flow conditioning + the phase of the true z within the plane cycle (+ sin/cos of it)
            self.vtx = nn.Sequential(nn.Linear(d_model + 64 + 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, N_VTX_CLASSES))
            # prong set flow, conditioned on z, flags, N, and the Tier 1 vector; cross-attends particle tokens
            self.prong = ProngFlow(PRONG_DIM, d_model, d_model + 64 + TIER1_DIM, prong_layers)

    def encode(self, b, return_tokens=False):
        z, h = self.enc(b["cls"], b["mom"], b["mask"], b["ctx"])
        return (z, h) if return_tokens else z

    def flow_cond(self, z, flags, nprong):
        return torch.cat([z, self.flag_emb(flags), self.n_emb(nprong.clamp(0, N_PRONG_CLASSES - 1))], -1)

    def vtx_logits(self, cond, phase):
        ph = phase[:, None] * 3.14159265
        return self.vtx(torch.cat([cond, phase[:, None], ph.sin(), ph.cos()], -1))

    def losses(self, b):
        z, h = self.encode(b, return_tokens=True)
        t0 = b["tier0"]; reco = t0[:, 0] > 0; minos = reco & (t0[:, 1] > 0)
        logits = self.tier0(z)
        l_exist = F.binary_cross_entropy_with_logits(logits[:, 0], t0[:, 0])
        l_minos = F.binary_cross_entropy_with_logits(logits[reco, 1], t0[reco, 1]) if reco.any() else logits.sum() * 0
        l_charge = F.binary_cross_entropy_with_logits(logits[minos, 2], t0[minos, 2]) if minos.any() else logits.sum() * 0
        l_card = F.cross_entropy(self.card(z[reco]), b["nprong"][reco]) if reco.any() else logits.sum() * 0
        fl = reco & (b["valid"] > 0)   # exclude the rare corrupt tuple entries from the flow loss
        cond = self.flow_cond(z[fl], t0[fl, 1:3], b["nprong"][fl])
        l_flow = self.flow.loss(b["x1"][fl], cond).mean() if fl.any() else logits.sum() * 0
        out = {"exist": l_exist, "minos": l_minos, "charge": l_charge, "card": l_card, "flow": l_flow}
        if self.tier2:
            out["vtx"] = F.cross_entropy(self.vtx_logits(cond, b["vphase"][fl]), b["vclass"][fl]) if fl.any() else logits.sum() * 0
            pm = b["pmask"][fl]; has = pm.any(1)
            if has.any():
                pc = torch.cat([cond[has], b["x1"][fl][has]], -1)
                out["prong"] = self.prong.loss(b["prongs"][fl][has], pc, h[fl][has], b["mask"][fl][has], pm[has])
            else:
                out["prong"] = logits.sum() * 0
        return out

    @torch.no_grad()
    def predict_probs(self, b):
        z = self.encode(b)
        return torch.sigmoid(self.tier0(z)), F.softmax(self.card(z), -1), z

    @torch.no_grad()
    def sample(self, b, n_steps=64, teacher_flags=None, teacher_nprong=None):
        """Sample Tier 0 flags, N prongs and Tier 1 (model space). Optionally condition on given flags/N."""
        p0, pn, z = self.predict_probs(b)
        u = torch.rand_like(p0)
        # The heads are conditional probabilities: p(reco exists | X), p(MINOS ok | reco), p(charge<0 | MINOS ok).
        # `exist` is a mask for the caller; minos/charge and the Tier 1 vector describe the event *if* it is
        # reconstructed and must not be zeroed by a False `exist` (that biased the closure test toward the
        # unmatched-muon mode).
        exist = u[:, 0] < p0[:, 0]
        minos = u[:, 1] < p0[:, 1]
        charge = minos & (u[:, 2] < p0[:, 2])
        flags = torch.stack([minos, charge], -1).float() if teacher_flags is None else teacher_flags
        nprong = torch.multinomial(pn, 1)[:, 0] if teacher_nprong is None else teacher_nprong
        cond = self.flow_cond(z, flags, nprong)
        x1 = self.flow.sample(cond, n_steps)
        out = {"exist": exist, "minos": flags[:, 0] > 0, "charge": flags[:, 1] > 0, "nprong": nprong, "x1": x1, "p0": p0, "pn": pn}
        if self.tier2:
            pv = F.softmax(self.vtx_logits(cond, b["vphase"]), -1)
            out["vclass"] = torch.multinomial(pv, 1)[:, 0]; out["pv"] = pv
            _, h = self.encode(b, return_tokens=True)
            N = int(min(nprong.max().item(), PRONG_DIM_CAP)) if len(nprong) else 0
            pmask = torch.arange(max(N, 1), device=z.device)[None, :] < nprong.clamp(max=PRONG_DIM_CAP)[:, None]
            out["pmask"] = pmask
            out["prongs"] = self.prong.sample(torch.cat([cond, x1], -1), h, b["mask"], pmask, n_steps) if N > 0 else torch.zeros(len(z), 1, PRONG_DIM, device=z.device)
        return out
