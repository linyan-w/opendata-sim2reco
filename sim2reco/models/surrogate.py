"""The M2 surrogate: set encoder -> Tier 0 heads, cardinality head, Tier 1 conditional flow matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SetEncoder
from .flow import FlowMatcher

N_PRONG_CLASSES = 9  # 0..8
TIER1_DIM = 11


class Surrogate(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, flow_hidden=512, flow_layers=4):
        super().__init__()
        self.enc = SetEncoder(d_model, n_heads, n_layers)
        self.tier0 = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3))
        self.card = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, N_PRONG_CLASSES))
        self.flag_emb = nn.Linear(2, 32)
        self.n_emb = nn.Embedding(N_PRONG_CLASSES, 32)
        self.flow = FlowMatcher(TIER1_DIM, d_model + 64, flow_hidden, flow_layers)

    def encode(self, b):
        z, _ = self.enc(b["cls"], b["mom"], b["mask"], b["ctx"])
        return z

    def flow_cond(self, z, flags, nprong):
        return torch.cat([z, self.flag_emb(flags), self.n_emb(nprong.clamp(0, N_PRONG_CLASSES - 1))], -1)

    def losses(self, b):
        z = self.encode(b)
        t0 = b["tier0"]; reco = t0[:, 0] > 0; minos = reco & (t0[:, 1] > 0)
        logits = self.tier0(z)
        l_exist = F.binary_cross_entropy_with_logits(logits[:, 0], t0[:, 0])
        l_minos = F.binary_cross_entropy_with_logits(logits[reco, 1], t0[reco, 1]) if reco.any() else logits.sum() * 0
        l_charge = F.binary_cross_entropy_with_logits(logits[minos, 2], t0[minos, 2]) if minos.any() else logits.sum() * 0
        l_card = F.cross_entropy(self.card(z[reco]), b["nprong"][reco]) if reco.any() else logits.sum() * 0
        fl = reco & (b["valid"] > 0)   # exclude the rare corrupt tuple entries from the flow loss
        cond = self.flow_cond(z[fl], t0[fl, 1:3], b["nprong"][fl])
        l_flow = self.flow.loss(b["x1"][fl], cond).mean() if fl.any() else logits.sum() * 0
        return {"exist": l_exist, "minos": l_minos, "charge": l_charge, "card": l_card, "flow": l_flow}

    @torch.no_grad()
    def predict_probs(self, b):
        z = self.encode(b)
        return torch.sigmoid(self.tier0(z)), F.softmax(self.card(z), -1), z

    @torch.no_grad()
    def sample(self, b, n_steps=64, teacher_flags=None, teacher_nprong=None):
        """Sample Tier 0 flags, N prongs and Tier 1 (model space). Optionally condition on given flags/N."""
        p0, pn, z = self.predict_probs(b)
        u = torch.rand_like(p0)
        exist = u[:, 0] < p0[:, 0]
        minos = exist & (u[:, 1] < p0[:, 1])
        charge = minos & (u[:, 2] < p0[:, 2])
        flags = torch.stack([minos, charge], -1).float() if teacher_flags is None else teacher_flags
        nprong = torch.multinomial(pn, 1)[:, 0] if teacher_nprong is None else teacher_nprong
        x1 = self.flow.sample(self.flow_cond(z, flags, nprong), n_steps)
        return {"exist": exist, "minos": flags[:, 0] > 0, "charge": flags[:, 1] > 0, "nprong": nprong, "x1": x1, "p0": p0, "pn": pn}
