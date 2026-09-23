"""Mixture density network with diagonal Gaussian components (M1 baseline for continuous targets)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_components: int = 8, hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.k, self.n_out = n_components, n_out
        self.head = nn.Linear(hidden, n_components * (1 + 2 * n_out))

    def forward(self, x):
        h = self.head(self.body(x))
        logit, mu, log_sig = torch.split(h, [self.k, self.k * self.n_out, self.k * self.n_out], dim=-1)
        mu = mu.view(-1, self.k, self.n_out)
        log_sig = log_sig.view(-1, self.k, self.n_out).clamp(-7, 3)
        return logit, mu, log_sig

    def nll(self, x, y):
        logit, mu, log_sig = self(x)
        z = (y[:, None, :] - mu) / log_sig.exp()
        logp = -0.5 * (z**2).sum(-1) - log_sig.sum(-1) - 0.5 * self.n_out * math.log(2 * math.pi)
        return -torch.logsumexp(F.log_softmax(logit, -1) + logp, -1)

    @torch.no_grad()
    def sample(self, x, n: int = 1):
        logit, mu, log_sig = self(x)
        idx = torch.distributions.Categorical(logits=logit).sample((n,)).T  # [B,n]
        b = torch.arange(len(x), device=x.device)[:, None]
        m, s = mu[b, idx], log_sig[b, idx].exp()
        return m + s * torch.randn_like(m)  # [B,n,n_out]
