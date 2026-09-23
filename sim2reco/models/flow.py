"""Conditional flow matching (linear/OT path, Gaussian source) for a fixed-size continuous target."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], -1)


class VelocityNet(nn.Module):
    def __init__(self, dim, cond_dim, hidden=512, n_layers=4, t_dim=64):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        layers, d = [], dim + cond_dim + t_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]; d = hidden
        self.body = nn.Sequential(*layers); self.out = nn.Linear(hidden, dim)

    def forward(self, x, t, cond):
        return self.out(self.body(torch.cat([x, cond, self.temb(t)], -1)))


class FlowMatcher(nn.Module):
    def __init__(self, dim, cond_dim, hidden=512, n_layers=4, sigma_min=1e-3):
        super().__init__()
        self.dim, self.sigma_min = dim, sigma_min
        self.v = VelocityNet(dim, cond_dim, hidden, n_layers)

    def loss(self, x1, cond):
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1 - (1 - self.sigma_min) * t)[:, None] * x0 + t[:, None] * x1
        target = x1 - (1 - self.sigma_min) * x0
        return ((self.v(xt, t, cond) - target) ** 2).mean(-1)

    @torch.no_grad()
    def sample(self, cond, n_steps=64):
        """Midpoint (RK2) integration from t=0 to 1."""
        x = torch.randn(len(cond), self.dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((len(cond),), i * dt, device=cond.device)
            k1 = self.v(x, t, cond)
            k2 = self.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond)
            x = x + dt * k2
        return x
