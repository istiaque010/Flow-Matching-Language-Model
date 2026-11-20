"""Time-conditioned bidirectional velocity Transformer (Phase 5).

Input  x_t (B, L, d) in the standardised embedding space, t (B,), optional prefix mask (B, L) and key padding mask.
Output v (B, L, d): one velocity vector per position.

  h = in_proj(x_t) + pos_emb + type_emb[is_prefix]      (type_emb tells the model which positions are a clean prompt)
  time: sinusoidal features of t -> MLP -> c (B, width)
  each block (pre-LayerNorm, bidirectional self-attention, no causal mask):
     adaLN conditioning (time_cond="adaln"): LayerNorm without affine parameters, then scale/shift/gate computed from
     c, gates initialised to zero so every block starts as the identity (as in DiT, Peebles & Xie 2023)
  out = Linear(adaLN(h)) -> d, initialised to zero (the network starts by predicting v = 0)

time_cond = "none" removes the time input (ablation): the blocks still use the same adaLN layers but with c = a
learned constant, so the parameter count stays the same and only the information about t is missing.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig:
    d_data: int = 384
    seq_len: int = 64
    n_layer: int = 4
    n_head: int = 4
    width: int = 256
    dropout: float = 0.0
    time_cond: str = "adaln"     # "adaln" | "none"
    time_freqs: int = 64

    def to_dict(self):
        return asdict(self)


def timestep_features(t: torch.Tensor, n: int) -> torch.Tensor:
    """Sinusoidal features of t in [0, 1] (scaled by 1000 as in diffusion models), (B,) -> (B, 2n)."""
    freqs = torch.exp(-math.log(10000.0) * torch.arange(n, dtype=torch.float32, device=t.device) / n)
    a = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    def __init__(self, w, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(w, 3 * w)
        self.proj = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(approximate="tanh"), nn.Linear(4 * w, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(w, 6 * w))
        self.drop = nn.Dropout(dropout)
        self.p_drop = dropout
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, attn_mask):
        B, L, W = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        h = modulate(self.ln1(x), sh1, sc1)
        q, k, v = self.qkv(h).split(W, dim=2)
        q, k, v = (z.view(B, L, self.n_head, W // self.n_head).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, L, W)
        x = x + g1.unsqueeze(1) * self.drop(self.proj(a))
        x = x + g2.unsqueeze(1) * self.drop(self.mlp(modulate(self.ln2(x), sh2, sc2)))
        return x


class VelocityTransformer(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.in_proj = nn.Linear(cfg.d_data, w)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, w))
        self.type_emb = nn.Embedding(2, w)            # 0 = flow variable, 1 = clean prompt position
        self.time_mlp = nn.Sequential(nn.Linear(2 * cfg.time_freqs, w), nn.SiLU(), nn.Linear(w, w))
        self.const_c = nn.Parameter(torch.zeros(1, w))   # used instead of the time embedding when time_cond="none"
        self.blocks = nn.ModuleList([Block(w, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        self.out = nn.Linear(w, cfg.d_data)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        for m in (self.ada_f[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xt, t, prefix_mask=None, key_padding_mask=None):
        """xt (B, L, d); t (B,); prefix_mask (B, L) bool (True = clean prompt); key_padding_mask (B, L) bool
        (True = real token that may be attended to)."""
        B, L, _ = xt.shape
        h = self.in_proj(xt) + self.pos_emb[:, :L]
        if prefix_mask is not None:
            h = h + self.type_emb(prefix_mask.long())
        else:
            h = h + self.type_emb.weight[0]
        if self.cfg.time_cond == "adaln":
            c = self.time_mlp(timestep_features(t, self.cfg.time_freqs))
        else:
            c = self.const_c.expand(B, -1)
        attn_mask = None
        if key_padding_mask is not None and not bool(key_padding_mask.all()):
            attn_mask = key_padding_mask[:, None, None, :]      # (B, 1, 1, L) True = attend
        for blk in self.blocks:
            h = blk(h, c, attn_mask)
        sh, sc = self.ada_f(c).chunk(2, dim=1)
        return self.out(modulate(self.ln_f(h), sh, sc))


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
