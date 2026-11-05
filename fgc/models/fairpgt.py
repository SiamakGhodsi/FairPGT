"""FairPGT: partitioned graph transformer for fair unsupervised node clustering.

The forward pass takes only node inputs [X || U || d~] and the (precomputed) METIS
partition. Sensitive attributes never enter it: they are used only by the losses,
which receive the attention matrices returned by the forward pass.

Layer l (pre-norm):
  local attention within each partition, self excluded:  A_p = softmax(Q_p K_p^T / sqrt(d_h))
  routing: tokens r_p = mean_{i in V_p} LN(h_i); token self-attention; node -> token
           cross-attention B (broadcast)
  h^ = h + (1 - g) * W_O [A_p V_p]_i + g * route_i ;  h' = h^ + FFN(LN(h^))
DEC head: z_i = LN(W_z h_i^(L)),  C_ik ∝ (1 + ||z_i - mu_k||^2)^-1
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split(x, heads):                      # (..., n, D) -> (..., H, n, D/H)
    *lead, n, D = x.shape
    return x.view(*lead, n, heads, D // heads).transpose(-3, -2)


def _merge(x):                             # (..., H, n, d) -> (..., n, H*d)
    *lead, H, n, d = x.shape
    return x.transpose(-3, -2).reshape(*lead, n, H * d)


class Partitioner:
    """Index bookkeeping for padded, batched attention over partitions."""

    def __init__(self, parts: torch.Tensor):
        self.parts = parts
        P = int(parts.max()) + 1
        sizes = torch.bincount(parts, minlength=P)
        self.P, self.n_max = P, int(sizes.max())
        order = torch.argsort(parts, stable=True)
        starts = torch.cumsum(sizes, 0) - sizes
        pos = torch.arange(len(parts), device=parts.device) - starts[parts[order]]
        self.idx = torch.full((P, self.n_max), -1, dtype=torch.long, device=parts.device)
        self.idx[parts[order], pos] = order                    # (P, n_max) node ids, -1 = padding
        self.valid = self.idx >= 0
        self.flat = self.idx[self.valid]                       # node ids in padded order

    def gather(self, h):                                       # (n, D) -> (P, n_max, D)
        out = h.new_zeros(self.P, self.n_max, h.shape[-1])
        out[self.valid] = h[self.flat]
        return out

    def scatter(self, x, n):                                   # (P, n_max, D) -> (n, D)
        out = x.new_zeros(n, x.shape[-1])
        out[self.flat] = x[self.valid]
        return out


class LocalAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, part: Partitioner):
        """x: (n, D) normalized states. Returns (n, D) messages and attention (P, H, m, m)."""
        n = x.shape[0]
        xb = part.gather(x)                                    # (P, m, D)
        q, k, v = (_split(t, self.heads) for t in self.qkv(xb).chunk(3, -1))
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])          # (P, H, m, m)
        m = part.n_max
        keymask = part.valid[:, None, None, :] & ~torch.eye(m, dtype=torch.bool, device=x.device)
        scores = scores.masked_fill(~keymask, -1e9)
        A = torch.softmax(scores, -1) * keymask
        A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)       # rows without keys -> 0
        A = A * part.valid[:, None, :, None]
        msg = _merge(self.drop(A) @ v)                         # dropout only on the aggregation
        return part.scatter(self.out(msg), n), A


class Routing(nn.Module):
    """One token per partition: mean pooling, token self-attention, node->token broadcast."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.tok_qkv = nn.Linear(dim, 3 * dim)
        self.tok_out = nn.Linear(dim, dim)
        self.tok_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, part: Partitioner):
        """x: (n, D) normalized states. Returns (n, D) messages and broadcast attention (H, n, P)."""
        P = part.P
        R = torch.zeros(P, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, part.parts, x)
        R = R / torch.bincount(part.parts, minlength=P).clamp_min(1)[:, None].to(x.dtype)
        q, k, v = (_split(t, self.heads) for t in self.tok_qkv(self.tok_norm(R)).chunk(3, -1))
        T = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        R = R + self.tok_out(_merge(T @ v))                    # (P, D)
        qn = _split(self.q(x), self.heads)                     # (H, n, d)
        kt, vt = (_split(t, self.heads) for t in self.kv(R).chunk(2, -1))
        B = torch.softmax(qn @ kt.transpose(-1, -2) / math.sqrt(qn.shape[-1]), -1)   # (H, n, P)
        return self.out(_merge(self.drop(B) @ vt)), B


class Layer(nn.Module):
    def __init__(self, dim, heads, dropout, routing=True):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.local = LocalAttention(dim, heads, dropout)
        self.routing = Routing(dim, heads, dropout) if routing else None
        self.gate = nn.Parameter(torch.zeros(()))             # gamma = sigmoid(gate), starts at 0.5
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, h, part):
        x = self.norm1(h)
        loc, A = self.local(x, part)
        if self.routing is not None:
            glob, B = self.routing(x, part)
            g = torch.sigmoid(self.gate)
            msg = (1 - g) * loc + g * glob
        else:
            B = None
            msg = loc
        h = h + msg
        h = h + self.ffn(self.norm2(h))
        return h, A, B, msg


class FairPGT(nn.Module):
    def __init__(self, in_dim, n_clusters, hidden=256, heads=4, layers=3, proj_dim=32,
                 dropout=0.1, routing=True):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([Layer(hidden, heads, dropout, routing) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.centers = nn.Parameter(torch.zeros(n_clusters, proj_dim))

    def encode(self, x, part, return_attn=False):
        h = self.inp(x)
        attn = []
        for layer in self.layers:
            h, A, B, msg = layer(h, part)
            if return_attn:
                attn.append((A, B))
        return self.proj(h), attn

    def assign(self, z):
        d = torch.cdist(z, self.centers).pow(2)
        C = 1.0 / (1.0 + d)
        return C / C.sum(1, keepdim=True)

    def forward(self, x, part, return_attn=False):
        z, attn = self.encode(x, part, return_attn)
        return z, self.assign(z), attn

    def gates(self):
        return [float(torch.sigmoid(l.gate)) for l in self.layers if l.routing is not None]
