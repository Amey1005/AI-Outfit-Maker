"""Type-aware pairwise compatibility network.

Each unordered coarse-type pair (e.g. top-bottom) has its own pair of small MLPs (512 -> 256 -> 128).
A pair's score is the cosine similarity of the two projected vectors; an outfit's score is the mean of
its pairwise scores.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.taxonomy import COARSE_TYPES

N_TYPES = len(COARSE_TYPES)
PAIRS = list(itertools.combinations_with_replacement(range(N_TYPES), 2))
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}
_PAIR_LOOKUP = np.zeros((N_TYPES, N_TYPES), dtype=np.int64)
for (_a, _b), _i in PAIR_INDEX.items():
    _PAIR_LOOKUP[_a, _b] = _PAIR_LOOKUP[_b, _a] = _i


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class TypeAwareCompat(nn.Module):
    def __init__(self, in_dim=512, hidden=256, out_dim=128, typed=True):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim, self.typed = in_dim, hidden, out_dim, typed
        n = len(PAIRS) if typed else 1
        # two heads per pair: side 0 projects the lower-type item, side 1 the other one
        self.heads = nn.ModuleList(
            [nn.ModuleList([_mlp(in_dim, hidden, out_dim), _mlp(in_dim, hidden, out_dim)]) for _ in range(n)]
        )

    def _project(self, x, pid, side):
        out = torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
        keys = pid if self.typed else torch.zeros_like(pid)
        for k in keys.unique().tolist():
            m = keys == k
            out[m] = self.heads[k][side](x[m])
        return F.normalize(out, dim=-1)

    def score_pairs(self, xa, ta, xb, tb):
        """Cosine score for item pairs. xa/xb [B,D]; ta/tb coarse type indices [B] (long tensors)."""
        swap = ta > tb
        lo_x = torch.where(swap[:, None], xb, xa)
        hi_x = torch.where(swap[:, None], xa, xb)
        lookup = torch.as_tensor(_PAIR_LOOKUP, device=xa.device)
        pid = lookup[ta, tb]
        return (self._project(lo_x, pid, 0) * self._project(hi_x, pid, 1)).sum(-1)

    @torch.no_grad()
    def score_numpy(self, xa: np.ndarray, ta: np.ndarray, xb: np.ndarray, tb: np.ndarray) -> np.ndarray:
        dev = next(self.parameters()).device

        def t(a, dt=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)

        return self.score_pairs(t(xa), t(ta, torch.long), t(xb), t(tb, torch.long)).cpu().numpy()

    def config(self):
        return {"in_dim": self.in_dim, "hidden": self.hidden, "out_dim": self.out_dim, "typed": self.typed}


def save_checkpoint(model: TypeAwareCompat, path: str, meta: dict | None = None):
    torch.save({"config": model.config(), "state": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TypeAwareCompat(**ck["config"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck.get("meta", {})
