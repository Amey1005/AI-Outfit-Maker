"""Synthetic Polyvore-shaped data used ONLY to test that the pipeline runs end to end.

Items get a latent style; an outfit is a set of same-style items of different types. Numbers produced on this
data say nothing about real-world performance and must never be reported as results.
"""
from __future__ import annotations

import numpy as np

from ml.data.polyvore import EmbeddingStore


def make_synthetic(n_outfits=1200, n_styles=8, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    type_vec = rng.normal(size=(3, dim))
    style_a = rng.normal(size=(n_styles, dim))          # style signal seen differently per type
    style_b = rng.normal(size=(3, n_styles, dim))
    ids, emb, types, rgb, outfits = [], [], [], [], []
    by = {(t, s): [] for t in range(3) for s in range(n_styles)}
    for o in range(n_outfits):
        style, items = int(rng.integers(n_styles)), []
        for t in range(3):                              # top, bottom, shoes (coarse indices 0..2)
            v = 1.0 * type_vec[t] + 0.5 * style_b[t, style] + 0.3 * style_a[style] + 0.6 * rng.normal(size=dim)
            iid = f"{o}_{t}"
            ids.append(iid)
            emb.append(v / np.linalg.norm(v))
            types.append(t)
            rgb.append(rng.integers(0, 255, 3).astype(float))
            by[(t, style)].append(iid)
            items.append(iid)
        outfits.append({"set_id": str(o), "items": items, "style": style})
    store = EmbeddingStore(ids, np.array(emb, dtype=np.float32), np.array(types), np.array(rgb), {k: i for i, k in enumerate(ids)})
    return store, outfits


def make_eval(store, outfits, n=300, seed=1):
    """FITB questions (4 candidates, true one at index `correct`) and balanced compatibility labels."""
    rng = np.random.default_rng(seed)
    fitb, compat = [], []
    for _ in range(n):
        o = outfits[int(rng.integers(len(outfits)))]
        blank = int(rng.integers(3))
        true = o["items"][blank]
        q = [i for k, i in enumerate(o["items"]) if k != blank]
        wrong = []
        while len(wrong) < 3:
            c = outfits[int(rng.integers(len(outfits)))]
            if c["style"] != o["style"]:
                wrong.append(c["items"][blank])
        pos = int(rng.integers(4))                      # the true answer sits at a random position
        answers = wrong[:pos] + [true] + wrong[pos:]
        fitb.append({"question": q, "answers": answers, "correct": pos})
        compat.append((1, list(o["items"])))
        neg = list(o["items"])
        neg[blank] = wrong[0]
        compat.append((0, neg))
    return fitb, compat
