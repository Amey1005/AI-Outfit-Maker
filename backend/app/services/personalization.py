"""Per-user taste vector: nudged toward liked outfits, away from disliked ones."""
from __future__ import annotations

import numpy as np

STEP = 0.15


def to_vec(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32).copy()


def update_pref(pref: np.ndarray | None, item_embs: np.ndarray, liked: bool, step: float = STEP) -> np.ndarray:
    """p += step * mean(outfit embeddings) on like, -= on dislike; the norm is capped at 1."""
    mean = item_embs.mean(0)
    mean = mean / max(np.linalg.norm(mean), 1e-8)
    p = np.zeros_like(mean) if pref is None else pref
    p = p + (step if liked else -step) * mean
    n = np.linalg.norm(p)
    return (p / n if n > 1 else p).astype(np.float32)
