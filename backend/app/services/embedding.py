"""Embedding/tagging backends.

* ClipBackend  - real FashionCLIP: zero-shot type/pattern/material tags + 512-d embeddings (production path).
* LiteBackend  - no model download: a color-histogram "embedding" and heuristic tags. For development, CI and
                 offline demos only; its type/material guesses are crude (users can override the type on upload).
Both expose the same interface so the rest of the app does not care which is active.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.config import settings
from ml.color import rgb_to_lab
from ml.taxonomy import MATERIALS, OCCASION_PROMPTS, OCCASIONS, PATTERNS, WOMENSWEAR_ONLY, labels_for

DIM = 512
SPORT_LABELS = {"track pants", "track jacket"}
SPORT_MIN_CONFIDENCE = 0.9
_BINS = 8


def _normalize(v):
    return (v / max(np.linalg.norm(v), 1e-8)).astype(np.float32)


class LiteBackend:
    name = "lite"
    has_text = False

    def embed(self, img: Image.Image, fg_pixels: np.ndarray) -> np.ndarray:
        """8x8x8 histogram over LAB of foreground pixels, square-rooted. Same-colored items get similar vectors."""
        lab = rgb_to_lab(fg_pixels.astype(np.float64))
        lo, hi = np.array([0, -100, -100]), np.array([100, 100, 100])
        idx = np.clip(((lab - lo) / (hi - lo) * _BINS).astype(int), 0, _BINS - 1)
        hist = np.bincount(idx[:, 0] * _BINS * _BINS + idx[:, 1] * _BINS + idx[:, 2], minlength=DIM).astype(np.float64)
        return _normalize(np.sqrt(hist / max(hist.sum(), 1)))

    def tag(self, img, emb, aspect: float, style: str | None = None):
        """Crude heuristics: wide-and-short shapes are shoes, very tall ones dresses, otherwise a top."""
        fine = "sneakers" if aspect > 1.6 else "dress" if aspect < 0.55 else "t-shirt"
        if style == "menswear" and fine in WOMENSWEAR_ONLY:
            fine = "t-shirt"
        return {"fine_type": fine, "type_confidence": 0.3, "pattern": "solid", "material": "cotton"}

    def archetype(self, fine: str, color_name: str, rgb) -> np.ndarray:
        px = np.tile(np.array(rgb, dtype=np.uint8), (400, 1))
        return self.embed(None, px)

    def occasion_probs(self, outfit_embs: np.ndarray):
        return None  # rule-based scoring is used instead


class ClipBackend:
    name = "fashionclip"
    has_text = True

    def __init__(self):
        from ml.fashionclip import FashionClip

        self.clip = FashionClip()
        self._occ = None

    def embed(self, img: Image.Image, fg_pixels) -> np.ndarray:
        return self.clip.embed_images([img])[0]

    def tag(self, img, emb, aspect: float, style: str | None = None):
        labels = labels_for(style)
        fine, conf, probs = self.clip.zero_shot(emb, labels)
        if fine in SPORT_LABELS and conf < SPORT_MIN_CONFIDENCE:
            # Zero-shot confuses ordinary trousers/jackets with track wear; a wrong sport tag hides an item from
            # party/formal outfits, so only accept it when very sure and otherwise take the best non-sport label.
            k = max((i for i, l in enumerate(labels) if l not in SPORT_LABELS), key=lambda i: probs[i])
            fine, conf = labels[k], float(probs[k])
        pat, _, _ = self.clip.zero_shot(emb, PATTERNS, templates=("a photo of {} clothing", "a {} pattern garment"))
        mat, _, _ = self.clip.zero_shot(emb, MATERIALS, templates=("a photo of {} clothing", "a garment made of {}"))
        return {"fine_type": fine, "type_confidence": conf, "pattern": pat, "material": mat}

    def archetype(self, fine: str, color_name: str, rgb) -> np.ndarray:
        return self.clip.embed_texts([f"a product image of a {color_name} {fine}"])[0]

    def occasion_probs(self, outfit_embs: np.ndarray):
        """Softmax over the 4 occasions of cosine(outfit embedding, averaged text prompts). -> [N,4]"""
        if self._occ is None:
            t = np.stack([self.clip.embed_texts(OCCASION_PROMPTS[o]).mean(0) for o in OCCASIONS])
            self._occ = t / np.linalg.norm(t, axis=1, keepdims=True)
        e = outfit_embs / np.linalg.norm(outfit_embs, axis=1, keepdims=True)
        z = 100.0 * e @ self._occ.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)


_backend = None


def get_backend():
    global _backend
    if _backend is None:
        _backend = ClipBackend() if settings.embed_backend == "fashionclip" else LiteBackend()
    return _backend
