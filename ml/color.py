"""Color utilities: LAB k-means dominant colors, named colors, and color-wheel compatibility rules.

Pure numpy so it can be shared by the training scripts and the backend without OpenCV.
"""
from __future__ import annotations

import colorsys

import numpy as np

NAMED_COLORS = {
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "grey": (128, 128, 128),
    "beige": (222, 204, 170),
    "brown": (110, 70, 40),
    "red": (200, 30, 40),
    "orange": (235, 130, 30),
    "yellow": (240, 210, 50),
    "green": (50, 140, 70),
    "blue": (50, 100, 200),
    "navy": (25, 40, 90),
    "light blue": (150, 180, 215),
    "teal": (0, 128, 128),
    "maroon": (110, 25, 40),
    "purple": (120, 60, 150),
    "pink": (240, 150, 180),
}


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, shape [...,3]) -> CIE LAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


_NAMED_LAB = {n: rgb_to_lab(np.array(v)) for n, v in NAMED_COLORS.items()}


def nearest_named_color(rgb) -> str:
    lab = rgb_to_lab(np.asarray(rgb))
    return min(_NAMED_LAB, key=lambda n: float(np.linalg.norm(lab - _NAMED_LAB[n])))


def _kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[labels == i].mean(0) if (labels == i).any() else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    labels = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    return centers, labels


def dominant_colors(pixels: np.ndarray, k: int = 3, drop_background: bool = True, max_pixels: int = 4000):
    """Return [(rgb tuple, fraction)] sorted by size. `pixels` is [N,3] uint8.

    With drop_background, near-white / near-black pixels are removed first (used when no mask is
    available, e.g. Polyvore product shots with white backgrounds).
    """
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    if drop_background:
        keep = ~((px.min(1) > 235) | (px.max(1) < 12))
        if keep.sum() >= 50:
            px = px[keep]
    if len(px) == 0:
        return [((128, 128, 128), 1.0)]
    if len(px) > max_pixels:
        px = px[np.random.default_rng(0).choice(len(px), max_pixels, replace=False)]
    lab = rgb_to_lab(px)
    centers, labels = _kmeans(lab, k)
    out = []
    for i in range(len(centers)):
        frac = float((labels == i).mean())
        if frac == 0:
            continue
        rgb = px[labels == i].mean(0)
        out.append((tuple(int(v) for v in rgb), frac))
    return sorted(out, key=lambda t: -t[1])


def hsv(rgb):
    r, g, b = (float(v) / 255.0 for v in rgb)
    return colorsys.rgb_to_hsv(r, g, b)  # h in [0,1]


def is_neutral(rgb) -> bool:
    h, s, v = hsv(rgb)
    return s < 0.18 or v < 0.2 or (v > 0.9 and s < 0.25)


def color_compat(rgb_a, rgb_b) -> float:
    """Color-wheel rule score in [0,1]: neutrals match everything, complementary (~180 deg) and
    analogous (within ~30 deg) hues score high, other hue gaps score lower."""
    if is_neutral(rgb_a) or is_neutral(rgb_b):
        return 0.8
    ha, hb = hsv(rgb_a)[0] * 360, hsv(rgb_b)[0] * 360
    d = abs(ha - hb)
    d = min(d, 360 - d)
    analogous = np.exp(-((d / 30.0) ** 2)) * 0.9
    complementary = np.exp(-(((d - 180) / 25.0) ** 2))
    return float(max(analogous, complementary, 0.25))
