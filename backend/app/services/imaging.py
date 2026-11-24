"""Ingestion: remove the background, crop, pad to a square and resize to 224 so phone photos look like
Polyvore product shots (this reduces the photo-domain gap)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

from app.config import settings
from ml.color import rgb_to_lab

SIZE = 224
DISPLAY_SIZE = 640  # the background-removed photo shown in the app
_rembg_session = None
_rembg_failed = False


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _rembg_mask(img: Image.Image):
    """Foreground mask from rembg (u2netp). Returns None when rembg is unavailable."""
    global _rembg_session, _rembg_failed
    if _rembg_failed or not settings.use_rembg:
        return None
    try:
        from rembg import new_session, remove

        if _rembg_session is None:
            _rembg_session = new_session("u2netp")
        cut = remove(img, session=_rembg_session)
        return np.asarray(cut.convert("RGBA"))[..., 3] > 128
    except Exception:
        _rembg_failed = True
        return None


def _border_mask(img: Image.Image) -> np.ndarray:
    """Fallback for plain backgrounds: pixels that differ from the median border color are foreground."""
    a = np.asarray(img.resize((256, 256))).astype(np.float64)
    border = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb_to_lab(a) - rgb_to_lab(bg), axis=-1)
    mask = dist > 12
    if mask.mean() < 0.01 or mask.mean() > 0.97:
        mask = np.ones_like(mask)
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size)) > 128


def preprocess(img: Image.Image, display_size: int | None = None):
    """-> (224x224 RGB image on white, foreground pixels [N,3] uint8), plus a `display_size` version of the same
    cut-out when asked (background removal runs once for both)."""
    mask = _rembg_mask(img)
    if mask is None or mask.mean() < 0.01:
        mask = _border_mask(img)
    arr = np.asarray(img)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        ys, xs, mask = np.arange(arr.shape[0]), np.arange(arr.shape[1]), np.ones(arr.shape[:2], bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.where(mask[y0:y1, x0:x1, None], arr[y0:y1, x0:x1], 255).astype(np.uint8)
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    square = Image.fromarray(canvas)
    out = square.resize((SIZE, SIZE), Image.LANCZOS)
    if display_size:
        # a little white margin so the garment does not touch the card edges
        pad = int(side * 0.06)
        framed = Image.new("RGB", (side + 2 * pad, side + 2 * pad), (255, 255, 255))
        framed.paste(square, (pad, pad))
        return out, arr[mask], framed.resize((display_size, display_size), Image.LANCZOS)
    return out, arr[mask]


def cutout_name(original_name: str) -> str:
    """`abc.jpg` -> `abc_cut.jpg` (the background-removed photo shown in the app)."""
    stem = original_name.rsplit(".", 1)[0]
    return f"{stem}_cut.jpg"


def original_name(shown_name: str) -> str:
    """Inverse of cutout_name; returns the name unchanged for items saved before cut-outs existed."""
    return shown_name.replace("_cut.jpg", ".jpg")


def aspect_ratio(img: Image.Image) -> float:
    """Width / height of the foreground bounding box (used by the lite type guesser)."""
    mask = _border_mask(img)
    ys, xs = np.where(mask)
    return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)) if len(ys) else 1.0
