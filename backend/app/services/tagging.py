"""Image in, tag dictionary out: {fine_type, coarse_type, pattern, colors, material, embedding}."""
from __future__ import annotations

import numpy as np

from app.services.embedding import get_backend
from app.services.imaging import DISPLAY_SIZE, aspect_ratio, load_image, preprocess
from ml.color import dominant_colors, nearest_named_color
from ml.taxonomy import FINE_TO_COARSE, coarse_of


def tag_image(data: bytes, type_hint: str | None = None, wear: str | None = None) -> dict:
    """Tag an uploaded photo. `type_hint` (a fine label) overrides the predicted type; `wear="menswear"` stops
    the tagger from predicting dresses and skirts."""
    img = load_image(data)
    aspect = aspect_ratio(img)
    clean, fg, display = preprocess(img, DISPLAY_SIZE)
    backend = get_backend()
    emb = backend.embed(clean, fg)
    tags = backend.tag(clean, emb, aspect, wear)
    if type_hint:
        if type_hint not in FINE_TO_COARSE:
            raise ValueError(f"unknown type '{type_hint}'")
        tags["fine_type"], tags["type_confidence"] = type_hint, 1.0
    colors = [
        {"name": nearest_named_color(rgb), "rgb": list(rgb), "fraction": round(frac, 3)}
        for rgb, frac in dominant_colors(fg, k=3, drop_background=False)
    ]
    return {
        **tags,
        "coarse_type": coarse_of(tags["fine_type"]),
        "colors": colors,
        "embedding": np.asarray(emb, dtype=np.float32),
        "display": display,   # background-removed photo for the app; the original is kept too
    }
