"""Wardrobe gap analysis: which new items would unlock the most good outfits?

Candidates are named styles (baggy jeans, bomber jacket, chelsea boots, ...) in common colors, embedded with the
active backend (FashionCLIP text embeddings, or solid-color images with the lite backend). Each candidate is scored
by how many good outfits it would form with what the user already owns.

Raw counts always favour whatever category the user owns least of (a second pair of shoes doubles every outfit),
so results are reported per category - best tops, bottoms, layers and shoes - instead of one global list.
"""
from __future__ import annotations

from urllib.parse import quote_plus

import numpy as np

from app.services.recommender import WItem, generate
from ml.color import NAMED_COLORS
from ml.taxonomy import COARSE_TYPES, coarse_of

NEUTRALS = ["black", "white", "grey", "navy", "beige", "brown"]
# (style name shown to the user, fine type used by the outfit/weather/occasion rules, colors to try)
STYLES = [
    ("plain t-shirt", "t-shirt", NEUTRALS + ["green", "blue"]),
    ("oversized t-shirt", "t-shirt", NEUTRALS),
    ("graphic t-shirt", "t-shirt", ["black", "white", "grey"]),
    ("polo shirt", "shirt", NEUTRALS + ["green"]),
    ("oxford shirt", "shirt", ["white", "blue", "pink"]),
    ("linen shirt", "shirt", ["white", "beige", "blue"]),
    ("flannel shirt", "shirt", ["red", "green", "navy", "brown"]),
    ("crewneck sweater", "sweater", NEUTRALS + ["green"]),
    ("hoodie", "hoodie", NEUTRALS + ["green"]),
    ("straight fit jeans", "jeans", ["blue", "black", "grey"]),
    ("baggy jeans", "jeans", ["blue", "black", "grey"]),
    ("slim fit jeans", "jeans", ["blue", "black", "grey"]),
    ("chinos", "trousers", ["beige", "navy", "black", "grey", "green"]),
    ("cargo pants", "trousers", ["black", "beige", "green", "grey"]),
    ("track pants", "track pants", ["black", "grey", "navy"]),
    ("shorts", "shorts", ["black", "grey", "navy", "beige"]),
    ("denim jacket", "jacket", ["blue", "black"]),
    ("bomber jacket", "jacket", ["black", "green", "navy", "brown"]),
    ("leather jacket", "jacket", ["black", "brown"]),
    ("puffer jacket", "coat", ["black", "navy", "beige"]),
    ("overcoat", "coat", ["black", "grey", "beige", "navy"]),
    ("track jacket", "track jacket", ["black", "navy", "grey"]),
    ("white sneakers", "sneakers", ["white"]),
    ("running shoes", "sneakers", ["black", "grey", "white"]),
    ("canvas sneakers", "sneakers", ["white", "black", "navy"]),
    ("chelsea boots", "boots", ["black", "brown"]),
    ("loafers", "formal shoes", ["black", "brown"]),
    ("derby shoes", "formal shoes", ["black", "brown"]),
    ("dress", "dress", ["black", "navy", "red", "beige"]),
    ("midi skirt", "skirt", ["black", "beige", "navy"]),
    ("blouse", "shirt", ["white", "pink", "beige", "black"]),
    ("crop top", "t-shirt", ["white", "black", "pink"]),
    ("heels", "formal shoes", ["black", "beige"]),
    ("ballet flats", "formal shoes", ["black", "beige"]),
]
WOMENSWEAR_STYLES = {"dress", "midi skirt", "blouse", "crop top", "heels", "ballet flats"}
CATEGORY_ORDER = ["top", "bottom", "outerwear", "shoes", "allbody"]
CATEGORY_LABEL = {"top": "Tops", "bottom": "Bottoms", "outerwear": "Jackets & layers", "shoes": "Shoes",
                  "allbody": "One-pieces"}
_catalog_cache: dict[str, list] = {}


def build_catalog(backend):
    """One WItem per (style, color). Cached per backend because text embeddings never change."""
    if backend.name not in _catalog_cache:
        cat = []
        for name, fine, colors in STYLES:
            for color in colors:
                rgb = NAMED_COLORS[color]
                it = WItem(-(len(cat) + 1), fine, coarse_of(fine), "cotton", rgb, backend.archetype(name, color, rgb),
                           "solid", color)
                it.style = name
                cat.append(it)
        _catalog_cache[backend.name] = cat
    return _catalog_cache[backend.name]


def count_good(items, M, allowed, force, threshold):
    return int(sum((base > threshold).sum() for _, base in generate(items, M, allowed=allowed, force=force)))


def shop_url(color: str, name: str, style: str) -> str:
    """A shopping search for the suggested item. Not a single product: picking one needs a store's product API."""
    who = {"menswear": " men", "womenswear": " women"}.get(style, "")
    return "https://www.google.com/search?tbm=shop&q=" + quote_plus(f"{color} {name}{who}")


def _wanted(c, style, owned):
    if style == "menswear":
        return c.style not in WOMENSWEAR_STYLES
    if style == "womenswear":
        return True
    # no preference: suggest one-pieces and other womenswear-only styles only to people who already wear them
    return c.style not in WOMENSWEAR_STYLES or owned["allbody"] > 0


def wardrobe_gaps(items, scorer, backend, per_category=3, top=5, percentile=75, style="all"):
    n = len(items)
    owned = {c: sum(i.coarse_type == c for i in items) for c in COARSE_TYPES}
    catalog = [c for c in build_catalog(backend) if _wanted(c, style, owned)]
    everything = items + catalog
    M = scorer.matrix(everything)
    wardrobe = list(range(n))
    thr = scorer.mapped_threshold()
    if thr is None:  # no Polyvore-derived threshold: use the 75th percentile of the wardrobe's own outfits
        own = [b for _, b in generate(items, M[:n, :n])]
        thr = float(np.percentile(np.concatenate(own), percentile)) if own else 0.5

    scored = []
    for k, c in enumerate(catalog):
        unlocked = count_good(everything, M, wardrobe + [n + k], n + k, thr)
        scored.append({"name": c.style, "color": c.color_name, "fine_type": c.fine_type, "category": c.coarse_type,
                       "new_outfits": unlocked, "shop_url": shop_url(c.color_name, c.style, style)})
    scored.sort(key=lambda r: -r["new_outfits"])

    by_category = []
    for cat in CATEGORY_ORDER:
        picks, seen = [], set()
        for r in scored:
            if r["category"] == cat and r["name"] not in seen and r["new_outfits"] > 0:
                picks.append(r)
                seen.add(r["name"])
            if len(picks) == per_category:
                break
        if picks:
            by_category.append({"category": cat, "label": CATEGORY_LABEL[cat], "owned": owned[cat], "picks": picks})

    # a short overall list that still mixes categories (at most two per category)
    overall, per = [], {}
    for r in scored:
        if per.get(r["category"], 0) < 2 and r["new_outfits"] > 0 and all(r["name"] != o["name"] for o in overall):
            overall.append(r)
            per[r["category"]] = per.get(r["category"], 0) + 1
        if len(overall) == top:
            break
    return {"threshold": thr, "style": style, "suggestions": overall, "by_category": by_category}
