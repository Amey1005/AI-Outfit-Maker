"""Outfit recommendation: candidate generation, pairwise scoring, occasion + weather + taste re-ranking.

    python -m app.services.recommender --wardrobe-dir path/to/photos --occasion college [--temp-c 12]

Run from `backend/`. Photo file names may start with a fine type (e.g. `jeans_blue01.jpg`, `formal_shoes_black.jpg`)
to skip type prediction; otherwise the tagger predicts it.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.services import weather as wx
from ml.color import color_compat
from ml.taxonomy import COARSE_TYPES, OCCASION_ALLOWED, OCCASION_DISCOURAGED, OCCASIONS, occasion_allows
from ml.taxonomy import FINE_TO_COARSE as _FINE_COARSE

# (required slots...) in coarse-type order. Outerwear is an optional extra layer.
TEMPLATES = [
    ("top", "bottom", "shoes"),
    ("top", "bottom", "shoes", "outerwear"),
    ("allbody", "shoes"),
    ("allbody", "shoes", "outerwear"),
]
OCCASION_WEIGHT = 0.5    # how much the occasion probability moves the final score
PERSONAL_WEIGHT = 0.3     # how much the user's taste vector moves the final score
DISCOURAGED_PENALTY = 0.3  # subtracted per fraction of items the occasion discourages (e.g. sneakers for formal)

# Fallback occasion fit by fine type, used when text embeddings are unavailable (lite backend).
RULE_OCCASION = {
    "formal": {"shirt": 1, "trousers": 1, "formal shoes": 1, "jacket": 0.9, "coat": 0.9, "skirt": 0.7, "dress": 0.8},
    "gym": {"t-shirt": 1, "shorts": 1, "sneakers": 1, "track pants": 1, "track jacket": 1, "hoodie": 0.8, "sweater": 0.6},
    "party": {"dress": 1, "skirt": 0.9, "boots": 0.9, "formal shoes": 0.8, "shirt": 0.8, "jacket": 0.8, "jeans": 0.6},
    "college": {"t-shirt": 1, "jeans": 1, "sneakers": 1, "hoodie": 0.9, "shirt": 0.7, "sweater": 0.8, "trousers": 0.6},
}


@dataclass
class WItem:
    id: int
    fine_type: str
    coarse_type: str
    material: str
    rgb: tuple
    emb: np.ndarray
    pattern: str = "solid"
    color_name: str = ""
    wear: str = "unisex"


# ---------------------------------------------------------------- pair scorers
class ModelPairScorer:
    name = "compatibility_model"

    def __init__(self, model, threshold=None):
        self.model, self.threshold = model, threshold          # threshold is in raw cosine space

    def matrix(self, items):
        from ml.taxonomy import COARSE_INDEX

        n = len(items)
        emb = np.stack([i.emb for i in items])
        typ = np.array([COARSE_INDEX[i.coarse_type] for i in items])
        a, b = np.triu_indices(n, 1)
        m = np.eye(n)
        if len(a):
            s = (self.model.score_numpy(emb[a], typ[a], emb[b], typ[b]) + 1) / 2   # map cosine to [0,1]
            m[a, b] = m[b, a] = s
        return m

    def mapped_threshold(self):
        return None if self.threshold is None else (self.threshold + 1) / 2


class ColorRulePairScorer:
    """Fallback when no trained checkpoint exists: color-wheel rules (the Week-1 baseline)."""
    name = "color_rules"
    threshold = None

    def matrix(self, items):
        n = len(items)
        m = np.eye(n)
        for a, b in itertools.combinations(range(n), 2):
            m[a, b] = m[b, a] = color_compat(items[a].rgb, items[b].rgb)
        return m

    def mapped_threshold(self):
        return None


def load_scorer():
    if settings.checkpoint.exists():
        from ml.models.compat import load_checkpoint

        model, meta = load_checkpoint(str(settings.checkpoint))
        return ModelPairScorer(model, (meta or {}).get("threshold_p75"))
    return ColorRulePairScorer()


# ---------------------------------------------------------------- generation + ranking
def generate(items, M, allowed=None, force=None):
    """Vectorised candidate generation. Returns [(combos [K,m] item indices, base scores [K])] per template.

    base = mean pairwise score of the outfit. `allowed` restricts to a subset of item indices; `force` keeps only
    outfits that contain that item (used for "what matches this item?" and wardrobe-gap analysis).
    """
    pool = range(len(items)) if allowed is None else allowed
    by = {c: [i for i in pool if items[i].coarse_type == c] for c in COARSE_TYPES}
    out = []
    for tpl in TEMPLATES:
        lists = [list(by[c]) for c in tpl]
        if force is not None:
            fc = items[force].coarse_type
            if fc not in tpl:
                continue
            lists[tpl.index(fc)] = [force]
        if any(not l for l in lists):
            continue
        combos = np.array(list(itertools.product(*lists)))
        pairs = list(itertools.combinations(range(len(tpl)), 2))
        base = np.mean([M[combos[:, x], combos[:, y]] for x, y in pairs], axis=0)
        out.append((combos, base))
    return out


def _occasion_scores(backend, items, combos, occasion):
    emb = np.stack([i.emb for i in items])
    if backend is not None and getattr(backend, "has_text", False):
        outfit = emb[combos].mean(1)
        return backend.occasion_probs(outfit)[:, OCCASIONS.index(occasion)]
    table = RULE_OCCASION[occasion]
    w = np.array([table.get(i.fine_type, 0.3) for i in items])
    return w[combos].mean(1)


_SLOT_NAMES = {"top": "tops", "bottom": "bottoms", "shoes": "shoes", "outerwear": "layers", "allbody": "one-pieces"}


def _missing_note(items, allowed, occasion):
    """Explain an empty result: which clothing slot has nothing suitable for this occasion."""
    have = {items[k].coarse_type for k in allowed}
    if {"top", "bottom", "shoes"} <= have or {"allbody", "shoes"} <= have:
        return None
    missing = [c for c in ("top", "bottom", "shoes") if c not in have]
    hint = ""
    if occasion in OCCASION_ALLOWED:
        ok = sorted(f for f in OCCASION_ALLOWED[occasion] if f in _FINE_COARSE and _FINE_COARSE[f] in missing)
        hint = f" (for example {', '.join(ok)})" if ok else ""
    return f"Your wardrobe has no {occasion}-suitable {' or '.join(_SLOT_NAMES[c] for c in missing)}{hint}. " \
           "Add some, or fix an item's type on the Wardrobe page if it was tagged wrongly."


def suggest(items, M, occasion, weather=None, pref=None, backend=None, top_k=10, max_repeat=2):
    """Rank outfits for an occasion. Returns {"outfits": up to `top_k` dicts, "notes": [str]}.

    Items the occasion does not allow (e.g. a party jacket for the gym) are never used. No item appears in more
    than `max_repeat` outfits. If it is cold and no allowed warm layer exists, outfits are returned without one
    and a note says so.
    """
    if occasion not in OCCASIONS:
        raise ValueError(f"occasion must be one of {OCCASIONS}")
    allowed = [k for k, it in enumerate(items) if occasion_allows(it.fine_type, occasion)]
    notes = []
    rows = _rank(items, M, occasion, weather, pref, backend, allowed, require_layer=True)
    if not rows and weather and weather["temp_c"] < wx.WEATHER_RULES["cold_below_c"]:
        rows = _rank(items, M, occasion, weather, pref, backend, allowed, require_layer=False)
        if rows:
            notes.append(f"It is cold, but you have no warm layer that suits {occasion} "
                         "(a hoodie, sweater or track jacket). Wear something warm over these.")
    if not rows:
        note = _missing_note(items, allowed, occasion)
        notes.append(note or "No outfit passes the weather rules with this wardrobe - try adding layers or other items.")
    # The diversity cap only applies where there is a choice: if a slot has fewer than 3 suitable items (e.g. the
    # only pair of shoes), those items may appear in every outfit.
    per_slot = {}
    for k in allowed:
        per_slot[items[k].coarse_type] = per_slot.get(items[k].coarse_type, 0) + 1
    capped = {k for k in allowed if per_slot[items[k].coarse_type] >= 3}
    used, out = {}, []
    for final, combo, b, o, p in rows:
        if any(i in capped and used.get(i, 0) >= max_repeat for i in combo):
            continue
        for i in combo:
            used[i] = used.get(i, 0) + 1
        out.append({"items": [items[i].id for i in combo], "score": final, "compatibility": b, "occasion": o, "taste": p})
        if len(out) == top_k:
            break
    return {"outfits": out, "notes": notes}


def _rank(items, M, occasion, weather, pref, backend, allowed, require_layer):
    emb = np.stack([i.emb for i in items]) if items else None
    discouraged = np.array([it.fine_type in OCCASION_DISCOURAGED.get(occasion, set()) for it in items], dtype=float)
    mens = np.array([it.wear == "menswear" for it in items])
    womens = np.array([it.wear == "womenswear" for it in items])
    rows = []
    for combos, base in generate(items, M, allowed=allowed):
        keep = np.array([wx.outfit_allowed([items[j] for j in c], weather, require_layer) for c in combos], dtype=bool)
        keep &= ~(mens[combos].any(1) & womens[combos].any(1))
        if not keep.any():
            continue
        combos, base = combos[keep], base[keep]
        occ = _occasion_scores(backend, items, combos, occasion)
        pers = np.zeros(len(combos))
        if pref is not None and np.linalg.norm(pref) > 0:
            mean = emb[combos].mean(1)
            pers = (mean @ pref) / (np.linalg.norm(mean, axis=1) * np.linalg.norm(pref) + 1e-8)
        final = base + OCCASION_WEIGHT * occ + PERSONAL_WEIGHT * pers - DISCOURAGED_PENALTY * discouraged[combos].mean(1)
        rows += [(float(f), tuple(int(x) for x in c), float(b), float(o), float(p))
                 for f, c, b, o, p in zip(final, combos, base, occ, pers)]
    rows.sort(key=lambda r: -r[0])
    return rows


def matches_for(items, M, index, limit=8):
    """Best partner items for one item (same pairwise matrix, sliced and sorted). Same-type items are excluded."""
    me = items[index]
    cand = [(float(M[index, j]), items[j].id) for j in range(len(items))
            if j != index and items[j].coarse_type != me.coarse_type]
    cand.sort(reverse=True)
    return [{"item_id": i, "score": s} for s, i in cand[:limit]]


# ---------------------------------------------------------------- CLI
def _load_dir(path):
    from pathlib import Path

    from app.services.tagging import tag_image
    from ml.color import nearest_named_color
    from ml.taxonomy import FINE_LABELS

    items = []
    for k, f in enumerate(sorted(p for p in Path(path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})):
        hint = next((l for l in FINE_LABELS if f.stem.lower().startswith(l.replace(" ", "_"))), None)
        t = tag_image(f.read_bytes(), hint)
        rgb = tuple(t["colors"][0]["rgb"])
        items.append((f.name, WItem(k, t["fine_type"], t["coarse_type"], t["material"], rgb, t["embedding"],
                                    t["pattern"], nearest_named_color(rgb))))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wardrobe-dir", required=True)
    ap.add_argument("--occasion", default="college", choices=OCCASIONS)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--temp-c", type=float, default=None, help="skip the weather API and use this temperature")
    ap.add_argument("--rain", action="store_true")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    a = ap.parse_args()

    from app.services.embedding import get_backend

    named = _load_dir(a.wardrobe_dir)
    items = [it for _, it in named]
    names = {it.id: n for n, it in named}
    weather = {"temp_c": a.temp_c, "rain": a.rain} if a.temp_c is not None else (
        wx.fetch_weather(a.lat, a.lon) if a.lat is not None else None)
    scorer = load_scorer()
    print(f"{len(items)} items | scorer: {scorer.name} | weather: {wx.condition(weather)}")
    for it in items:
        print(f"  {names[it.id]:32s} {it.fine_type:13s} {it.color_name:7s} {it.pattern:8s} {it.material}")
    out = suggest(items, scorer.matrix(items), a.occasion, weather, backend=get_backend(), top_k=a.top)
    res = out["outfits"]
    for note in out["notes"]:
        print("NOTE:", note)
    print(f"\nTop {len(res)} outfits for '{a.occasion}':")
    for k, r in enumerate(res, 1):
        print(f" {k}. score {r['score']:.3f} (compat {r['compatibility']:.2f}, occasion {r['occasion']:.2f}): "
              + " + ".join(names[i] for i in r["items"]))


if __name__ == "__main__":
    main()
