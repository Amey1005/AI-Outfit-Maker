"""Polyvore Outfits loaders (disjoint split) and the embedding cache reader.

File formats follow the dataset repo `mvasil/polyvore-outfits`:
  disjoint/{train,valid,test}.json            [{"set_id", "items": [{"item_id", "index"}]}]
  disjoint/compatibility_*.txt                "<label> <setid_index> <setid_index> ..."
  disjoint/fill_in_blank_*.json               [{"question": [...], "blank_position", "answers": [...]}]
  polyvore_item_metadata.json                 {item_id: {..., "semantic_category": ...}}
Question/answer tokens are "<set_id>_<index>" (1-based); `build_resolver` maps them to item ids.
If a token already is an item id it is passed through unchanged.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from ml.taxonomy import COARSE_INDEX, SEMANTIC_TO_COARSE


def load_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def load_item_types(metadata_path: str) -> dict[str, int]:
    """item_id -> coarse type index; items with ignored categories (bags, jewellery...) are omitted."""
    meta = load_json(metadata_path)
    out = {}
    for item_id, m in meta.items():
        coarse = SEMANTIC_TO_COARSE.get(m.get("semantic_category"))
        if coarse:
            out[item_id] = COARSE_INDEX[coarse]
    return out


def load_outfits(path: str) -> list[dict]:
    """-> [{"set_id": str, "items": [item_id, ...]}]"""
    return [{"set_id": str(o["set_id"]), "items": [str(i["item_id"]) for i in o["items"]]} for o in load_json(path)]


def build_resolver(outfits_raw_path: str) -> dict[str, str]:
    """Map '<set_id>_<index>' tokens to item ids, using the raw outfit json of the same split."""
    res = {}
    for o in load_json(outfits_raw_path):
        for it in o["items"]:
            res[f"{o['set_id']}_{it['index']}"] = str(it["item_id"])
    return res


def load_fitb(path: str, resolver: dict[str, str]):
    """-> [{"question": [item_id...], "answers": [4 item_ids], "correct": index of the true answer}].

    Answers are shuffled in the file; the true one is the answer token that comes from the question's own
    outfit (same set id). Questions where that is ambiguous are dropped."""
    out = []
    for q in load_json(path):
        qsets = {t.rsplit("_", 1)[0] for t in q["question"]}
        match = [k for k, t in enumerate(q["answers"]) if t.rsplit("_", 1)[0] in qsets]
        if len(match) != 1:  # cannot tell which answer is the true one
            continue
        out.append({
            "question": [resolver.get(t, t) for t in q["question"]],
            "answers": [resolver.get(t, t) for t in q["answers"]],
            "correct": match[0],
        })
    return out


def load_compat(path: str, resolver: dict[str, str]):
    """-> [(label, [item_id...])]"""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            out.append((int(parts[0]), [resolver.get(t, t) for t in parts[1:]]))
    return out


@dataclass
class EmbeddingStore:
    ids: list[str]
    emb: np.ndarray            # [N,512] float32, L2-normalised
    types: np.ndarray          # [N] coarse type index, -1 if unknown
    rgb: np.ndarray            # [N,3] float, NaN if unknown
    index: dict[str, int]

    def row(self, item_id: str) -> int | None:
        return self.index.get(item_id)


def _pick(npz, names):
    for n in names:
        if n in npz.files:
            return npz[n]
    raise KeyError(f"none of {names} in {npz.files}")


def load_cache(cache_dir: str, item_types: dict[str, int] | None = None) -> EmbeddingStore:
    """Merge every *.npz chunk in `cache_dir`, de-duplicating item ids (first occurrence wins).

    Accepts keys ids/item_ids and emb/embeddings; `rgb` is optional (older chunks lack it).
    """
    ids, embs, rgbs, seen = [], [], [], set()
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        z = np.load(path, allow_pickle=True)
        c_ids = [str(i) for i in _pick(z, ["ids", "item_ids"])]
        c_emb = _pick(z, ["emb", "embeddings"]).astype(np.float32)
        c_rgb = z["rgb"].astype(np.float64) if "rgb" in z.files else np.full((len(c_ids), 3), np.nan)
        keep = [i for i, k in enumerate(c_ids) if k not in seen]
        for i in keep:
            seen.add(c_ids[i])
        ids += [c_ids[i] for i in keep]
        embs.append(c_emb[keep])
        rgbs.append(c_rgb[keep])
    if not ids:
        raise FileNotFoundError(f"no embedding chunks in {cache_dir}")
    emb = np.concatenate(embs)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    types = np.array([(item_types or {}).get(i, -1) for i in ids])
    return EmbeddingStore(ids, emb, types, np.concatenate(rgbs), {k: i for i, k in enumerate(ids)})
