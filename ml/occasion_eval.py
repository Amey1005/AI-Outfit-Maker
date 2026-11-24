"""Measure occasion scoring on your hand-labeled outfits against a majority-class baseline.

    python -m ml.occasion_eval --labels data/occasion_labels.csv --cache-dir data/polyvore_embedding_cache

`labels.csv` has two columns: `item_ids` (space-separated item ids that exist in the embedding cache) and `occasion`
(college | party | formal | gym). Label ~150 Polyvore test outfits plus ~50 combos from your own wardrobe (for your own
items, add their embeddings to a chunk in the cache dir). If CLIP prompts are weak, `--logreg` trains a 5-fold
cross-validated logistic regression on outfit embeddings instead.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter

import numpy as np

from ml.taxonomy import OCCASION_PROMPTS, OCCASIONS


def prompt_predict(outfit_embs: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    """outfit_embs [N,D] (normalised mean of item embeddings), text_embs [4,D] -> predicted occasion indices."""
    return (outfit_embs @ text_embs.T).argmax(1)


def evaluate(outfit_embs, labels, text_embs, logreg=False):
    y = np.array([OCCASIONS.index(l) for l in labels])
    majority = Counter(y.tolist()).most_common(1)[0][1] / len(y)
    if logreg:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        folds = min(5, int(np.bincount(y)[np.bincount(y) > 0].min()))
        acc = float(cross_val_score(LogisticRegression(max_iter=2000), outfit_embs, y, cv=max(folds, 2)).mean())
    else:
        acc = float((prompt_predict(outfit_embs, text_embs) == y).mean())
    return {"accuracy": acc, "majority_baseline": majority, "n": len(y), "method": "logreg-5fold" if logreg else "clip-prompts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--logreg", action="store_true")
    a = ap.parse_args()

    from ml.data.polyvore import load_cache
    from ml.fashionclip import FashionClip

    store = load_cache(a.cache_dir)
    embs, labels = [], []
    with open(a.labels, newline="") as f:
        for row in csv.DictReader(f):
            rows = [store.row(i) for i in row["item_ids"].split()]
            if any(r is None for r in rows):
                continue
            e = store.emb[rows].mean(0)
            embs.append(e / np.linalg.norm(e))
            labels.append(row["occasion"].strip().lower())
    clip = FashionClip()
    text = np.stack([clip.embed_texts(OCCASION_PROMPTS[o]).mean(0) for o in OCCASIONS])
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    print(evaluate(np.array(embs), labels, text, a.logreg))


if __name__ == "__main__":
    main()
