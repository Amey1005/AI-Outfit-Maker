"""Four interchangeable outfit scorers (random, color rules, raw cosine, trained model) and the metrics
that evaluate any of them the same way: fill-in-the-blank accuracy and compatibility AUC."""
from __future__ import annotations

import itertools

import numpy as np

from ml.color import color_compat
from ml.data.polyvore import EmbeddingStore


class Scorer:
    name = "base"

    def pair_scores(self, store: EmbeddingStore, ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomScorer(Scorer):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def pair_scores(self, store, ra, rb):
        return self.rng.random(len(ra))


class CosineScorer(Scorer):
    name = "cosine"

    def pair_scores(self, store, ra, rb):
        return (store.emb[ra] * store.emb[rb]).sum(1)


class ColorRuleScorer(Scorer):
    name = "color_rules"

    def pair_scores(self, store, ra, rb):
        out = np.full(len(ra), 0.5)
        for i, (a, b) in enumerate(zip(ra, rb)):
            ca, cb = store.rgb[a], store.rgb[b]
            if not (np.isnan(ca).any() or np.isnan(cb).any()):
                out[i] = color_compat(ca, cb)
        return out


class ModelScorer(Scorer):
    name = "model"

    def __init__(self, model):
        self.model = model

    def pair_scores(self, store, ra, rb):
        return self.model.score_numpy(store.emb[ra], store.types[ra], store.emb[rb], store.types[rb])


def outfit_scores(scorer: Scorer, store: EmbeddingStore, outfits: list[list[int]]) -> np.ndarray:
    """Mean pairwise score for each outfit (a list of store rows). All pairs are scored in one batch."""
    ra, rb, seg = [], [], []
    for k, rows in enumerate(outfits):
        for a, b in itertools.combinations(rows, 2):
            ra.append(a)
            rb.append(b)
            seg.append(k)
    s = scorer.pair_scores(store, np.array(ra), np.array(rb))
    sums = np.bincount(seg, weights=s, minlength=len(outfits))
    counts = np.bincount(seg, minlength=len(outfits))
    return sums / np.maximum(counts, 1)


def _rows(store, item_ids):
    """Rows of the items that are embedded and have a supported coarse type.

    Accessories (bags, jewellery, ...) are never embedded, so they are left out rather than
    disqualifying the whole outfit - almost every Polyvore outfit contains at least one."""
    rows = []
    for i in item_ids:
        r = store.row(i)
        if r is not None and store.types[r] >= 0:
            rows.append(r)
    return rows


def fitb_accuracy(scorer: Scorer, store: EmbeddingStore, questions: list[dict]):
    """Fill-in-the-blank accuracy. A question is used when all of its answers are embedded clothing items
    (so questions whose blank is an accessory are skipped) and at least one question item is; accessories
    in the question are dropped. Returns (accuracy, n_used, n_total)."""
    outfits, owners = [], []
    for q in questions:
        qrows, arows = _rows(store, q["question"]), _rows(store, q["answers"])
        if len(arows) != len(q["answers"]) or len(arows) < 2 or not qrows:
            continue
        for a in arows:
            outfits.append(qrows + [a])
        owners.append((len(arows), q.get("correct", 0)))
    if not owners:
        return float("nan"), 0, len(questions)
    s = outfit_scores(scorer, store, outfits)
    correct, pos = 0, 0
    for n, c in owners:
        seg = s[pos:pos + n]
        tied = seg >= seg.max() - 1e-12
        correct += tied[c] / tied.sum()  # a tie earns only a share, so constant scores get chance level
        pos += n
    return correct / len(owners), len(owners), len(questions)


def compat_auc(scorer: Scorer, store: EmbeddingStore, compat: list[tuple[int, list[str]]]):
    """Compatibility AUC over labelled outfits. Returns (auc, n_used, n_total)."""
    from sklearn.metrics import roc_auc_score

    outfits, labels = [], []
    for label, ids in compat:
        rows = _rows(store, ids)
        if len(rows) >= 2:
            outfits.append(rows)
            labels.append(label)
    if len(set(labels)) < 2:
        return float("nan"), len(labels), len(compat)
    return float(roc_auc_score(labels, outfit_scores(scorer, store, outfits))), len(labels), len(compat)
