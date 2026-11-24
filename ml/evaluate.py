"""Evaluate random / color rules / cosine / trained model the same way (FITB accuracy + compatibility AUC).

    python -m ml.evaluate --data-dir <polyvore root> --cache-dir <chunks> --checkpoint ml/checkpoints/compat.pt --split test

Run with --split valid while iterating; run --split test once at the end. Only questions/outfits whose items are
all in the embedding cache are scored, and the coverage is printed so you never over-claim.
"""
from __future__ import annotations

import argparse
import json
import os

from ml.data import polyvore as pv
from ml.models.compat import load_checkpoint
from ml.scorers import ColorRuleScorer, CosineScorer, ModelScorer, RandomScorer, compat_auc, fitb_accuracy


def run(store, fitb, compat, scorers):
    rows = []
    for s in scorers:
        acc, nq, tq = fitb_accuracy(s, store, fitb)
        auc, nc, tc = compat_auc(s, store, compat)
        rows.append({"method": s.name, "fitb_acc": acc, "fitb_n": nq, "fitb_total": tq,
                     "auc": auc, "auc_n": nc, "auc_total": tc})
    return rows


def to_markdown(rows, split):
    lines = [f"| Method | FITB accuracy ({split}) | FITB questions used | AUC ({split}) | AUC outfits used |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['fitb_acc']:.3f} | {r['fitb_n']}/{r['fitb_total']} | "
                     f"{r['auc']:.3f} | {r['auc_n']}/{r['auc_total']} |")
    lines.append("")
    lines.append("FITB chance level is 0.25 (4 candidates). Color-rule scores need `rgb` in the cache; "
                 "without it that row is meaningless (all pairs score 0.5).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    d = os.path.join(a.data_dir, "disjoint")
    store = pv.load_cache(a.cache_dir, pv.load_item_types(os.path.join(a.data_dir, "polyvore_item_metadata.json")))
    res = pv.build_resolver(os.path.join(d, f"{a.split}.json"))
    fitb = pv.load_fitb(os.path.join(d, f"fill_in_blank_{a.split}.json"), res)
    compat = pv.load_compat(os.path.join(d, f"compatibility_{a.split}.txt"), res)
    model, meta = load_checkpoint(a.checkpoint)
    rows = run(store, fitb, compat, [RandomScorer(), ColorRuleScorer(), CosineScorer(), ModelScorer(model)])
    print(to_markdown(rows, a.split))
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
