# Results

Only measured numbers belong here. Cells marked **TBD** have not been measured yet - fill them from the commands shown, and copy only these numbers into your resume bullets.

## 1. Compatibility model on Polyvore (disjoint split, test set)

Measured 15 November 2026 in Colab (T4 GPU), test split run once:
`python -m ml.evaluate --data-dir ... --cache-dir ... --checkpoint ml/checkpoints/compat.pt --split test`

| Method | FITB accuracy (chance 0.25) | Compatibility AUC |
|---|---|---|
| Random | 0.254 | 0.499 |
| Color rules | 0.274 | 0.536 |
| Raw cosine (FashionCLIP) | 0.502 | 0.712 |
| **Type-aware model** | **0.531** | **0.772** |

* Scored 8,791 of 15,145 FITB questions and 28,280 of 30,290 compatibility outfits. A FITB question is skipped when its
  blank is an accessory (bags, jewellery, ...), because only tops, bottoms, shoes, outerwear and one-piece items are
  modelled. Because of that filter, these numbers are **not directly comparable** with published Polyvore results.
* Model vs raw cosine: +2.9 points FITB, +0.060 AUC. Both baselines and the model are scored by the same code.
* Training: FashionCLIP embeddings of 39,805 train items (49,213 positive within-outfit pairs), cosine triplet loss,
  margin 0.3, Adam lr 1e-3, batch 512, 25 epochs; the epoch with the best validation FITB was kept (validation:
  FITB 0.534 on 1,690 questions, AUC 0.785 on 5,722 outfits).
* Doubling the train items (20,252 -> 39,805) moved validation FITB 0.514 -> 0.534 and AUC 0.772 -> 0.785.
* Negatives are random same-type swaps (presumed incompatible), not confirmed bad outfits.

Ablations (validation split only): shared head vs per-type heads, embedding size 64 vs 128 - TBD.

## 2. Tagging accuracy on your own photos (`my_items.csv`)

| Attribute | Accuracy | n |
|---|---|---|
| Type | TBD | TBD |
| Pattern | TBD | TBD |
| Color | TBD | TBD |

Context from the progress report (not independent evidence): zero-shot FashionCLIP on 50 downloaded clothing photos scored 47/50, then 48/50 after prompts were tuned on those same images. 48/50 is an optimistic estimate; re-measure on photos you did not tune on.

## 3. Occasion scoring (`python -m ml.occasion_eval`)

| Method | Accuracy | Majority baseline | n |
|---|---|---|---|
| CLIP prompts | TBD | TBD | ~200 |
| Logistic regression (5-fold CV) | TBD | TBD | ~200 |

## 4. Latency

Measured on this development machine with `python backend/scripts/bench_latency.py` using the **lite backend, 24 synthetic items, CPU**, in-process (no network): `/outfits/suggest` p50 11.6 ms, p95 17.6 ms (cold first call 33 ms). This shows the ranking code is cheap; it is **not** a figure for the deployed FashionCLIP app. Re-run against the real backend and deployment before quoting a number.

## 5. Blind user test (see [USER_TEST.md](USER_TEST.md))

| Source | Mean rating (1-5) | n raters |
|---|---|---|
| Model | TBD | TBD |
| Color rules | TBD | TBD |
| Random | TBD | TBD |

## Code verification (what has actually been run)

* `pytest ml/tests` - 3 tests pass. The training test uses **synthetic** data purely to prove the pipeline learns something and checkpoints round-trip; its scores mean nothing about real fashion.
* `pytest backend/tests` - 12 tests pass (upload/tagging, user isolation, outfit validity + diversity cap, weather rules, occasion ranking, matches, personalization shift, gap analysis, delete-my-data, latency endpoint, weather parsing/caching).
