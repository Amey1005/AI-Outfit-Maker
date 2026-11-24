# Project status - what was built, what was verified, what is left

> **Update (November 2026):** this page describes the state on 21 October 2026. Since then the compatibility model
> was trained and evaluated on real Polyvore data (test FITB 0.531, AUC 0.772 - see [results.md](results.md)), the app
> runs end to end with the trained model, and background removal, occasion rules, per-item menswear/womenswear,
> type/color correction and per-category buy-next suggestions were added. Still open: tagging and occasion accuracy
> on real photos, Docker/CI runs, deployment and the user test.

As of 21 October 2026. Checked against your plan, week-wise plan and progress report.

## Summary

All the **code** for Weeks 1-5 now exists and is tested: embedding, training, evaluation, tagging, recommender, API, database, React app, Docker, CI, docs. What does **not** exist yet are the things that need your data, your accounts or your friends: the real Polyvore embeddings/training run (the dataset is gated behind your Hugging Face login), your own labeled photos, the deployed link, and the friend test. Those are listed in "What is left", with the exact command for each.

## Week by week

| Week | Plan | Status |
|---|---|---|
| 1 Learn and baseline | PyTorch basics, FashionCLIP embeddings, tagging, colors, 3 baselines | **Your work, already done** per the progress report (embeddings, 94-96% zero-shot on 50 images, cosine pairs). Code for the baselines is now in `ml/scorers.py` and `ml/color.py`. Not done: your own 50 photos + `my_items.csv`, and the baseline table on your own outfits. |
| 2 Train the model | Polyvore, cached embeddings, type-aware net, `evaluate.py` | **Code complete** (`embed_polyvore.py`, `train.py`, `evaluate.py`, `models/compat.py`). **Not run on real data** - needs your HF login. Verified only on synthetic data. |
| 3 Pipeline | rembg + tagging, candidates, occasion, weather | **Done and tested** (`backend/app/services/`). Occasion *accuracy* on your 200 labeled outfits not measured (`ml/occasion_eval.py` ready). |
| 4 The app | FastAPI, DB, React, latency | **Done**; API tested, UI exercised in a browser against the API. |
| 5 Polish | Personalization, gap analysis, Docker, CI, deploy, user test, README | Personalization and gap analysis **done and tested**. Docker/CI files written but **not executed** (no Docker on this machine; CI not run). Deploy, user test and demo GIF **not done**. README written. |

## What was verified (and how)

* `ruff check .` passes. `pytest ml/tests` (3) and `pytest backend/tests` (12) pass.
* Backend tests cover: upload + tagging + color naming, bad uploads, per-user isolation, valid outfit templates, the no-item-in-more-than-2-of-top-10 rule, hot/cold/rain weather rules, occasion changing the ranking, item matches, a simulated "likes only blue" user shifting results bluer, gap analysis, delete-my-data (rows and files), latency endpoint, weather parsing + 30-minute cache.
* The real FashionCLIP path (`EMBED_BACKEND=fashionclip`) loads and runs the recommender CLI end to end on a synthetic wardrobe. Its tags on flat drawn shapes are noise (e.g. "knit jeans") - that says nothing about accuracy on real photos.
* Frontend `npm run build` succeeds; wardrobe grid and outfit suggestions were checked in a browser against a live API.
* The training code learns a signal on synthetic data and beats random and cosine baselines there. **This is a plumbing test, not a result.**

## What was NOT verified - be careful here

* `ml/embed_polyvore.py` and the Polyvore file parsers (`ml/data/polyvore.py`) were written from the dataset's documented layout but **never run against the real dataset** (gated, 401 without your login). Expect to fix small things - e.g. the exact format of the compatibility/FITB tokens (your report says to inspect these first; the loader maps `set_id_index` tokens through the outfit JSON and passes through anything that is already an item id).
* Docker images, `docker compose up`, and the GitHub Actions workflow have never been executed.
* rembg was not installed here; the border-mask fallback is what ran. Background removal on real, messy photos is untested.
* No real wardrobe photos were available, so tagging/color accuracy on real photos is unmeasured.
* The occasion and "gap" scoring weights (`OCCASION_WEIGHT`, `PERSONAL_WEIGHT` in `recommender.py`) are sensible guesses, not tuned values. With no checkpoint the recommender falls back to color rules and the occasion score can dominate (a sandals outfit ranked high for "formal" in a lite-backend run).
* Latency: 11.6 ms p50 / 17.6 ms p95 was measured on the lite backend with 24 synthetic items - not a number for the real app.

## Design decisions worth knowing

* Pair scorer per **unordered type pair** with two heads (one per side); fallback shared head (`--shared-head`) for ablation.
* Negatives are random same-type items (presumed negatives); training pairs and evaluation never mix because the disjoint split and valid/test files are kept separate and the best epoch is picked on validation.
* The embedding cache loader merges chunks and **de-duplicates ids**, and accepts your existing `embeddings_001.npz` (`item_ids`/`ids`, `embeddings`/`emb` keys). Older chunks have no color, so the color baseline needs chunks made by the new embed script (it also stores dominant RGB).
* FITB/AUC only score questions whose items are all in the cache and report coverage (`used/total`), so a partial cache cannot silently over-claim.
* The `lite` backend (color-histogram embeddings, heuristic tags) exists so the app, tests and CI run without a 600 MB download. It is never the default.

## What is left (in order)

1. **Run the pipeline in Colab** ([notebook](../ml/notebooks/02_polyvore_pipeline.ipynb)): log in to Hugging Face, `embed_polyvore` (valid + test + up to ~20k train items), `train`, `evaluate --split valid`, then `--split test` once. Fix any format surprises in the loaders. Paste the table into [results.md](results.md). This is the credibility core of the project.
2. Photograph your ~50 items (flat, plain background, daylight); write `my_items.csv` (filename, type, color, pattern) and about 30 outfits; measure tagging accuracy on photos you did **not** tune prompts on.
3. Hand-label ~200 outfits by occasion (150 Polyvore test, 50 yours) and run `ml.occasion_eval`; use `--logreg` if prompts are weak.
4. Copy `compat.pt` to `ml/checkpoints/`, run the app, check `"scorer": "compatibility_model"`, tune the two weights by eye on your wardrobe.
5. `docker compose up --build` locally; fix whatever breaks; push to GitHub and confirm CI is green.
6. Deploy ([DEPLOY.md](DEPLOY.md)), watch RAM limits.
7. Friend test ([USER_TEST.md](USER_TEST.md)), demo GIF, architecture diagram.
8. Fill X / Y / Z in your resume bullets **only** from `results.md`.

## Where your existing work plugs in

Your Drive files (`outfit-maker-data/polyvore_embedding_cache/embeddings_001.npz`, `my_items.csv`, `dataset_manifest.csv`, ...) are not on this machine. Copy `embeddings_001.npz` into the cache folder used by the notebook and `embed_polyvore` will skip those 1,000 items and embed the rest (including rgb colors for new ones).
