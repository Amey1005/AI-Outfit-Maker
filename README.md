# Outfit Maker

An AI wardrobe assistant. Upload photos of your clothes; the app removes the background, tags each item (type, color, pattern, material), shows what matches any item, suggests outfits for an occasion adjusted to today's weather, learns your taste from likes/dislikes, and tells you which new items would unlock the most new outfits.

## Results

Type-aware compatibility model trained on FashionCLIP embeddings of **Polyvore Outfits (disjoint split)**, evaluated once on the held-out test split:

| Method | Fill-in-the-blank accuracy (chance 0.25) | Compatibility AUC |
|---|---|---|
| Random | 0.254 | 0.499 |
| Color-wheel rules | 0.274 | 0.536 |
| Raw FashionCLIP cosine | 0.502 | 0.712 |
| **Type-aware model** | **0.531** | **0.772** |

8,791 FITB questions and 28,280 outfits scored; questions whose blank is an accessory are skipped, so the numbers are not directly comparable with published Polyvore results. Details, training setup and what is still unmeasured (tagging accuracy on real photos, occasion accuracy, latency of the deployed app, user test) are in [docs/results.md](docs/results.md). Every number in this repo was measured; nothing is estimated.

## Features

* **Upload** a photo; you say whether it is menswear, womenswear or unisex. The background is removed (rembg `u2netp`), and the item is tagged by FashionCLIP zero-shot prompts plus LAB k-means colors. Wrong type or color can be fixed in one click.
* **Matches:** tap any item to see what goes with it, ranked by the compatibility model.
* **Outfit suggestions** for college, party, formal or gym, filtered by live weather (Open-Meteo) - e.g. no shorts when it is cold, no suede in rain - and by occasion rules (a party jacket never goes to the gym). Menswear and womenswear are never mixed in one outfit.
* **Taste learning:** likes/dislikes move a per-user preference vector that re-ranks suggestions.
* **What to buy next:** for each category (tops, bottoms, layers, shoes) the named styles - baggy jeans, bomber jacket, loafers... - that would create the most new good outfits, with a shopping-search link.
* **Privacy:** no login (a random browser ID), and one button deletes every photo and record.

## How it works

```
photo -> rembg (u2netp) / border-mask -> crop, pad, 224px -> FashionCLIP -> embedding
                                                         \-> zero-shot type / pattern / material
                                                         \-> LAB k-means dominant colors
wardrobe -> type-aware compatibility net (pairwise) -> candidate outfits (top+bottom+shoes[+outerwear], dress+shoes[+outerwear])
         -> + occasion score (CLIP text prompts) + weather rules (Open-Meteo) + taste vector -> diversity-capped top-k
```

* **Compatibility model** (`ml/models/compat.py`): one pair of small MLPs (512 -> 256 -> 128) per coarse type pair (top-bottom, top-shoes, ...); a pair's score is the cosine of the projections, an outfit's score is the mean pairwise score. Trained with a cosine triplet loss on cached FashionCLIP embeddings of Polyvore Outfits (disjoint split), so training takes minutes.
* **Baselines** (`ml/scorers.py`): random, color-wheel rules, raw cosine similarity - all evaluated by the same code as the model (`ml/evaluate.py`).
* **Fallback:** if no trained checkpoint is present the API scores outfits with color rules and says so in every response (`"scorer": "color_rules"`).

## Run it

**Everything in Docker**

```bash
docker compose up --build
```

App at http://localhost:8080, API docs at http://localhost:8000/docs. FashionCLIP (~600 MB) downloads on first start. `EMBED_BACKEND=lite docker compose up --build` skips the download (color-histogram stand-in, crude tags - for demos only).

**Local development**

```bash
pip install torch && pip install -r requirements-dev.txt
cd backend && EMBED_BACKEND=lite uvicorn app.main:app --reload      # or omit EMBED_BACKEND for real FashionCLIP
cd frontend && npm install && npm run dev                            # http://localhost:5173
```

No photos handy? `python backend/scripts/make_demo_wardrobe.py demo_wardrobe` draws 24 flat-color garments, and `python backend/scripts/seed_demo.py <user-id>` uploads them to a running API. (They are synthetic: good for trying the UI, useless for measuring accuracy.)

**Command-line recommender** (from `backend/`):

```bash
python -m app.services.recommender --wardrobe-dir path/to/photos --occasion college --temp-c 12
```

**Tests and lint**

```bash
ruff check .
pytest ml/tests -q
pytest backend/tests -q
```

## Train and evaluate the compatibility model

Polyvore Outfits is a gated Hugging Face dataset (`mvasil/polyvore-outfits`); request access and log in with your own token. The easiest route is the Colab notebook [ml/notebooks/02_polyvore_pipeline.ipynb](ml/notebooks/02_polyvore_pipeline.ipynb), or locally:

```bash
python -m ml.embed_polyvore --data-dir data/polyvore --cache-dir data/polyvore_embedding_cache --max-train-items 20000
python -m ml.train    --data-dir data/polyvore --cache-dir data/polyvore_embedding_cache --out ml/checkpoints/compat.pt
python -m ml.evaluate --data-dir data/polyvore --cache-dir data/polyvore_embedding_cache --checkpoint ml/checkpoints/compat.pt --split valid
python -m ml.evaluate ... --split test      # once, at the very end
python -m ml.occasion_eval --labels data/occasion_labels.csv --cache-dir data/polyvore_embedding_cache
```

Embedding is resumable and de-duplicated, reads only the row groups it needs from each parquet file, and embeds validation/test items first so evaluation has full coverage. Dataset licence: check the dataset card and keep it noted here before redistributing anything derived from it.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/wardrobe/items` | upload a photo (multipart `file`, optional `type`) -> tagged item |
| GET | `/wardrobe/items` | list the wardrobe |
| DELETE | `/wardrobe/items/{id}` | remove one item |
| GET | `/wardrobe/items/{id}/matches` | what matches this item |
| GET | `/wardrobe/gaps` | top-5 single items that unlock the most new outfits |
| POST | `/outfits/suggest` | `{occasion, lat?, lon?, temp_c?, rain?, top_k?}` -> ranked outfits |
| POST | `/feedback` | like/dislike an outfit (updates the taste vector) |
| POST | `/wear` | log "I wore this" |
| DELETE | `/users/me/data` | erase everything for this user |
| GET | `/metrics/latency` | p50/p95 per route since start |

Identity is a random UUID sent as `X-User-Id` (no login by design).

## Limitations

* **Photo-domain gap:** the compatibility model is trained on clean Polyvore product shots; phone photos are messier. Background removal and the "lay it flat" hint reduce but do not remove the gap. Report any measured drop honestly.
* Polyvore negatives are *presumed* (random same-type swaps), not confirmed bad outfits.
* The fallback `lite` backend and the synthetic demo wardrobe are for development only.
* Free hosting tiers have ~512 MB RAM and may not fit FashionCLIP + PyTorch + rembg - see [docs/DEPLOY.md](docs/DEPLOY.md). Free instances sleep, so the first request is slow.
* Auth is a browser-local UUID, not real login; do not store anything sensitive.

## Layout

```
ml/        embedding, training, evaluation, baselines, occasion eval, Colab notebook
backend/   FastAPI app (routers/, services/), tests, demo scripts, Dockerfile
frontend/  React + Vite + Tailwind
docs/      results.md, STATUS.md, DEPLOY.md, USER_TEST.md
```
