"""Resumable, deduplicated FashionCLIP embedding of Polyvore items into chunked .npz files.

    python -m ml.embed_polyvore --data-dir data/polyvore --cache-dir data/polyvore_embedding_cache \
        --max-train-items 20000

Why this is faster than scanning a stream: for each split's parquet file we first read ONLY the `item_id`
column (cheap) to build an id -> row-group index, then read the image bytes of just the row groups that contain
items we still need. Evaluation items (valid/test compatibility + fill-in-the-blank files) are embedded first, so
evaluate.py has full coverage; train items fill the remaining budget. Existing chunks (including your older
embeddings_001.npz) are respected: ids already cached are never re-embedded.

Each chunk stores: ids, emb [N,512] (L2-normalised), rgb [N,3] (dominant non-background color, for the
color-rule baseline).
"""
from __future__ import annotations

import argparse
import glob
import io
import os

import numpy as np

from ml.color import dominant_colors
from ml.data import polyvore as pv

REPO = "mvasil/polyvore-outfits"
SPLITS = {"train": "train", "valid": "validation", "test": "test"}  # json name -> parquet name
CHUNK = 1000


def fetch(name, data_dir):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, name, repo_type="dataset", local_dir=data_dir)


def cached_ids(cache_dir):
    seen = set()
    for p in glob.glob(os.path.join(cache_dir, "*.npz")):
        z = np.load(p, allow_pickle=True)
        key = "ids" if "ids" in z.files else "item_ids"
        seen.update(str(i) for i in z[key])
    return seen


def wanted_items(data_dir, item_types, max_train_items):
    """-> {split: ordered list of item ids}; only items of the 5 supported coarse types."""
    want = {}
    for split in ("valid", "test"):
        res = pv.build_resolver(os.path.join(data_dir, "disjoint", f"{split}.json"))
        ids = []
        for q in pv.load_fitb(os.path.join(data_dir, "disjoint", f"fill_in_blank_{split}.json"), res):
            ids += q["question"] + q["answers"]
        for _, items in pv.load_compat(os.path.join(data_dir, "disjoint", f"compatibility_{split}.txt"), res):
            ids += items
        want[split] = [i for i in dict.fromkeys(ids) if i in item_types]
    train, seen = [], set()
    for o in pv.load_outfits(os.path.join(data_dir, "disjoint", "train.json")):
        items = [i for i in o["items"] if i in item_types and i not in seen]
        if len(seen) + len(items) > max_train_items:
            break
        train += items
        seen.update(items)
    want["train"] = train
    return want


def rgb_of(img):
    px = np.asarray(img.convert("RGB").resize((64, 64)))
    return np.array(dominant_colors(px.reshape(-1, 3), k=3)[0][0], dtype=np.float32)


def embed_split(split, ids_wanted, cache_dir, model, chunk_no, batch_size=128):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image

    todo = set(ids_wanted) - cached_ids(cache_dir)
    if not todo:
        print(f"[{split}] nothing to do")
        return chunk_no
    fs = HfFileSystem()
    path = f"datasets/{REPO}/data/disjoint/{SPLITS[split]}.parquet"
    pf = pq.ParquetFile(fs.open(path, "rb"))
    groups = []
    for rg in range(pf.num_row_groups):  # cheap: reads only the item_id column
        ids = pf.read_row_group(rg, columns=["item_id"]).column("item_id").to_pylist()
        hit = [i for i, x in enumerate(ids) if str(x) in todo]
        if hit:
            groups.append((rg, hit))
    print(f"[{split}] {len(todo)} items to embed across {len(groups)} row groups")
    buf_ids, buf_emb, buf_rgb, done = [], [], [], 0

    def flush():
        nonlocal buf_ids, buf_emb, buf_rgb, chunk_no
        if not buf_ids:
            return
        chunk_no += 1
        out = os.path.join(cache_dir, f"{split}_{chunk_no:04d}.npz")
        np.savez(out, ids=np.array(buf_ids), emb=np.concatenate(buf_emb), rgb=np.stack(buf_rgb))
        print(f"  saved {out} ({len(buf_ids)} items)")
        buf_ids, buf_emb, buf_rgb = [], [], []

    for rg, hit in groups:
        t = pf.read_row_group(rg, columns=["item_id", "image"])
        ids = t.column("item_id").to_pylist()
        imgs = t.column("image").to_pylist()
        sel = [i for i in hit if str(ids[i]) in todo]
        for s in range(0, len(sel), batch_size):
            batch = sel[s:s + batch_size]
            pil = [Image.open(io.BytesIO(imgs[i]["bytes"])).convert("RGB") for i in batch]
            emb = model.embed_images(pil)
            assert np.isfinite(emb).all()
            buf_ids += [str(ids[i]) for i in batch]
            buf_emb.append(emb)
            buf_rgb += [rgb_of(p) for p in pil]
            done += len(batch)
        if len(buf_ids) >= CHUNK:
            flush()
    flush()
    print(f"[{split}] embedded {done}/{len(todo)}")
    return chunk_no


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/polyvore")
    ap.add_argument("--cache-dir", default="data/polyvore_embedding_cache")
    ap.add_argument("--max-train-items", type=int, default=20000)
    ap.add_argument("--splits", nargs="+", default=["valid", "test", "train"])
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    for name in ["polyvore_item_metadata.json"] + [f"disjoint/{n}" for s in ("train", "valid", "test") for n in (
            [f"{s}.json"] + ([f"compatibility_{s}.txt", f"fill_in_blank_{s}.json"] if s != "train" else []))]:
        fetch(name, a.data_dir)
    item_types = pv.load_item_types(os.path.join(a.data_dir, "polyvore_item_metadata.json"))
    want = wanted_items(a.data_dir, item_types, a.max_train_items)
    from ml.fashionclip import FashionClip

    model = FashionClip()
    chunk_no = len(glob.glob(os.path.join(a.cache_dir, "*.npz")))
    for split in a.splits:
        chunk_no = embed_split(split, want[split], a.cache_dir, model, chunk_no)
    store = pv.load_cache(a.cache_dir, item_types)
    print(f"cache now holds {len(store.ids)} unique items; norms ok: {np.allclose(np.linalg.norm(store.emb, axis=1), 1, atol=1e-3)}")


if __name__ == "__main__":
    main()
