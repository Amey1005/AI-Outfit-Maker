"""Train the type-aware compatibility network on cached FashionCLIP embeddings.

    python -m ml.train --data-dir <polyvore root> --cache-dir <embedding chunks> --out ml/checkpoints/compat.pt

Positives are item pairs from real outfits; the negative for each (anchor, positive) is a random item of the
positive's type (a presumed negative). Loss is a cosine triplet loss. The checkpoint with the best validation
fill-in-the-blank accuracy is kept, and the test split is never touched here.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch

from ml.data import polyvore as pv
from ml.models.compat import TypeAwareCompat, save_checkpoint
from ml.scorers import ModelScorer, compat_auc, fitb_accuracy, outfit_scores


def build_pairs(store, outfits):
    """All within-outfit pairs of embedded, typed items -> arrays of rows (a, p)."""
    a, p = [], []
    for o in outfits:
        rows = [store.row(i) for i in o["items"]]
        rows = [r for r in rows if r is not None and store.types[r] >= 0]
        for x, y in itertools.combinations(rows, 2):
            a.append(x)
            p.append(y)
    return np.array(a), np.array(p)


def train(store, train_outfits, val_fitb=None, val_compat=None, epochs=25, batch=512, lr=1e-3, margin=0.3,
          out_dim=128, typed=True, seed=0, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, P = build_pairs(store, train_outfits)
    if len(A) == 0:
        raise ValueError("no training pairs: embed more train items first")
    train_rows = np.unique(np.concatenate([A, P]))
    pools = {t: train_rows[store.types[train_rows] == t] for t in range(5)}
    emb = torch.as_tensor(store.emb, device=dev)
    types = torch.as_tensor(store.types, device=dev, dtype=torch.long)
    model = TypeAwareCompat(out_dim=out_dim, typed=typed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, history = -1.0, None, []
    log(f"{len(A)} positive pairs, {len(train_rows)} train items, device={dev}")
    for ep in range(epochs):
        model.train()
        flip = rng.random(len(A)) < 0.5
        a, p = np.where(flip, P, A), np.where(flip, A, P)
        neg = np.array([rng.choice(pools[store.types[x]]) for x in p])
        order = rng.permutation(len(a))
        total = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            ia, ip, ineg = (torch.as_tensor(v[idx], device=dev) for v in (a, p, neg))
            pos_s = model.score_pairs(emb[ia], types[ia], emb[ip], types[ip])
            neg_s = model.score_pairs(emb[ia], types[ia], emb[ineg], types[ineg])
            loss = torch.relu(margin - pos_s + neg_s).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        scorer = ModelScorer(model)
        if val_fitb:
            val_acc = fitb_accuracy(scorer, store, val_fitb)[0]
        else:  # fall back to triplet accuracy on the training pairs' held-out negatives
            val_acc = float((model.score_numpy(store.emb[a[:2000]], store.types[a[:2000]], store.emb[p[:2000]], store.types[p[:2000]])
                             > model.score_numpy(store.emb[a[:2000]], store.types[a[:2000]], store.emb[neg[:2000]], store.types[neg[:2000]])).mean())
        history.append({"epoch": ep + 1, "loss": total / len(a), "val": val_acc})
        log(f"epoch {ep + 1:02d} loss {total / len(a):.4f} val {val_acc:.4f}")
        if not np.isnan(val_acc) and val_acc > best:
            best, best_state = val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), {"best_val": best, "history": history, "n_pairs": int(len(A)), "typed": typed, "out_dim": out_dim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with polyvore_item_metadata.json and disjoint/")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", default="ml/checkpoints/compat.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--shared-head", action="store_true", help="ablation: one head instead of one per type pair")
    a = ap.parse_args()

    d = os.path.join(a.data_dir, "disjoint")
    store = pv.load_cache(a.cache_dir, pv.load_item_types(os.path.join(a.data_dir, "polyvore_item_metadata.json")))
    train_outfits = pv.load_outfits(os.path.join(d, "train.json"))
    vres = pv.build_resolver(os.path.join(d, "valid.json"))
    val_fitb = pv.load_fitb(os.path.join(d, "fill_in_blank_valid.json"), vres)
    val_compat = pv.load_compat(os.path.join(d, "compatibility_valid.txt"), vres)

    model, info = train(store, train_outfits, val_fitb, val_compat, a.epochs, a.batch, a.lr, a.margin, a.out_dim, not a.shared_head)
    scorer = ModelScorer(model)
    # 75th percentile of real validation-outfit scores: used by wardrobe-gap analysis as a "good outfit" threshold
    real = [[store.row(i) for i in o["items"] if store.row(i) is not None and store.types[store.row(i)] >= 0]
            for o in pv.load_outfits(os.path.join(d, "valid.json"))]
    real = [r for r in real if len(r) >= 2]
    thr = float(np.percentile(outfit_scores(scorer, store, real), 75)) if real else None
    info["threshold_p75"] = thr
    info["val_auc"] = compat_auc(scorer, store, val_compat)[0]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_checkpoint(model, a.out, info)
    print(f"saved {a.out}  best val FITB {info['best_val']:.4f}  val AUC {info['val_auc']:.4f}  p75 threshold {thr}")


if __name__ == "__main__":
    main()
