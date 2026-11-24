import re
from collections import OrderedDict

import numpy as np
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db import Item, User, get_db
from app.services.recommender import WItem

_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def current_user(x_user_id: str = Header(...), db: Session = Depends(get_db)) -> User:
    """Simple identity: the browser generates a UUID once and sends it as X-User-Id. No passwords by design."""
    if not _ID_RE.match(x_user_id):
        raise HTTPException(400, "X-User-Id must be 8-64 letters, digits or dashes")
    user = db.get(User, x_user_id)
    if user is None:
        user = User(id=x_user_id)
        db.add(user)
        db.commit()
    return user


def to_witem(it: Item) -> WItem:
    top = it.colors[0] if it.colors else {"rgb": [128, 128, 128], "name": "grey"}
    return WItem(it.id, it.fine_type, it.coarse_type, it.material, tuple(top["rgb"]),
                 np.frombuffer(it.embedding, dtype=np.float32), it.pattern, top["name"], it.wear or "unisex")


def item_json(it: Item) -> dict:
    return {"id": it.id, "image_url": f"/media/{it.image_path}", "type": it.fine_type, "coarse_type": it.coarse_type,
            "colors": it.colors, "pattern": it.pattern, "material": it.material,
            "type_confidence": round(it.type_confidence, 3), "wear": it.wear or "unisex"}


class _LRU(OrderedDict):
    def __init__(self, size=64):
        super().__init__()
        self.size = size

    def put(self, k, v):
        self[k] = v
        if len(self) > self.size:
            self.popitem(last=False)


# Pairwise score matrices are expensive-ish and only change when the wardrobe does, so cache by item-id set.
matrix_cache = _LRU()


def pair_matrix(request, user_id: str, witems):
    key = (user_id, request.app.state.scorer.name, tuple((i.id, i.coarse_type, i.rgb) for i in witems))
    if key not in matrix_cache:
        matrix_cache.put(key, request.app.state.scorer.matrix(witems))
    return matrix_cache[key]
