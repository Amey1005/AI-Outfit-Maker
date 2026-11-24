"""Re-run type tagging on items already in the wardrobe (e.g. after new types such as "track pants" were added).

    python scripts/retag.py            # show what would change
    python scripts/retag.py --apply    # save the changes

Uses the stored image embeddings, so no photo is re-processed. Types you set by hand on the Wardrobe page are
never touched. Restart the API afterwards so its cached scores pick up the new types.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import Item, SessionLocal  # noqa: E402
from app.services.embedding import get_backend  # noqa: E402
from ml.taxonomy import FINE_TO_COARSE  # noqa: E402

apply = "--apply" in sys.argv
backend = get_backend()
db = SessionLocal()
changed = 0
for it in db.scalars(select(Item).order_by(Item.id)):
    if it.type_confidence == 1.0:          # set by hand
        continue
    tags = backend.tag(None, np.frombuffer(it.embedding, dtype=np.float32), 1.0)
    if tags["fine_type"] != it.fine_type:
        changed += 1
        print(f"item {it.id}: {it.fine_type} -> {tags['fine_type']} (confidence {tags['type_confidence']:.2f})")
        if apply:
            it.fine_type, it.coarse_type = tags["fine_type"], FINE_TO_COARSE[tags["fine_type"]]
            it.type_confidence = min(tags["type_confidence"], 0.999)
if apply:
    db.commit()
print(f"{changed} item(s) {'updated' if apply else 'would change; run with --apply to save'}")
