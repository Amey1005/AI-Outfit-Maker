"""Redo background removal for items already in the wardrobe: new colors and a cleaner FashionCLIP embedding.

    python scripts/recolor.py            # show what would change
    python scripts/recolor.py --apply    # save the changes

Use it after turning background removal on (USE_REMBG=1, the default), for photos uploaded while it was off.
Colors you picked by hand on the Wardrobe page are kept; types are never changed. Restart the API afterwards.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Item, SessionLocal  # noqa: E402
from app.services import imaging  # noqa: E402
from app.services.embedding import get_backend  # noqa: E402
from ml.color import dominant_colors, nearest_named_color  # noqa: E402

if not settings.use_rembg:
    sys.exit("Background removal is off (USE_REMBG=0). Open a new window, or set $env:USE_REMBG=\"1\", and run again.")

apply = "--apply" in sys.argv
backend = get_backend()
db = SessionLocal()
changed = 0
for it in db.scalars(select(Item).order_by(Item.id)):
    path = settings.media_dir / imaging.original_name(it.image_path)
    if not path.exists():
        print(f"item {it.id}: photo missing, skipped")
        continue
    clean, fg = imaging.preprocess(imaging.load_image(path.read_bytes()))
    if imaging._rembg_failed:
        sys.exit("rembg could not run - check that it is installed (pip install \"rembg[cpu]\").")
    new = [{"name": nearest_named_color(rgb), "rgb": list(rgb), "fraction": round(frac, 3)}
           for rgb, frac in dominant_colors(fg, k=3, drop_background=False)]
    manual = [c for c in it.colors if c.get("manual")]
    if manual:  # keep the hand-picked main color, refresh the rest
        new = manual + [c for c in new if c["name"] != manual[0]["name"]][:2]
    before, after = [c["name"] for c in it.colors], [c["name"] for c in new]
    if before != after:
        changed += 1
    print(f"item {it.id:3d} {it.fine_type:12s} {', '.join(before):28s} -> {', '.join(after)}")
    if apply:
        it.colors = new
        it.embedding = backend.embed(clean, fg).tobytes()
if apply:
    db.commit()
print(f"\n{changed} item(s) get different colors; embeddings {'were refreshed for all items' if apply else 'would be refreshed for all items'}"
      + ("" if apply else ". Run with --apply to save."))
