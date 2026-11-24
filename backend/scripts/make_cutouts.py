"""Create background-removed photos for items uploaded before the app saved them, and show those instead.

    python scripts/make_cutouts.py

Originals are kept next to the new `<name>_cut.jpg` files. Safe to run again: items that already have a cut-out
are skipped.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Item, SessionLocal  # noqa: E402
from app.services import imaging  # noqa: E402

if not settings.use_rembg:
    sys.exit("Background removal is off (USE_REMBG=0). Open a new window, or set $env:USE_REMBG=\"1\", and run again.")

db = SessionLocal()
done = skipped = 0
for it in db.scalars(select(Item).order_by(Item.id)):
    if it.image_path.endswith("_cut.jpg"):
        skipped += 1
        continue
    src = settings.media_dir / it.image_path
    if not src.exists():
        print(f"item {it.id}: photo missing, skipped")
        continue
    _, _, display = imaging.preprocess(imaging.load_image(src.read_bytes()), imaging.DISPLAY_SIZE)
    if imaging._rembg_failed:
        sys.exit("rembg could not run - check that it is installed (pip install \"rembg[cpu]\").")
    folder, name = it.image_path.rsplit("/", 1)
    display.save(settings.media_dir / folder / imaging.cutout_name(name), quality=90)
    it.image_path = f"{folder}/{imaging.cutout_name(name)}"
    db.commit()
    done += 1
    print(f"item {it.id}: done")
print(f"\n{done} cut-out(s) made, {skipped} already had one")
