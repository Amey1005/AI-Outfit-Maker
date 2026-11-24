"""Upload the synthetic demo wardrobe to a running API for one user id (handy for demos and screenshots).

    python scripts/seed_demo.py <user-id> [http://localhost:8000]

Then in the browser console: localStorage.setItem("outfit-user-id", "<user-id>") and reload.
"""
import io
import sys

import httpx
from make_demo_wardrobe import WARDROBE, draw_item

user = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
with httpx.Client(base_url=base, headers={"X-User-Id": user}, timeout=60) as c:
    for fine, color in WARDROBE:
        buf = io.BytesIO()
        draw_item(fine, color).save(buf, "PNG")
        r = c.post("/wardrobe/items", files={"file": (f"{fine}_{color}.png", buf.getvalue(), "image/png")}, data={"type": fine})
        r.raise_for_status()
print(f"seeded {len(WARDROBE)} items for {user}")
