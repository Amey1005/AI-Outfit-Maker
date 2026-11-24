"""Measure /outfits/suggest latency (p50/p95) in-process against the demo wardrobe.

    EMBED_BACKEND=lite python scripts/bench_latency.py [n_requests]

Uses a throwaway database. Run it with your real backend (EMBED_BACKEND=fashionclip) and real photos for numbers you
can quote; this synthetic run only shows the ranking code is fast.
"""
import io
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{tmp}/bench.db")
os.environ.setdefault("MEDIA_DIR", f"{tmp}/media")
os.environ.setdefault("USE_REMBG", "0")

from fastapi.testclient import TestClient  # noqa: E402
from make_demo_wardrobe import WARDROBE, draw_item  # noqa: E402

from app.main import app  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
with TestClient(app) as c:
    h = {"X-User-Id": "bench-user-0001"}
    for fine, color in WARDROBE:
        buf = io.BytesIO()
        draw_item(fine, color).save(buf, "PNG")
        c.post("/wardrobe/items", headers=h, files={"file": ("x.png", buf.getvalue(), "image/png")}, data={"type": fine})
    ms = []
    for k in range(n):
        t = time.perf_counter()
        r = c.post("/outfits/suggest", headers=h, json={"occasion": ["college", "party", "formal", "gym"][k % 4], "temp_c": 20})
        ms.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200
    print(f"{len(WARDROBE)} items, {n} requests: p50 {np.percentile(ms, 50):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms, "
          f"first (cold, builds score matrix) {ms[0]:.1f} ms; backend={os.environ.get('EMBED_BACKEND')}")
