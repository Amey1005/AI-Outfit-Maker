import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.update(EMBED_BACKEND="lite", USE_REMBG="0", DATABASE_URL=f"sqlite:///{_tmp}/test.db",
                  MEDIA_DIR=f"{_tmp}/media", COMPAT_CHECKPOINT=f"{_tmp}/none.pt")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from make_demo_wardrobe import WARDROBE, draw_item  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def user():
    return {"X-User-Id": str(uuid.uuid4())}


def png(fine, color):
    buf = io.BytesIO()
    draw_item(fine, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def stocked(client, user):
    ids = {}
    for fine, color in WARDROBE:
        r = client.post("/wardrobe/items", headers=user, files={"file": ("x.png", png(fine, color), "image/png")}, data={"type": fine})
        assert r.status_code == 201, r.text
        ids[(fine, color)] = r.json()["id"]
    return user, ids
