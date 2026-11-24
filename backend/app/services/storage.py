"""Where photos live. "disk" (default) keeps files under MEDIA_DIR; "db" keeps them in the database, for hosts whose
disk is wiped on every restart (e.g. a free Hugging Face Space). Paths look like "<user_id>/<name>.jpg" in both."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import ImageBlob


def jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _disk_path(path: str) -> Path | None:
    root = settings.media_dir.resolve()
    p = (root / path).resolve()
    return p if p.is_relative_to(root) else None   # never serve files outside MEDIA_DIR


def save(db: Session, path: str, data: bytes) -> None:
    if settings.media_store == "db":
        db.merge(ImageBlob(path=path, user_id=path.split("/", 1)[0], data=data))
    else:
        p = _disk_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def load(db: Session, path: str) -> bytes | None:
    if settings.media_store == "db":
        blob = db.get(ImageBlob, path)
        return blob.data if blob else None
    p = _disk_path(path)
    return p.read_bytes() if p and p.is_file() else None


def delete(db: Session, path: str) -> None:
    if settings.media_store == "db":
        blob = db.get(ImageBlob, path)
        if blob:
            db.delete(blob)
    else:
        p = _disk_path(path)
        if p:
            p.unlink(missing_ok=True)


def delete_user(db: Session, user_id: str) -> None:
    """Every stored photo of one user (both stores, so switching modes never leaves photos behind)."""
    import shutil

    for blob in db.scalars(select(ImageBlob).where(ImageBlob.user_id == user_id)):
        db.delete(blob)
    p = _disk_path(user_id)
    if p:
        shutil.rmtree(p, ignore_errors=True)
