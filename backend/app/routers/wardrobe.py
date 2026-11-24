import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Item, User, get_db
from app.routers.common import current_user, item_json, pair_matrix, to_witem
from app.services.gaps import wardrobe_gaps
from app.services.recommender import matches_for
from app.services.tagging import tag_image
from ml.color import NAMED_COLORS
from ml.taxonomy import FINE_TO_COARSE, STYLE_PREFS, WEAR

router = APIRouter(prefix="/wardrobe", tags=["wardrobe"])


def _items(db: Session, user: User) -> list[Item]:
    return list(db.scalars(select(Item).where(Item.user_id == user.id).order_by(Item.id)))


@router.post("/items", status_code=201)
async def upload_item(request: Request, file: UploadFile = File(...), type: str | None = Form(None),
                      wear: str = Form("unisex"), user: User = Depends(current_user), db: Session = Depends(get_db)):
    if wear not in WEAR:
        raise HTTPException(400, f"wear must be one of {WEAR}")
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "image too large")
    try:
        tags = tag_image(data, type or None, wear)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception:
        raise HTTPException(400, "could not read that file as an image") from None
    folder = settings.media_dir / user.id
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.jpg"
    from app.services.imaging import cutout_name, load_image

    load_image(data).save(folder / name, quality=90)                    # original, kept for re-processing
    tags["display"].save(folder / cutout_name(name), quality=90)        # what the app shows
    item = Item(user_id=user.id, image_path=f"{user.id}/{cutout_name(name)}", fine_type=tags["fine_type"],
                coarse_type=tags["coarse_type"], colors=tags["colors"], pattern=tags["pattern"],
                material=tags["material"], type_confidence=tags["type_confidence"], wear=wear,
                embedding=tags["embedding"].tobytes())
    db.add(item)
    db.commit()
    return item_json(item)


@router.get("/items")
def list_items(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [item_json(i) for i in _items(db, user)]


class ItemUpdate(BaseModel):
    type: str | None = None
    color: str | None = None
    wear: str | None = None


@router.patch("/items/{item_id}")
def update_item(item_id: int, body: ItemUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Correct a wrongly tagged type and/or main color. A manual type is stored with confidence 1.0 so re-tagging
    leaves it alone; a manual color becomes the item's main color (a background color is often picked by mistake)."""
    item = db.get(Item, item_id)
    if item is None or item.user_id != user.id:
        raise HTTPException(404, "item not found")
    if body.type is not None:
        if body.type not in FINE_TO_COARSE:
            raise HTTPException(400, f"unknown type '{body.type}'")
        item.fine_type, item.coarse_type, item.type_confidence = body.type, FINE_TO_COARSE[body.type], 1.0
    if body.wear is not None:
        if body.wear not in WEAR:
            raise HTTPException(400, f"wear must be one of {WEAR}")
        item.wear = body.wear
    if body.color is not None:
        if body.color not in NAMED_COLORS:
            raise HTTPException(400, f"unknown color '{body.color}'")
        main = {"name": body.color, "rgb": list(NAMED_COLORS[body.color]), "fraction": 1.0, "manual": True}
        item.colors = [main] + [c for c in item.colors if c["name"] != body.color and not c.get("manual")]
    db.commit()
    return item_json(item)


@router.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if item is None or item.user_id != user.id:
        raise HTTPException(404, "item not found")
    from app.services.imaging import original_name

    (Path(settings.media_dir) / item.image_path).unlink(missing_ok=True)
    (Path(settings.media_dir) / original_name(item.image_path)).unlink(missing_ok=True)
    db.delete(item)
    db.commit()


@router.get("/items/{item_id}/matches")
def item_matches(item_id: int, request: Request, limit: int = 8, user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    items = _items(db, user)
    idx = next((k for k, i in enumerate(items) if i.id == item_id), None)
    if idx is None:
        raise HTTPException(404, "item not found")
    w = [to_witem(i) for i in items]
    by_id = {i.id: i for i in items}
    return [{**m, "item": item_json(by_id[m["item_id"]])} for m in matches_for(w, pair_matrix(request, user.id, w), idx, limit)]


@router.get("/gaps")
def gaps(request: Request, style: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """`style` defaults to what the wardrobe holds: only menswear -> menswear picks, only womenswear -> womenswear."""
    items = _items(db, user)
    if len(items) < 3:
        raise HTTPException(400, "add at least 3 items first")
    from app.services.embedding import get_backend

    if style is None:
        wears = {i.wear for i in items}
        style = ("menswear" if "menswear" in wears and "womenswear" not in wears
                 else "womenswear" if "womenswear" in wears and "menswear" not in wears else "all")
    if style not in STYLE_PREFS:
        raise HTTPException(422, f"style must be one of {STYLE_PREFS}")
    return wardrobe_gaps([to_witem(i) for i in items], request.app.state.scorer, get_backend(), style=style)
