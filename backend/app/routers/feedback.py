import shutil

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Feedback, Item, User, WearEvent, get_db
from app.routers.common import current_user, matrix_cache
from app.services.personalization import to_vec, update_pref

router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    item_ids: list[int] = Field(min_length=2)
    liked: bool
    occasion: str | None = None


class WearIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)


def _own_items(db, user, ids):
    items = list(db.scalars(select(Item).where(Item.user_id == user.id, Item.id.in_(ids))))
    if len(items) != len(set(ids)):
        raise HTTPException(404, "one or more items not found")
    return items


@router.post("/feedback", status_code=201)
def post_feedback(body: FeedbackIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    items = _own_items(db, user, body.item_ids)
    embs = np.stack([np.frombuffer(i.embedding, dtype=np.float32) for i in items])
    user.pref_vector = update_pref(to_vec(user.pref_vector), embs, body.liked).tobytes()
    db.add(Feedback(user_id=user.id, item_ids=body.item_ids, liked=body.liked, occasion=body.occasion))
    db.commit()
    return {"ok": True}


@router.post("/wear", status_code=201)
def post_wear(body: WearIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    _own_items(db, user, body.item_ids)
    db.add(WearEvent(user_id=user.id, item_ids=body.item_ids))
    db.commit()
    return {"ok": True}


@router.delete("/users/me/data", status_code=204)
def delete_my_data(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Privacy: removes the user row, every item, feedback and wear event, and the stored photos."""
    shutil.rmtree(settings.media_dir / user.id, ignore_errors=True)
    for m in (Feedback, WearEvent):
        for row in db.scalars(select(m).where(m.user_id == user.id)):
            db.delete(row)
    db.delete(user)
    db.commit()
    for k in [k for k in matrix_cache if k[0] == user.id]:
        del matrix_cache[k]
