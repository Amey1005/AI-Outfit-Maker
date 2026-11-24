from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Item, User, get_db
from app.routers.common import current_user, item_json, pair_matrix, to_witem
from app.services import weather as wx
from app.services.embedding import get_backend
from app.services.personalization import to_vec
from app.services.recommender import suggest
from ml.taxonomy import OCCASIONS

router = APIRouter(prefix="/outfits", tags=["outfits"])


class SuggestRequest(BaseModel):
    occasion: str
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    temp_c: float | None = Field(None, description="manual temperature; skips the weather API")
    rain: bool = False
    top_k: int = Field(10, ge=1, le=30)


@router.post("/suggest")
def suggest_outfits(body: SuggestRequest, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.occasion not in OCCASIONS:
        raise HTTPException(422, f"occasion must be one of {OCCASIONS}")
    items = list(db.scalars(select(Item).where(Item.user_id == user.id).order_by(Item.id)))
    if len(items) < 3:
        raise HTTPException(400, "add at least 3 items to your wardrobe first")
    w = [to_witem(i) for i in items]
    if body.temp_c is not None:
        weather = {"temp_c": body.temp_c, "rain": body.rain}
    elif body.lat is not None and body.lon is not None:
        weather = wx.fetch_weather(body.lat, body.lon)
    else:
        weather = None
    res = suggest(w, pair_matrix(request, user.id, w), body.occasion, weather, to_vec(user.pref_vector),
                  get_backend(), body.top_k)
    by_id = {i.id: i for i in items}
    return {
        "scorer": request.app.state.scorer.name,
        "weather": {**weather, "condition": wx.condition(weather)} if weather else None,
        "notes": res["notes"],
        "outfits": [{**r, "items": [item_json(by_id[i]) for i in r["items"]], "item_ids": r["items"]} for r in res["outfits"]],
    }
