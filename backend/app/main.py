import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.routers import feedback, outfits, wardrobe
from app.services.embedding import get_backend
from app.services.recommender import load_scorer

log = logging.getLogger("outfit")
latencies: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are loaded once at startup, never per request.
    init_db()
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    get_backend()
    app.state.scorer = load_scorer()
    log.info("backend=%s scorer=%s", settings.embed_backend, app.state.scorer.name)
    yield


app = FastAPI(title="Outfit Maker", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
settings.media_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t) * 1000
    route = request.scope.get("route")
    latencies[f"{request.method} {route.path if route else request.url.path}"].append(ms)
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "backend": settings.embed_backend}


@app.get("/metrics/latency")
def latency_metrics():
    """p50/p95 request latency in ms per route since the server started."""
    return {k: {"n": len(v), "p50_ms": round(float(np.percentile(v, 50)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
            for k, v in latencies.items()}


app.include_router(wardrobe.router)
app.include_router(outfits.router)
app.include_router(feedback.router)
