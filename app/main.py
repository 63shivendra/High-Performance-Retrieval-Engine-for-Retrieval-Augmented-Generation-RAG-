"""
FastAPI web server for the SelfIndex live demo.

    uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import MODELS, MODES, ROOT, DemoEngine

STATIC = os.path.join(os.path.dirname(__file__), "static")
engine: DemoEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = DemoEngine()
    print(f"Engine ready in {engine.startup_seconds:.1f}s — {engine.stats()}")
    yield


app = FastAPI(
    title="SelfIndex — search engine built from scratch",
    description="Boolean, TF and TF-IDF inverted indexes with TAAT/DAAT query processing.",
    version="1.0.0",
    lifespan=lifespan,
)


def _check(model: str, mode: str) -> None:
    if model not in MODELS:
        raise HTTPException(400, f"model must be one of {MODELS}")
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=300),
    model: str = "tfidf",
    mode: str = "taat",
    k: int = Query(10, ge=1, le=50),
):
    _check(model, mode)
    return engine.search(q, model, mode, k)


@app.get("/api/compare")
def compare(q: str = Query(..., min_length=1, max_length=300)):
    return {"query": q, "rows": engine.compare(q)}


@app.get("/api/trace")
def trace(q: str = Query(..., min_length=1, max_length=300)):
    """Real postings for the TAAT vs DAAT animation."""
    return engine.trace(q)


@app.get("/api/stats")
def stats():
    return engine.stats()


@app.get("/api/benchmarks")
def benchmarks():
    return engine.benchmarks


@app.get("/healthz")
def healthz():
    return {"ok": engine is not None}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/plots", StaticFiles(directory=os.path.join(ROOT, "plots")), name="plots")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
