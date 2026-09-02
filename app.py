"""FastAPI endpoint + tiny web UI for the earring recommender.

    uvicorn app:app --reload
    open http://127.0.0.1:8000

Endpoints
    GET  /                       -> demo UI
    GET  /api/products           -> full inventory
    GET  /api/recommend?necklace_id=N01&top=5
    POST /api/recommend          -> multipart form: file=<necklace image>, top=5
    GET  /images/{product_id}    -> raw product image
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.index import IMAGES, load, recommend

app = FastAPI(title="Jewellery Match — earrings for a necklace")
ROOT = Path(__file__).resolve().parent
_ITEMS = None


def items():
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = load()
    return _ITEMS


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/products")
def products():
    return [
        {"id": it.id, "product_type": it.product_type, "image_file": it.image_file}
        for it in items().values()
    ]


@app.get("/images/{product_id}")
def image(product_id: str):
    it = items().get(product_id)
    if not it:
        raise HTTPException(404, "unknown product id")
    return FileResponse(it.path)


@app.get("/api/recommend")
def reco_by_id(necklace_id: str, top: int = 5):
    if necklace_id not in items():
        raise HTTPException(404, "unknown necklace id")
    return recommend(necklace_id, items(), top_k=top)


@app.post("/api/recommend")
async def reco_by_upload(file: UploadFile = File(...), top: int = Form(5)):
    suffix = Path(file.filename or "up.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return recommend(None, items(), top_k=top, query_path=tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
