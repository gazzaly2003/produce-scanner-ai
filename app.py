"""Ripe & Ready - FastAPI server.   Run:  uvicorn app:app --reload"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from produce_ai.pipeline import Pipeline
from produce_ai.data import PRODUCE

log = logging.getLogger("ripe-and-ready")
BASE = Path(__file__).parent


def load_clip():
    if os.getenv("USE_CLIP", "1") == "0":
        log.warning("USE_CLIP=0 -> running with color analysis only.")
        return None
    try:
        from produce_ai.vision_clip import ClipEngine
        engine = ClipEngine()
        log.warning("AI vision (CLIP) loaded on %s%s", engine.device, " + trained probe" if engine.probe else "")
        return engine
    except Exception as e:  # torch/transformers missing or model download failed
        log.warning("AI vision unavailable (%s). Falling back to color analysis only.", e)
        return None


pipeline = Pipeline(load_clip())
app = FastAPI(title="Ripe & Ready")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ai_vision": pipeline.clip is not None, "produce_count": len(PRODUCE)}


@app.get("/api/produce")
def produce_list():
    return [dict(key=k, name=v["name"], category=v["category"]) for k, v in PRODUCE.items()]


@app.post("/api/scan")
async def scan(file: UploadFile = File(...), produce: str = Form("auto")):
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image is too large (max 12 MB).")
    try:
        return pipeline.scan(data, produce)
    except ValueError as e:
        raise HTTPException(400, str(e))
