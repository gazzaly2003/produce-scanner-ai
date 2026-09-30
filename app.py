"""Ripe & Ready - FastAPI server."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from produce_ai.pipeline import Pipeline
from produce_ai.data import PRODUCE


log = logging.getLogger("ripe-and-ready")

BASE = Path(__file__).parent

# Load local secrets from .env when present.
# .env is ignored by Git and must never be committed.
load_dotenv(BASE / ".env")


def load_vision():
    # ------------------------------------------------------------
    # 1. Prefer Groq vision when an API key is available.
    # ------------------------------------------------------------

    if os.getenv("GROQ_API_KEY"):
        try:
            from produce_ai.vision_groq import GroqVisionEngine

            engine = GroqVisionEngine()

            log.warning(
                "AI vision loaded using Groq: %s",
                getattr(engine, "provider", "groq"),
            )

            return engine

        except Exception as exc:
            log.warning(
                "Groq vision unavailable: %s",
                exc,
            )

    # ------------------------------------------------------------
    # 2. Optional local CLIP fallback.
    # ------------------------------------------------------------

    if os.getenv("USE_CLIP", "1") == "0":
        log.warning(
            "USE_CLIP=0 -> running without local CLIP."
        )
        return None

    try:
        from produce_ai.vision_clip import ClipEngine

        engine = ClipEngine()

        log.warning(
            "Local CLIP vision loaded on %s",
            engine.device,
        )

        return engine

    except Exception as exc:
        log.warning(
            "AI vision unavailable (%s). "
            "Falling back to color analysis only.",
            exc,
        )

        return None


pipeline = Pipeline(load_vision())

app = FastAPI(title="Ripe & Ready")

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static",
)


@app.get("/")
def index():
    return FileResponse(
        BASE / "static" / "index.html"
    )


@app.get("/api/health")
def health():
    engine = pipeline.clip

    return {
        "ai_vision": engine is not None,
        "vision_provider": (
            getattr(engine, "provider", "none")
            if engine
            else "none"
        ),
        "produce_count": len(PRODUCE),
    }


@app.get("/api/produce")
def produce_list():
    return [
        dict(
            key=k,
            name=v["name"],
            category=v["category"],
        )
        for k, v in PRODUCE.items()
    ]


@app.post("/api/scan")
async def scan(
    file: UploadFile = File(...),
    produce: str = Form("auto"),
):
    data = await file.read()

    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(
            413,
            "Image is too large (max 12 MB).",
        )

    try:
        return pipeline.scan(data, produce)

    except ValueError as exc:
        raise HTTPException(
            400,
            str(exc),
        )