"""
Groq-based vision engine for Ripe & Ready.

Uses Groq Cloud vision instead of local CLIP/PyTorch when GROQ_API_KEY
is available.

Expected Pipeline interface:
    image_features()
    identify()
    stage_probs()
"""

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from .data import PRODUCE, STAGES


log = logging.getLogger("ripe-and-ready.groq")

MODEL = "qwen/qwen3.8-27b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env values.
load_dotenv(ENV_FILE, override=True)


CATALOG = "\n".join(
    f"- {key}: {value['name']} ({value['category']})"
    for key, value in PRODUCE.items()
)


class GroqVisionEngine:
    provider = "groq"

    def __init__(self):
        # Re-load environment inside the actual process/worker.
        load_dotenv(ENV_FILE, override=True)

        from groq import Groq, DefaultHttpxClient

        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        if not api_key.startswith("gsk_"):
            raise RuntimeError(
                "GROQ_API_KEY is not a valid Groq key format."
            )

        client_kwargs = {
            "api_key": api_key,
            "timeout": 60.0,
            "max_retries": 2,
        }

        # ========================================================
        # PYTHONANYWHERE OUTBOUND PROXY
        # ========================================================
        #
        # PythonAnywhere free accounts route external HTTP/HTTPS
        # traffic through proxy.server:3128.
        #
        # The web worker provides DOMAIN_SOCKET, so use that to
        # detect the ASGI environment.
        #
        # We also check common PythonAnywhere environment variables
        # in case the platform already provides a proxy URL.
        # ========================================================

        running_on_pythonanywhere = bool(
            os.getenv("DOMAIN_SOCKET")
            or os.getenv("PYTHONANYWHERE_SITE")
        )

        if running_on_pythonanywhere:
            proxy = (
                os.getenv("https_proxy")
                or os.getenv("HTTPS_PROXY")
                or os.getenv("http_proxy")
                or os.getenv("HTTP_PROXY")
                or "http://proxy.server:3128"
            ).strip()

            log.info(
                "PythonAnywhere detected. Configuring Groq "
                "through outbound proxy."
            )

            log.info(
                "Groq proxy configured: %s",
                proxy,
            )

            client_kwargs["http_client"] = DefaultHttpxClient(
                proxy=proxy
            )

        else:
            log.info(
                "Running outside PythonAnywhere. "
                "Using direct Groq connection."
            )

        self.client = Groq(**client_kwargs)

        log.info(
            "Groq vision engine initialized. Model=%s",
            MODEL,
        )

    # ============================================================
    # IMAGE ENCODING
    # ============================================================

    @staticmethod
    def _encode_image(pil: Image.Image) -> str:
        """
        Convert uploaded image into a compact JPEG base64 string.
        """

        if not isinstance(pil, Image.Image):
            raise ValueError(
                "Invalid image supplied to vision engine."
            )

        image = pil.convert("RGB").copy()

        # Keep request size small enough for cloud vision.
        image.thumbnail((1024, 1024))

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=85,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return encoded

    # ============================================================
    # GROQ REQUEST
    # ============================================================

    def _request(
        self,
        prompt: str,
        image_b64: str,
    ) -> dict[str, Any]:
        """
        Send a multimodal request to Groq.
        """

        try:
            completion = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the visual intelligence engine "
                            "for a fruit and vegetable scanner. "
                            "Analyze only visible evidence in the image. "
                            "Do not invent details. "
                            "Follow the requested JSON format exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        + image_b64
                                    )
                                },
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_completion_tokens=500,
                response_format={
                    "type": "json_object"
                },
            )

            if not completion.choices:
                raise RuntimeError(
                    "Groq returned no choices."
                )

            content = completion.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            try:
                result = json.loads(content)
            except json.JSONDecodeError as exc:
                log.error(
                    "Groq returned invalid JSON: %s",
                    content[:500],
                )
                raise RuntimeError(
                    "Groq returned invalid JSON."
                ) from exc

            if not isinstance(result, dict):
                raise RuntimeError(
                    "Groq returned a non-object response."
                )

            return result

        except Exception as exc:
            # Keep the real error in the server log.
            # Never expose the API key to the user.
            log.exception(
                "Groq vision request failed: %s: %s",
                type(exc).__name__,
                str(exc),
            )

            raise ValueError(
                "The AI vision service could not analyze this image. "
                "Please try again with a clear, well-lit photo."
            ) from exc

    # ============================================================
    # IMAGE ANALYSIS
    # ============================================================

    def image_features(self, pil: Image.Image):
        image_b64 = self._encode_image(pil)

        prompt = f"""
Analyze this image for a fruit/vegetable scanner.

Allowed produce catalog:

{CATALOG}

Return JSON only using exactly this structure:

{{
  "is_produce": true,
  "produce_key": "banana",
  "confidence": 0.95,
  "alternatives": [
    {{"key": "plantain", "confidence": 0.03}},
    {{"key": "mango", "confidence": 0.02}}
  ],
  "stage_probs": {{
    "unripe": 0.05,
    "ripe": 0.80,
    "overripe": 0.12,
    "rotten": 0.03
  }}
}}

Rules:

1. "produce_key" MUST be one of the catalog keys above or null.
2. "confidence" must be between 0 and 1.
3. Give at most 3 alternatives.
4. "stage_probs" must contain all four stages.
5. Stage probabilities must be numbers between 0 and 1.
6. Stage probabilities must sum to 1 when produce is detected.
7. Use "unripe" for visibly immature/underripe produce.
8. Use "ripe" for fresh/ready produce.
9. Use "overripe" for aging, past-prime, or very soft produce.
10. Use "rotten" only when visible rot, mold, or severe spoilage exists.
11. Do not call ordinary brown spots on a banana "rotten" unless
    the visible evidence indicates actual spoilage.
12. If the image is not clearly a fruit or vegetable, return:

{{
  "is_produce": false,
  "produce_key": null,
  "confidence": 0,
  "alternatives": [],
  "stage_probs": {{
    "unripe": 0,
    "ripe": 0,
    "overripe": 0,
    "rotten": 0
  }}
}}
"""

        analysis = self._request(
            prompt,
            image_b64,
        )

        return {
            "_image_b64": image_b64,
            "_analysis": analysis,
        }

    # ============================================================
    # PRODUCE IDENTIFICATION
    # ============================================================

    def identify(self, feat):
        analysis = feat["_analysis"]

        is_produce = bool(
            analysis.get("is_produce", False)
        )

        if not is_produce:
            return {
                "is_produce": False,
                "ranked": [],
            }

        key = analysis.get("produce_key")

        if key not in PRODUCE:
            key = self._resolve_key(analysis)

        if key not in PRODUCE:
            return {
                "is_produce": False,
                "ranked": [],
            }

        confidence = self._clamp(
            analysis.get("confidence", 0.0)
        )

        ranked = [
            (key, confidence)
        ]

        alternatives = analysis.get(
            "alternatives",
            [],
        )

        if isinstance(alternatives, list):
            for alt in alternatives:
                if not isinstance(alt, dict):
                    continue

                alt_key = alt.get("key")

                if alt_key not in PRODUCE:
                    continue

                alt_conf = self._clamp(
                    alt.get("confidence", 0.0)
                )

                if alt_key == key:
                    continue

                ranked.append(
                    (alt_key, alt_conf)
                )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            "is_produce": True,
            "ranked": ranked[:3],
        }

    # ============================================================
    # RIPENESS / STAGE ANALYSIS
    # ============================================================

    def stage_probs(self, feat, key):
        analysis = feat["_analysis"]

        identified_key = analysis.get(
            "produce_key"
        )

        # Automatic identification:
        # reuse the probabilities from the same vision request.
        if identified_key == key:
            probs = self._normalise_probs(
                analysis.get("stage_probs")
            )

            if probs is not None:
                return probs

        # Manual produce selection:
        # perform a second vision request.
        image_b64 = feat["_image_b64"]

        item = PRODUCE[key]

        prompt = f"""
The user selected this produce item:

Key: {key}
Name: {item["name"]}
Category: {item["category"]}

Analyze the uploaded image specifically for this produce.

Return JSON only:

{{
  "stage_probs": {{
    "unripe": 0.00,
    "ripe": 0.00,
    "overripe": 0.00,
    "rotten": 0.00
  }}
}}

Rules:

- All four stage keys are required.
- Every value must be between 0 and 1.
- The values must sum to 1.
- Judge only visible evidence.
- "rotten" means actual visible rot, mold, or severe spoilage.
- "overripe" means past-prime, aging, or very soft appearance.
- "ripe" means fresh and ready/in good condition.
- "unripe" means visibly immature when that concept applies.
"""

        result = self._request(
            prompt,
            image_b64,
        )

        probs = self._normalise_probs(
            result.get("stage_probs")
        )

        if probs is None:
            return self._fallback_stage_probs()

        return probs

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _clamp(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, number),
        )

    @staticmethod
    def _normalise_probs(values):
        if not isinstance(values, dict):
            return None

        arr = []

        for stage in STAGES:
            try:
                value = float(
                    values.get(stage, 0.0)
                )
            except (TypeError, ValueError):
                value = 0.0

            arr.append(
                max(0.0, value)
            )

        total = sum(arr)

        if total <= 0:
            return None

        arr = [
            value / total
            for value in arr
        ]

        return np.array(
            arr,
            dtype=float,
        )

    @staticmethod
    def _fallback_stage_probs():
        return np.array(
            [0.25, 0.25, 0.25, 0.25],
            dtype=float,
        )

    @staticmethod
    def _resolve_key(analysis):
        name = str(
            analysis.get(
                "produce_name",
                "",
            )
        ).strip().lower()

        if not name:
            return None

        for key, item in PRODUCE.items():
            if item["name"].strip().lower() == name:
                return key

        return None