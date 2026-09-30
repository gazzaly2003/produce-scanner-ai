"""
Groq-based vision engine for Ripe & Ready.

This replaces the heavy local CLIP/PyTorch dependency when a GROQ_API_KEY
is available. It keeps the same interface expected by Pipeline:
    image_features()
    identify()
    stage_probs()
"""

import base64
import io
import json
import os
from typing import Any

from PIL import Image

from .data import PRODUCE, STAGES


MODEL = "qwen/qwen3.8-27b"

CATALOG = "\n".join(
    f"- {key}: {value['name']} ({value['category']})"
    for key, value in PRODUCE.items()
)


class GroqVisionEngine:
    provider = "groq"

    def __init__(self):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=api_key)

    # ------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------

    @staticmethod
    def _encode_image(pil: Image.Image) -> str:
        buffer = io.BytesIO()

        image = pil.convert("RGB").copy()
        image.thumbnail((1024, 1024))

        image.save(
            buffer,
            format="JPEG",
            quality=85,
            optimize=True,
        )

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ------------------------------------------------------------
    # Groq request
    # ------------------------------------------------------------

    def _request(self, prompt: str, image_b64: str) -> dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the visual intelligence engine for "
                            "a fruit and vegetable scanner. "
                            "Analyze only what is visible in the image. "
                            "Do not invent details."
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
                response_format={"type": "json_object"},
            )

            content = completion.choices[0].message.content

            if not content:
                raise ValueError("Empty response from vision model.")

            return json.loads(content)

        except Exception as exc:
            raise ValueError(
                "The AI vision service could not analyze this image. "
                "Please try again with a clear, well-lit photo."
            ) from exc

    # ------------------------------------------------------------
    # Generic image analysis
    # ------------------------------------------------------------

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
5. Stage probabilities must be numbers between 0 and 1 and sum to 1.
6. Use "unripe" for immature/underripe produce.
7. Use "ripe" for fresh/ready produce.
8. Use "overripe" for aging, very soft, heavily spotted or past-prime produce.
9. Use "rotten" for visible rot, mold or severe spoilage.
10. If the image is not clearly a fruit or vegetable, set:
    "is_produce": false
    "produce_key": null
    "confidence": 0
    "alternatives": []
    and set all stage probabilities to 0.
"""

        analysis = self._request(prompt, image_b64)

        return {
            "_image_b64": image_b64,
            "_analysis": analysis,
        }

    # ------------------------------------------------------------
    # Produce identification
    # ------------------------------------------------------------

    def identify(self, feat):
        analysis = feat["_analysis"]

        is_produce = bool(analysis.get("is_produce", False))

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

        ranked = [(key, confidence)]

        alternatives = analysis.get("alternatives", [])

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

                if alt_key != key:
                    ranked.append((alt_key, alt_conf))

        ranked = ranked[:3]

        return {
            "is_produce": True,
            "ranked": ranked,
        }

    # ------------------------------------------------------------
    # Stage analysis
    # ------------------------------------------------------------

    def stage_probs(self, feat, key):
        analysis = feat["_analysis"]

        identified_key = analysis.get("produce_key")

        # For automatic identification, use the probabilities returned
        # in the first vision request when they belong to the detected item.
        if identified_key == key:
            probs = self._normalise_probs(
                analysis.get("stage_probs")
            )

            if probs is not None:
                return probs

        # For manually selected produce, ask the vision model specifically
        # about the selected item.
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
- Values must be between 0 and 1.
- Values must sum to 1.
- Judge only visible evidence.
- "rotten" means actual visible rot/mold/severe spoilage.
- "overripe" means past-prime, aging or very soft appearance.
- "ripe" means fresh and ready/in good condition.
- "unripe" means visibly immature when that concept applies.
"""

        result = self._request(prompt, image_b64)

        probs = self._normalise_probs(
            result.get("stage_probs")
        )

        if probs is None:
            return self._fallback_stage_probs()

        return probs

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    @staticmethod
    def _clamp(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalise_probs(values):
        if not isinstance(values, dict):
            return None

        arr = [
            max(0.0, float(values.get(stage, 0.0)))
            for stage in STAGES
        ]

        total = sum(arr)

        if total <= 0:
            return None

        arr = [value / total for value in arr]

        return __import__("numpy").array(arr, dtype=float)

    @staticmethod
    def _fallback_stage_probs():
        return __import__("numpy").array(
            [0.25, 0.25, 0.25, 0.25],
            dtype=float,
        )

    @staticmethod
    def _resolve_key(analysis):
        name = str(analysis.get("produce_name", "")).strip().lower()

        for key, item in PRODUCE.items():
            if item["name"].lower() == name:
                return key

        return None