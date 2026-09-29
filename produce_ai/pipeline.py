"""Fuses AI vision + color analysis into one verdict, buy/skip decision and nutrition facts."""
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
from . import vision_cv
from .data import PRODUCE, STAGES, DAILY_VALUES, VITAMIN_KEYS

SCORE = {"ripe": 100, "unripe": 40, "overripe": 55, "rotten": -50, "danger": -100, "unknown": 30}


def _dist(pos, sigma=0.45):
    w = np.exp(-((np.arange(4) - pos) ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def _decision(stage, item):
    if stage == "ripe":     return dict(tone="buy", icon="✅", label="Choose this one")
    if stage == "unripe":
        if item["ripens"]:  return dict(tone="wait", icon="⏳", label="Good to buy - just needs a few days")
        return dict(tone="soon", icon="⚠️", label="Better skip - it won't ripen further")
    if stage == "overripe":
        return dict(tone="soon", icon="⚠️", label="Buy only if using today" if item["ripens"] else "Pass, or use it today")
    if stage == "rotten":   return dict(tone="skip", icon="❌", label="Put this one back")
    if stage == "danger":   return dict(tone="skip", icon="⛔", label="Don't buy - safety risk")
    return dict(tone="wait", icon="🤔", label="Check by hand (see picking tips)")


def _message(stage, item):
    n = item["name"].lower()
    m = {
        "unripe": (f"Not ripe yet. {item['name']} keeps ripening after picking - leave it at room temperature for a few days."
                   if item["ripens"] else f"Under-ripe, and {n} won't get sweeter after picking - choose a riper one."),
        "ripe": f"At its best - good to eat now." if item["ripens"] else "Fresh and in good shape.",
        "overripe": (f"Very ripe - eat it today or use it in baking/smoothies." if item["ripens"]
                     else "Past its peak - use soon and trim any bad parts."),
        "rotten": "Signs of rot or mold - don't buy or eat it.",
        "danger": "Green-tinted skin means solanine, a natural toxin. This is a safety issue, not ripeness - discard it if the green runs deep.",
        "unknown": "Couldn't judge the condition from the picture.",
    }[stage]
    return (m + " " + item["note"]).strip() if item["note"] and stage in ("ripe", "unknown", "unripe", "overripe") else m


# Two vocabularies: produce that keeps *ripening* after picking uses ripeness language;
# everything else (most vegetables, and fruit that only freshens/spoils) uses freshness language.
RIPEN_TITLES = {"unripe": "Not ripe yet", "ripe": "Prime pick", "overripe": "Very ripe", "rotten": "Spoiled",
                "danger": "Don't eat this", "unknown": "Inconclusive"}
FRESH_TITLES = {"unripe": "Underripe", "ripe": "Fresh", "overripe": "Aging — use soon", "rotten": "Spoiled",
                "danger": "Don't eat this", "unknown": "Inconclusive"}
RIPENING_TYPES = {"ripener", "avocado"}

def _title(stage, item):
    vocab = RIPEN_TITLES if item["cv"]["type"] in RIPENING_TYPES else FRESH_TITLES
    return vocab[stage]


def nutrition(item):
    n = item["n"]
    nutrients = [dict(key=k, name=DAILY_VALUES[k][0], unit=DAILY_VALUES[k][1], amount=n[k], dv=DAILY_VALUES[k][2],
                      group="vitamin" if k in VITAMIN_KEYS else "mineral")
                 for k in DAILY_VALUES if k in n]
    return dict(serving_g=item["serving_g"], serving_label=item["serving_label"], basis="per 100 g raw edible portion (USDA FoodData Central)",
                per_100g=dict(calories=n["kcal"], carbs_g=n["carbs"], fiber_g=n["fiber"], sugar_g=n["sugar"], protein_g=n["protein"], fat_g=n["fat"]),
                nutrients=nutrients)


class Pipeline:
    def __init__(self, clip=None):
        self.clip = clip

    def scan(self, image_bytes, produce="auto"):
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        except Exception:
            raise ValueError("That file couldn't be read as an image.")
        img.thumbnail((1024, 1024))

        feat = self.clip.image_features(img) if self.clip else None

        # 1) identify
        source, id_conf, alts = "user", None, []
        if produce == "auto":
            if not self.clip:
                return dict(status="needs_selection", message="AI identification isn't installed on this server, so pick the produce type manually.")
            r = self.clip.identify(feat)
            if not r["is_produce"]:
                return dict(status="not_produce", message="That doesn't look like a fruit or vegetable. Try a closer, well-lit photo of the item.")
            key, id_conf = r["ranked"][0]
            alts = [dict(key=k, name=PRODUCE[k]["name"], prob=round(p, 3)) for k, p in r["ranked"][1:]]
            source = "ai"
            low_confidence = id_conf < 0.40   # e.g. an item outside the ~50-item catalog: best guess only
        elif produce in PRODUCE:
            key = produce
        else:
            raise ValueError("Unknown produce type.")
        item = PRODUCE[key]
        if produce != "auto":
            low_confidence = False

        # 2) condition: color/spot analysis + deep model
        pct = vision_cv.analyze(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        a = vision_cv.assess(item["cv"], pct)
        cv_p = _dist(a["pos"]) if a["pos"] is not None else None
        ai_p = self.clip.stage_probs(feat, key) if self.clip else None

        if cv_p is not None and ai_p is not None: fused = .5 * cv_p + .5 * ai_p
        else: fused = cv_p if cv_p is not None else ai_p

        if fused is not None and not a["judgeable"]:
            # skin can't reveal ripeness (watermelon, kiwi): only trust visible rot
            fused = fused if (STAGES[int(np.argmax(fused))] == "rotten" and fused[3] > .6) else None

        if fused is None:
            stage, conf, agree = "unknown", 40, None
            probs = {s: 0.0 for s in STAGES}
        else:
            stage = STAGES[int(np.argmax(fused))]
            if a["pos"] == 3 and stage != "rotten" and fused[3] > .3:      # safety veto: visible rot wins
                stage = "rotten"
            agree = None
            if cv_p is not None and ai_p is not None:
                agree = int(np.argmax(cv_p)) == int(np.argmax(ai_p))
            top = float(fused[STAGES.index(stage)])
            conf = 100 * top + (8 if agree else 0)
            conf = min(conf, 97 if agree else (70 if agree is False else 78))
            conf = int(round(max(40, conf)))
            probs = {s: round(float(v), 3) for s, v in zip(STAGES, fused)}
        if a["danger"]:
            stage, conf = "danger", max(conf, 82)

        fresh, spot = int(round(min(100, a["fresh"]))), int(round(min(100, a["spot"])))
        return dict(
            status="ok",
            produce=dict(key=key, name=item["name"], category=item["category"], source=source,
                         confidence=round(id_conf, 3) if id_conf is not None else None, alternatives=alts,
                         low_confidence=low_confidence),
            condition=dict(stage=stage, title=_title(stage, item), message=_message(stage, item), confidence=conf,
                           probabilities=probs, fresh_pct=fresh, spot_pct=spot, spot_blobs=pct["spot_blobs"],
                           signals=dict(ai_vision=STAGES[int(np.argmax(ai_p))] if ai_p is not None else None,
                                        color_analysis=STAGES[int(np.argmax(cv_p))] if cv_p is not None else None, agree=agree)),
            decision=_decision(stage, item),
            score=round(SCORE[stage] + (fresh - spot) * 0.3, 1),
            nutrition=nutrition(item),
            tips=item["tips"],
            engine=dict(ai_vision=self.clip is not None),
        )
