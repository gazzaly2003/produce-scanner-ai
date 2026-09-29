"""Classical computer-vision engine (OpenCV + NumPy).

Steps: GrabCut foreground segmentation -> HSV color bucketing inside the fruit only ->
dark/brown spot blob detection -> per-produce ripeness rules -> position on a
0..3 scale (unripe .. ripe .. overripe .. rotten).
"""
import numpy as np
import cv2

BUCKETS = ["green", "coolgreen", "yellow", "orange", "red", "purple", "brown", "dark", "pale"]


def _segment(bgr):
    """Return a 0/1 mask of the produce (falls back to the whole frame)."""
    h, w = bgr.shape[:2]
    full = np.ones((h, w), np.uint8)
    try:
        mask = np.zeros((h, w), np.uint8)
        rect = (int(w * .06), int(h * .06), int(w * .88), int(h * .88))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        frac = fg.mean()
        return fg if 0.10 < frac < 0.98 else full
    except cv2.error:
        return full


def analyze(bgr):
    """bgr: uint8 image. Returns bucket percentages (0-100) + spot statistics."""
    scale = 256 / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = _segment(bgr)
    eroded = cv2.erode(mask, np.ones((7, 7), np.uint8))
    if eroded.mean() > 0.05:          # drop the fringe where fruit blends into background/shadow
        mask = eroded
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2
    s = hsv[..., 1].astype(np.float32) / 255 * 100
    v = hsv[..., 2].astype(np.float32) / 255 * 100

    # priority-ordered rules, identical to the browser prototype
    conds = [(s < 10) & (v > 88), v < 30, (s < 20) & (v < 70),
             (h >= 15) & (h < 50) & (v < 55) & (s > 30),
             (h >= 50) & (h < 68), (h >= 68) & (h < 170), (h >= 170) & (h < 260),
             (h >= 260) & (h < 335), (h >= 335) | (h < 15)]
    names = ["pale", "dark", "pale", "brown", "yellow", "green", "coolgreen", "purple", "red"]
    idx = np.select(conds, list(range(len(names))), default=len(names))
    labels = np.array(names + ["orange"])[idx]

    inside = mask.astype(bool)
    total = max(1, int(inside.sum()))
    pct = {b: 100.0 * float(np.count_nonzero((labels == b) & inside)) / total for b in BUCKETS}
    pct["spot"] = pct["dark"] * .9 + pct["brown"] * .65 + pct["pale"] * .6 + pct["coolgreen"] * .5

    # count distinct dark/brown blobs (bruises, rot pockets)
    spots = (((labels == "dark") | (labels == "brown")) & inside).astype(np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(spots)
    min_area = 0.004 * total
    pct["spot_blobs"] = int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area))
    pct["foreground_pct"] = 100.0 * total / mask.size
    return pct


def _w(p, weights):
    return sum(p[k] * w for k, w in weights.items())


def assess(cfg, p):
    """Apply a produce profile. Returns dict(pos, fresh, spot, danger, judgeable)."""
    t, spot = cfg["type"], p["spot"]
    out = dict(pos=1.0, fresh=max(0.0, 100 - spot), spot=spot, danger=False, judgeable=True)

    if t == "ripener":
        fresh = sum(p[k] for k in cfg["fresh"])
        if spot > cfg["rot_at"]: pos = 3
        elif p["green"] > cfg["unripe_green"]: pos = 0
        elif fresh > cfg["ripe_min"] and p["brown"] < cfg["brown_max"]: pos = 1.3 if p["spot_blobs"] >= 3 else 1
        elif p["green"] > 20: pos = 0.6
        elif fresh > cfg["ripe_min"] * .5: pos = 2
        else: pos = 0.6
        out.update(pos=pos, fresh=fresh)
    elif t == "avocado":
        fresh = p["dark"] + p["green"] * .15
        if p["pale"] > 14 or spot > 46: pos = 3
        elif p["green"] > 42: pos = 0
        elif p["dark"] > 32 and p["brown"] < 20: pos = 1
        else: pos = 0.6
        out.update(pos=pos, fresh=fresh)
    elif t == "freshness":
        fresh = sum(p[k] for k in cfg["fresh"])
        if spot > cfg["rot_at"]: pos = 3
        elif cfg["pale_at"] and fresh < cfg["pale_at"]: pos = 0
        elif spot > cfg["rot_at"] * .6: pos = 2
        else: pos = 1
        out.update(pos=pos, fresh=fresh)
    elif t == "spot":
        sc = _w(p, cfg["weights"]) if cfg["weights"] else spot
        if sc > cfg["rot_at"]: pos = 3
        elif sc > cfg["aging_at"]: pos = 2
        elif cfg["green_at"] and p["green"] > cfg["green_at"]: pos = 2      # e.g. sprouting onion
        elif cfg["min_green"] and p["green"] < cfg["min_green"]: pos = 2    # e.g. dried-out husk
        else: pos = 1
        out.update(pos=pos, spot=sc, fresh=max(0.0, 100 - sc))
    elif t == "yellowing":
        rot = _w(p, cfg["weights"])
        if rot > cfg["rot_at"]: pos = 3
        elif p["yellow"] > cfg["yellow_at"]: pos = 2
        else: pos = 1
        out.update(pos=pos, spot=rot, fresh=max(0.0, 100 - rot - p["yellow"]))
    elif t == "solanine":                      # potato: green skin = toxin, not "unripe"
        if p["green"] > 11:
            out.update(pos=2.5, danger=True, spot=p["green"], fresh=100 - p["green"])
        else:
            out.update(pos=3 if spot > 32 else 1)
    elif t == "unjudgeable":                   # ripeness can't be read from the skin
        rot = _w(p, cfg["weights"])
        out.update(spot=rot, fresh=max(0.0, 100 - rot))
        if rot > cfg["rot_at"]: out["pos"] = 3
        else: out.update(pos=None, judgeable=False)
    return out
