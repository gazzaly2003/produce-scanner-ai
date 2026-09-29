"""Deep-learning engine: OpenAI CLIP (zero-shot vision-language model) via Hugging Face.

  * identify()    - which produce is this?  (any item in the knowledge base, plus a
                    "not produce" rejection class)
  * stage_probs() - unripe / ripe / overripe / rotten probabilities.  Uses your trained
                    probe (models/stage_probe.joblib, see train_probe.py) if present,
                    otherwise zero-shot text prompts.
"""
import os
import numpy as np
from .data import PRODUCE, STAGES

NEGATIVES = ["a person", "an empty room", "a hand", "packaged food in a box", "a pet animal", "a cooked meal on a plate", "a blank wall"]
TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "{} on a table."]
STAGE_PROMPTS = {
    "unripe": ["an unripe green {n}", "a hard underripe {n}", "a pale immature {n}"],
    "ripe": ["a fresh ripe {n}", "a perfectly ripe {n} ready to eat"],
    "overripe": ["an overripe {n} with brown spots", "a very soft old {n}"],
    "rotten": ["a rotten moldy {n}", "a spoiled {n} covered in mold"],
}
FRESH_PROMPTS = {"ripe": ["a fresh crisp {n}", "a fresh healthy {n}"], "overripe": ["a wilted old yellowing {n}", "a dried out aging {n}"]}


class ClipEngine:
    def __init__(self, model_name=None, probe_path="models/stage_probe.joblib"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        name = model_name or os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.scale = float(self.model.logit_scale.exp().item())
        self.probe = None
        if os.path.exists(probe_path):
            import joblib
            self.probe = joblib.load(probe_path)
        self._stage_cache = {}
        self._build_identity_index()

    # ---- helpers ----
    def _t(self, out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def _text(self, prompts):
        with self.torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            f = self._t(self.model.get_text_features(**inputs))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    def _mean_norm(self, prompts):
        m = self._text(prompts).mean(0)
        return m / np.linalg.norm(m)

    def _build_identity_index(self):
        self.keys = list(PRODUCE)
        rows = []
        for k in self.keys:
            p = PRODUCE[k]
            desc = f"a {p['name'].lower()}, a type of {p['category']}"
            rows.append(self._mean_norm([t.format(desc) for t in TEMPLATES]))
        for neg in NEGATIVES:
            rows.append(self._mean_norm([t.format(neg) for t in TEMPLATES]))
        self.id_matrix = np.stack(rows)

    # ---- public API ----
    def image_features(self, pil):
        with self.torch.no_grad():
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            f = self._t(self.model.get_image_features(**inputs))
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy()

    def identify(self, feat):
        logits = self.scale * (self.id_matrix @ feat)
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        n = len(self.keys)
        produce_probs = probs[:n]
        is_produce = produce_probs.sum() > probs[n:].sum()
        norm = produce_probs / max(produce_probs.sum(), 1e-9)
        order = np.argsort(-norm)[:3]
        return dict(is_produce=bool(is_produce), ranked=[(self.keys[i], float(norm[i])) for i in order])

    def stage_probs(self, feat, key):
        """np.array of 4 probabilities aligned with STAGES."""
        if self.probe is not None:
            proba = self.probe["model"].predict_proba(feat[None])[0]
            out = np.zeros(4)
            for cls, pr in zip(self.probe["classes"], proba):
                out[STAGES.index(cls)] = pr
            return out
        p = PRODUCE[key]
        if key not in self._stage_cache:
            n = p["name"].lower()
            prompts = STAGE_PROMPTS if p["ripens"] else {**STAGE_PROMPTS, **FRESH_PROMPTS}
            if not p["has_unripe"]:
                prompts = {s: v for s, v in prompts.items() if s != "unripe"}
            present = [s for s in STAGES if s in prompts]
            mat = np.stack([self._mean_norm([t.format(n=n) for t in prompts[s]]) for s in present])
            self._stage_cache[key] = (present, mat)
        present, mat = self._stage_cache[key]
        logits = self.scale * (mat @ feat)
        pr = np.exp(logits - logits.max()); pr /= pr.sum()
        out = np.zeros(4)
        for s, v in zip(present, pr):
            out[STAGES.index(s)] = v
        return out
