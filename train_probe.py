"""OPTIONAL - make the ripeness model more accurate with your own labelled photos.

Folder layout (any number of images per folder, jpg/png):
    dataset/unripe/*.jpg   dataset/ripe/*.jpg   dataset/overripe/*.jpg   dataset/rotten/*.jpg

(Kaggle "Fruits fresh and rotten for classification" works: put fresh -> ripe, rotten -> rotten.)

    python train_probe.py dataset

Trains a logistic-regression "linear probe" on frozen CLIP image embeddings (fast, no GPU needed,
strong with a few hundred photos per class) and saves models/stage_probe.joblib.
The server picks it up automatically on next start and uses it instead of zero-shot prompts.
"""
import sys
from pathlib import Path
import numpy as np
import joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from produce_ai.data import STAGES
from produce_ai.vision_clip import ClipEngine

root = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
engine = ClipEngine(probe_path="__none__")          # zero-shot engine, used only as a feature extractor
X, y = [], []
for stage in STAGES:
    files = [f for f in (root / stage).glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    print(f"{stage}: {len(files)} images")
    for f in files:
        try:
            img = Image.open(f).convert("RGB"); img.thumbnail((512, 512))
            X.append(engine.image_features(img)); y.append(stage)
        except Exception as e:
            print("skip", f.name, e)
if len(set(y)) < 2:
    sys.exit("Need at least two non-empty stage folders.")
X, y = np.stack(X), np.array(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=2000, C=10).fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte)))
clf.fit(X, y)                                          # refit on everything for the final model
Path("models").mkdir(exist_ok=True)
joblib.dump({"model": clf, "classes": list(clf.classes_)}, "models/stage_probe.joblib")
print("Saved models/stage_probe.joblib - restart the server to use it.")
