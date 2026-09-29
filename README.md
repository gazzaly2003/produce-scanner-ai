# Ripe & Ready — AI Produce Scanner (Python)

Scan any fruit or vegetable → **what it is**, **choose / skip**, **ripeness**, **calories, vitamins & minerals**, and **how to pick a good one**.

## How it works
| Layer | Tech | Job |
|---|---|---|
| Identification | **CLIP** (zero-shot vision-language model, PyTorch + 🤗 Transformers) | Recognises 31 produce types and rejects non-produce photos |
| Condition (AI) | CLIP prompts, or your own trained probe (`train_probe.py`) | unripe / ripe / overripe / rotten probabilities |
| Condition (vision) | **OpenCV** GrabCut segmentation + HSV colour + dark-spot blob detection | Independent second opinion, per-produce rules (bananas ripen after picking, strawberries don't, green potato = toxin…) |
| Fusion | Weighted ensemble + rot-safety veto | If the two signals disagree, confidence drops and the UI tells you |
| Nutrition | USDA FoodData Central values (per 100 g) scaled to your weight; % of FDA Daily Value | Calories, macros, 10 vitamins, 5 minerals |
| Server / UI | FastAPI + single-page web UI | Upload or camera, compare two items, scan history |

## Setup (Python 3.10+)
```bash
cd produce-scanner-ai
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1) PyTorch (CPU build is fine; skip --index-url if you have a GPU and want CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) everything else
pip install -r requirements.txt

# 3) run
uvicorn app:app --reload
```
Open **http://localhost:8000**. The first start downloads the CLIP weights (~600 MB, one time, needs internet).
`localhost` is a secure context, so the **camera button works** (it doesn't on a plain `file://` page).

Run without the AI model (colour analysis only, you pick the produce type): `USE_CLIP=0 uvicorn app:app`
Use a bigger, more accurate CLIP: `CLIP_MODEL=openai/clip-vit-large-patch14 uvicorn app:app`

## API
```bash
curl -F "file=@banana.jpg" -F "produce=auto" http://localhost:8000/api/scan
```
`produce` = `auto` (AI identifies) or a key from `GET /api/produce` (e.g. `banana`).

## Make it more accurate (recommended)
1. Collect photos into `dataset/unripe|ripe|overripe|rotten/` (or use the Kaggle "Fruits fresh and rotten" dataset).
2. `python train_probe.py dataset` — trains in minutes on CPU, saves `models/stage_probe.joblib`.
3. Restart the server; it uses your probe automatically.

## Honest limitations
* No system can give an *exact* answer from a photo: lighting, camera white-balance, variety (green apples, red-fleshed mangoes) and hidden bruises all matter. Treat the score as strong guidance and confirm with touch and smell.
* Ripeness can't be read from the rind of watermelon or kiwi — the app says so instead of guessing.
* Nutrition values are reference averages for raw produce; real values vary by variety, size and ripeness. Not medical advice.
* Adding produce: add one `add(...)` entry in `produce_ai/data.py` — nutrition, tips and colour profile — and it appears everywhere.

## Layout
```
app.py                    FastAPI server
produce_ai/data.py        produce knowledge base (nutrition, profiles, tips)
produce_ai/vision_cv.py   OpenCV engine
produce_ai/vision_clip.py CLIP engine
produce_ai/pipeline.py    fusion, decision, nutrition
static/index.html         web UI
train_probe.py            optional fine-tuning
```
