import pickle
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Financial Sentiment API", version="1.0")

MODEL_PATH = Path(__file__).parent / "model.pkl"

_model = None
_tfidf = None


def _load_model():
    global _model, _tfidf
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    _model = bundle["model"]
    _tfidf = bundle["tfidf"]


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+\.?\d*", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.on_event("startup")
def startup():
    _load_model()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _tfidf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    cleaned = _clean_text(req.text)
    features = _tfidf.transform([cleaned])
    sentiment = _model.predict(features)[0]
    probs = _model.predict_proba(features)[0]
    prob_dict = {label: round(float(p), 4) for label, p in zip(_model.classes_, probs)}

    return PredictResponse(sentiment=sentiment, probabilities=prob_dict)
