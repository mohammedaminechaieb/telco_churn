"""API FastAPI : /health /models /predict."""
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.training import load_models  # noqa: E402

DEFAULT_MODEL = "Random Forest"

app = FastAPI(title="Telco Churn API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS = load_models()


def default_model() -> Optional[str]:
    if DEFAULT_MODEL in MODELS:
        return DEFAULT_MODEL
    return next(iter(MODELS), None)


class PredictRequest(BaseModel):
    features: Dict[str, Any]
    model: Optional[str] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS), "default_model": default_model()}


@app.get("/models")
def models():
    return {"models": list(MODELS), "default": default_model()}


@app.post("/predict")
def predict(req: PredictRequest):
    if not MODELS:
        raise HTTPException(503, "Aucun modèle chargé : exécutez notebook.ipynb d'abord.")
    name = req.model or default_model()
    if name not in MODELS:
        raise HTTPException(404, f"Modèle inconnu : {name}. Disponibles : {list(MODELS)}")
    pipe = MODELS[name]
    cols = list(pipe.named_steps["prep"].feature_names_in_)
    missing = [c for c in cols if c not in req.features]
    if missing:
        raise HTTPException(422, f"Variables manquantes : {missing}")
    row = {c: req.features[c] for c in cols}
    df = pd.DataFrame([row], columns=cols)
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in cols:
        if c not in ["tenure", "MonthlyCharges", "TotalCharges"]:
            df[c] = df[c].astype(str)
    proba = float(pipe.predict_proba(df)[0, 1])
    return {
        "model": name,
        "probability": round(proba, 4),
        "decision": "Churn" if proba >= req.threshold else "No Churn",
        "threshold": req.threshold,
    }
