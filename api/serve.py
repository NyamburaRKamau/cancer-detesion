from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Breast Cancer Detection API",
    description="Predicts malignant vs benign tumors from cell nuclei measurements",
    version="1.0.0"
)

with open("models/classifier.pkl", "rb") as f:
    clf = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

class TumorMeasurements(BaseModel):
    features: List[float]  # 30 cell nuclei measurements

@app.get("/")
def root():
    return {
        "status":   "healthy",
        "model":    "Breast Cancer Detector v1.0",
        "accuracy": "98%",
        "roc_auc":  "0.9954"
    }

@app.get("/features")
def get_features():
    return {"features": feature_names}

@app.post("/predict")
def predict(tumor: TumorMeasurements):
    X      = np.array(tumor.features).reshape(1, -1)
    X_sc   = scaler.transform(X)
    pred   = clf.predict(X_sc)[0]
    probs  = clf.predict_proba(X_sc)[0]

    return {
        "prediction":    "Benign" if pred == 1 else "Malignant",
        "confidence":    round(float(probs.max()), 4),
        "malignant_prob": round(float(probs[0]), 4),
        "benign_prob":    round(float(probs[1]), 4),
        "urgent":         bool(pred == 0)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
