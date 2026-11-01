from pathlib import Path
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

class Patient(BaseModel):
    age: float = Field(..., ge=29, le=78)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=90, le=200)
    chol: float = Field(..., ge=120, le=430)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=70, le=210)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=6)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=0, le=3)

@app.get("/")
def root():
    return {"message":"Heart Disease Prediction API","docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"ok","model":type(model).__name__}

@app.post("/predict")
def predict(patient: Patient):
    try:
        row=[[getattr(patient,f) for f in FEATURES]]
        pred=int(model.predict(row)[0])
        result={"prediction":pred,"label":"Disease" if pred==1 else "No Disease"}
        if hasattr(model,"predict_proba"):
            result["disease_probability"]=round(float(model.predict_proba(row)[0][1]),4)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
