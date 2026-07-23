from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(
    title="AEWS Machine Learning Service",
    description="Service untuk prediksi risiko akademik menggunakan Random Forest",
    version="1.0.0"
)

MODEL_PATH = "random_forest_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"WARNING: File {MODEL_PATH} tidak ditemukan!")

# KITA HAPUS GENDER DAN AGE DI SINI
class StudentDataInput(BaseModel):
    attendanceRate: float
    assignmentScore: float
    discussionPart: float

@app.post("/predict")
async def predict_risk(data: StudentDataInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model Machine Learning belum diload.")

    try:
        # KITA UBAH ARRAY FEATURES MENJADI 3 ITEM SAJA
        features = np.array([[
            data.attendanceRate, 
            data.assignmentScore, 
            data.discussionPart
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        risk_probability = float(probabilities[1])
        
        if risk_probability >= 0.70:
            risk_level = "Tinggi"
        elif risk_probability >= 0.40:
            risk_level = "Sedang"
        else:
            risk_level = "Rendah"
            
        # KITA UBAH NAMA FITUR MENJADI 3 SAJA
        feature_names = ['attendanceRate', 'assignmentScore', 'discussionPart']
        importances = model.feature_importances_
        
        factors_with_scores = list(zip(feature_names, importances))
        factors_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        dominant_factors = [factors_with_scores[0][0], factors_with_scores[1][0]]

        return {
            "status": "success",
            "prediction": {
                "riskLevel": risk_level,
                "probability": round(risk_probability, 2),
                "dominantFactors": dominant_factors
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AEWS ML Service is running"}