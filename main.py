from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- Import CORS Middleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="AEWS Machine Learning Service", version="2.0")

# ---------------------------------------------------------
# SETUP CORS (Cross-Origin Resource Sharing)
# Mengizinkan Frontend Next.js (port 3000 / browser) menghubungi FastAPI
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan request dari semua origin (Next.js, NestJS, Postman, dll)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua HTTP Methods (GET, POST, OPTIONS, dll)
    allow_headers=["*"],  # Mengizinkan semua HTTP Headers
)

# 1. Load Kedua Model (.pkl)
# Menggunakan os.path untuk penanganan path file yang lebih fleksibel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_W4_PATH = os.path.join(BASE_DIR, "model_w4 (1).pkl")
MODEL_W8_PATH = os.path.join(BASE_DIR, "model_w8 (1).pkl")

# Cadangan jika file bernama standar (tanpa ' (1)')
if not os.path.exists(MODEL_W4_PATH):
    MODEL_W4_PATH = os.path.join(BASE_DIR, "model_w4.pkl")
if not os.path.exists(MODEL_W8_PATH):
    MODEL_W8_PATH = os.path.join(BASE_DIR, "model_w8.pkl")

try:
    model_w4 = joblib.load(MODEL_W4_PATH)
    model_w8 = joblib.load(MODEL_W8_PATH)
    print("✅ Model W4 dan W8 berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model_w4 = None
    model_w8 = None

# 2. Skema Request (Pydantic)
class PredictRequest(BaseModel):
    weekNumber: int               # 4, 8, 12, dst.
    mataKuliah: str | None = ""   # Konteks Spesifik Matkul
    ipk: float
    attendanceRate: float         # Kehadiran (%)
    assignmentScore: float        # Nilai Tugas
    quizScore: float              # Kuis
    atsScore: float | None = 0.0  # Nilai UTS (Opsional/0 jika Pre-UTS)

@app.post("/predict")
def predict_risk(data: PredictRequest):
    if model_w4 is None or model_w8 is None:
        raise HTTPException(
            status_code=500, 
            detail="Model ML belum dimuat dengan benar. Periksa ketersediaan file .pkl"
        )

    try:
        # LOGIKA DYNAMIC MODEL SELECTION
        if data.weekNumber < 8:
            # FASE 1: PRE-UTS (Minggu 1 - 7) -> Menggunakan model_w4 (4 Variabel)
            features = np.array([[data.ipk, data.attendanceRate, data.assignmentScore, data.quizScore]])
            risk_status = model_w4.predict(features)[0]
            probabilities = model_w4.predict_proba(features)[0]
            classes = list(model_w4.classes_)
            
        else:
            # FASE 2: POST-UTS (Minggu 8 - 16) -> Menggunakan model_w8 (5 Variabel)
            ats_val = data.atsScore if data.atsScore is not None else 0.0
            features = np.array([[data.ipk, data.attendanceRate, data.assignmentScore, data.quizScore, ats_val]])
            risk_status = model_w8.predict(features)[0]
            probabilities = model_w8.predict_proba(features)[0]
            classes = list(model_w8.classes_)

        # KALKULASI PREDICTED SCORE (SKOR RISIKO INFORMATIF)
        # Memetakan probabilitas setiap kelas
        prob_map = dict(zip(classes, probabilities))
        
        # Skor Risiko Gabungan Berbobot (0 - 100%)
        # HIGH RISK dikali bobot 1.0, MEDIUM RISK dikali bobot 0.5, SAFE dikali 0.0
        high_prob = prob_map.get('HIGH RISK', 0.0)
        medium_prob = prob_map.get('MEDIUM RISK', 0.0)
        
        predicted_score = round((high_prob * 1.0 + medium_prob * 0.5) * 100, 1)

        # Generating Dynamic Recommendation
        if risk_status == "HIGH RISK":
            recommendation = f"Mahasiswa berisiko tinggi gagal pada mata kuliah {data.mataKuliah}. Segera lakukan pemanggilan untuk konseling akademik."
        elif risk_status == "MEDIUM RISK":
            recommendation = f"Mahasiswa menunjukkan penurunan performa pada mata kuliah {data.mataKuliah}. Berikan teguran lisan atau motivasi."
        else:
            recommendation = f"Performa mahasiswa pada mata kuliah {data.mataKuliah} tergolong stabil dan aman."

        return {
            "weekNumber": data.weekNumber,
            "mataKuliah": data.mataKuliah,
            "riskStatus": risk_status,
            "predictedScore": predicted_score,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat melakukan prediksi: {str(e)}")