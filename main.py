from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AEWS Machine Learning Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LOAD MODEL RANDOM FOREST ASLI DARI COLAB (.joblib)
try:
    rf_model = joblib.load('random_forest_model.joblib')
    print("✅ Model Random Forest ASLI berhasil dimuat!")
except Exception as e:
    print("❌ Gagal memuat model:", e)

# PERBAIKAN: Ubah discussionHours menjadi discussionScore agar sinkron dengan Frontend
class StudentMetrics(BaseModel):
    attendanceRate: float
    assignmentScore: float
    discussionScore: float 

@app.get("/")
def read_root():
    return {"message": "AEWS Machine Learning Engine is Online"}

@app.post("/predict")
def predict_student_risk(data: StudentMetrics):
    input_df = pd.DataFrame([{
        'Kehadiran (%)': data.attendanceRate,
        'Nilai Tugas (rata-rata)': data.assignmentScore,
        # PERBAIKAN: Panggil data.discussionScore di sini
        'Partisipasi Diskusi (skor)': data.discussionScore 
    }])
    
    # Ambil probabilitas kelas [Peluang Aman (0), Peluang Berisiko (1)]
    probabilities = rf_model.predict_proba(input_df)[0]
    print("Array Probabilitas:", probabilities)
    
    # Ambil persentase risiko dari indeks ke-1 (Kelas Berisiko)
    final_score = float(probabilities[1]) * 100
    final_score = round(final_score, 2)
    
    # Tentukan status berdasarkan besar kecilnya skor risiko
    if final_score >= 60:
        risk_status = "HIGH RISK"
        recommendation = "PERINGATAN DINI AI: Risiko kegagalan tinggi. Butuh intervensi segera!"
    elif final_score >= 30:
        risk_status = "MEDIUM RISK"
        recommendation = "AI mendeteksi potensi penurunan performa akademik."
    else:
        risk_status = "SAFE"
        recommendation = "Performa mahasiswa stabil dan aman."

    return {
        "predictedScore": final_score,
        "riskStatus": risk_status,
        "recommendation": recommendation
    }