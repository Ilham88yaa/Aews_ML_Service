from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="AEWS AI Engine API")

# Konfigurasi CORS agar Next.js (port 3000) bisa ngobrol dengan FastAPI (port 8001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti "http://localhost:3000" jika ingin lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model Random Forest yang baru (5 Variabel)
try:
    model = joblib.load('random_forest_model_ats.joblib')
    print("✅ Model Machine Learning berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")

# 2. Definisikan format data yang akan diterima dari Next.js
class StudentData(BaseModel):
    ipk: float
    attendanceRate: float
    assignmentScore: float
    quizScore: float
    atsScore: float

@app.get("/")
def read_root():
    return {"message": "AEWS Machine Learning Engine is Running!"}

# 3. Endpoint Prediksi
@app.post("/predict")
def predict_risk(data: StudentData):
    try:
        # Urutan array HARUS SAMA PERSIS dengan saat training di Colab:
        # ['IPK', 'Kehadiran (%)', 'Nilai Tugas', 'Kuis', 'Nilai ATS']
        input_data = np.array([[
            data.ipk, 
            data.attendanceRate, 
            data.assignmentScore, 
            data.quizScore, 
            data.atsScore
        ]])

        # Melakukan prediksi status (0 = Aman, 1 = Berisiko)
        prediction = model.predict(input_data)[0]
        
        # Mengambil persentase probabilitas berisiko (indeks 1)
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = float(probabilities[1]) * 100 
        
        # Logika Status Risiko (Terserah lu mau diset di angka berapa, ini contoh ideal)
        if risk_probability >= 60:
            risk_status = "HIGH RISK"
            recommendation = "Mahasiswa memiliki probabilitas tinggi untuk gagal. Segera lakukan pemanggilan untuk konseling akademik."
        elif risk_probability >= 30:
            risk_status = "MEDIUM RISK"
            recommendation = "Mahasiswa mulai menunjukkan tanda penurunan akademik. Berikan teguran lisan atau pesan motivasi."
        else:
            risk_status = "SAFE"
            recommendation = "Performa akademik mahasiswa stabil dan aman. Lanjutkan pemantauan rutin."

        return {
            "predictedScore": round(risk_probability, 1),
            "riskStatus": risk_status,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))