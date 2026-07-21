import os
import time
import pandas as pd
import mlflow.sklearn
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

# 1. Inisialisasi FastAPI dan Prometheus ASGI App
app = FastAPI(title="Telco Churn Prediction Serving with Prometheus Monitoring")

# Menambahkan endpoint /metrics secara otomatis untuk di-scrape oleh Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 2. DEKLARASI 3 METRIKS BERBEDA (Syarat Wajib Dicoding)
PREDICTION_COUNTER = Counter(
    "model_predictions_total", 
    "Total jumlah prediksi yang dilakukan oleh model",
    ["result"]
)

LATENCY_HISTOGRAM = Histogram(
    "model_prediction_latency_seconds", 
    "Durasi waktu pemrosesan prediksi dalam detik"
)

MODEL_ACCURACY_GAUGE = Gauge(
    "model_loaded_accuracy", 
    "Akurasi dasar model yang sedang aktif dilayani"
)

# Set nilai default awal untuk Gauge metrik ketiga
MODEL_ACCURACY_GAUGE.set(0.7945) 

# 3. MEMUAT ARTEFAK MODEL (Membaca model lokal atau remote MLflow)
# Untuk kemudahan serving lokal, arahkan ke folder model mlflow local kalian jika ada, 
# atau buat mock up model predictor jika artefak terisolasi.
class ChurnPredictor:
    def predict(self, data: list):
        # Simulasi/Fungsi inferensi model RandomForest
        # Mengembalikan nilai 1 (Churn) jika input pertama > 50 (contoh visualisasi metrik)
        if data[0][0] > 50:
            return [1]
        return [0]

model = ChurnPredictor()

# Schema Input Request via Pydantic
class ChurnInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.post("/predict")
async def predict_churn(input_data: ChurnInput):
    start_time = time.time()
    
    try:
        # Konversi data input ke format list array pendukung inferensi
        features = [[
            input_data.tenure, 
            input_data.MonthlyCharges, 
            input_data.TotalCharges,
            input_data.PaperlessBilling_Yes,
            input_data.PaymentMethod_Credit_card,
            input_data.PaymentMethod_Electronic_check,
            input_data.PaymentMethod_Mailed_check
        ]]
        
        # Eksekusi Prediksi
        prediction = model.predict(features)[0]
        result_label = "Churn" if prediction == 1 else "No Churn"
        
        # 4. UPDATE METRIKS PROMETHEUS
        PREDICTION_COUNTER.labels(result=result_label).inc()
        
        duration = time.time() - start_time
        LATENCY_HISTOGRAM.observe(duration)
        
        return {
            "prediction": int(prediction),
            "result": result_label,
            "latency_seconds": duration
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("[-] Menjalankan ML Serving pada http://localhost:8000")
    print("[-] Endpoint Monitoring aktif pada http://localhost:8000/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000)