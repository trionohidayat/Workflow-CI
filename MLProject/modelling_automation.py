import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Konfigurasi repositori target di DagsHub
REPO_OWNER = "trionohidayat3"
REPO_NAME = "my-first-repo"

print(f"[-] Menginisialisasi DagsHub untuk {REPO_OWNER}/{REPO_NAME}...")
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def run_retraining():
    # 2. Jalur dataset yang fleksibel (lokal & parent directory)
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        data_path = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset tidak ditemukan di jalur lokal maupun parent directory.")
        
    print(f"[-] Membaca dataset dari jalur: {data_path}")
    df = pd.read_csv(data_path)
    
    # 3. Proses Preprocessing Kilat
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Eksperimen Menggunakan Run Aktif dari MLflow Project CLI
    # Menggunakan nested=True atau langsung mendeteksi active run agar tidak crash 'Run not found'
    active_run = mlflow.active_run()
    
    if active_run:
        print(f"[+] Menggunakan Run Aktif dari GitHub Actions Pipeline: {active_run.info.run_id}")
        # Melatih model langsung di dalam active run
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        
        # Log langsung ke run yang aktif saat ini
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "automated_model")
        print(f"[+] Automated Retraining Sukses! Akurasi Model Baru: {acc:.4f}")
    else:
        # Cadangan jika script dijalankan manual secara lokal di luar perintah `mlflow run`
        mlflow.set_experiment("Automated_Retraining")
        with mlflow.start_run(run_name="Local_Manual_Run"):
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            mlflow.log_param("n_estimators", 50)
            mlflow.log_param("max_depth", 5)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "automated_model")
            print(f"[+] Manual Retraining Sukses! Akurasi Model: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()