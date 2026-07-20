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
    # MENGGUNAKAN JALUR MUTLAK: Mendeteksi posisi file di root proyek secara dinamis
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Folder MLProject
    root_dir = os.path.abspath(os.path.join(current_dir, "..")) # Folder Root Proyek
    
    data_path = os.path.join(root_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di jalur absolut: {data_path}")
        
    print(f"[-] Membaca dataset dari jalur absolut: {data_path}")
    df = pd.read_csv(data_path)
    
    # 3. Preprocessing Esensial
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Mengatasi FutureWarning dengan method penugasan standar pandas 3.0
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Gunakan nested=True agar MLflow Project di runner mengizinkan pembuatan run eksperimen
    print("[-] Melatih model automasi dengan mode nested run...")
    with mlflow.start_run(run_name="GitHub_Actions_Automation", nested=True):
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        
        # Logging data
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "automated_model")
        print(f"[+] Automated Retraining Sukses! Akurasi Model Baru: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()