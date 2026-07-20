import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Otentikasi otomatis menggunakan Environment Variables di GitHub Actions
REPO_OWNER = "trionohidayat3"
REPO_NAME = "my-first-repo"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def run_retraining():
    # MEMPERBAIKI JALUR DATASET: Membaca file lokal yang ada di root repository
    # Jalur ini disesuaikan karena GitHub Actions menjalankan perintah dari root folder
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di jalur: {os.path.abspath(data_path)}")
        
    print(f"[-] Membaca dataset lokal untuk otomatisasi retraining...")
    df = pd.read_csv(data_path)
    
    # Preprocessing esensial kilat agar pipeline berjalan cepat dan aman
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # One-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tracking otomatis ke MLflow DagsHub
    mlflow.set_experiment("Automated_Retraining")
    with mlflow.start_run(run_name="GitHub_Actions_Automation"):
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "automated_model")
        print(f"[+] Automated Retraining Sukses! Akurasi Model Baru: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()