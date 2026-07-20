import os
import sys
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# PENTING: Memaksa token dari GitHub Secrets masuk ke konfigurasi internal DagsHub
token = os.getenv("DAGSHUB_CLIENT_TOKEN")
if token:
    dagshub.auth.add_app_token(token)
    os.environ["DAGSHUB_CLIENT_TOKEN"] = token

# Konfigurasi repositori DagsHub kalian
REPO_OWNER = "trionohidayat3"
REPO_NAME = "my-first-repo"

print(f"[-] Menghubungkan MLOps ke DagsHub: {REPO_OWNER}/{REPO_NAME}")
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def run_retraining():
    # MEMBACA PARAMETER DINAMIS DARI ARGUMEN CLI (Syarat Reviewer)
    # Default value jika argumen tidak diberikan: n_estimators=50, max_depth=5
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"[+] Menjalankan retraining dengan parameter dinamis -> n_estimators: {n_estimators}, max_depth: {max_depth}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(root_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print(f"[-] Membaca dataset dari jalur: {data_path}")
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
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
    
    mlflow.set_experiment("Automated_Retraining")
    with mlflow.start_run(run_name="GitHub_Actions_Direct"):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "automated_model")
        print(f"[+] Retraining Sukses! Akurasi: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()