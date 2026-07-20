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
    # Menggunakan URL dataset langsung agar aman saat dijalankan di server GitHub
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv" 
    # Catatan: Kita ganti dengan link raw github dataset churn kalian nanti agar pipeline lancar
    
    # Preprocessing kilat untuk automasi
    df = pd.read_csv("https://raw.githubusercontent.com/Triono-Hidayat/dataset-dummy/main/churn_cleaned.csv") # Contoh link raw
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Automated_Retraining")
    with mlflow.start_run(run_name="GitHub_Actions_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "automated_model")
        print(f"[+] Retraining Selesai! Akurasi: {acc:.4f}")

if __name__ == "__main__":
    run_retraining()