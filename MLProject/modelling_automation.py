import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# PENTING: Menggunakan token dari GitHub Secrets agar otentikasi otomatis berjalan
dagshub_token = os.getenv("DAGSHUB_CLIENT_TOKEN")
if dagshub_token:
    os.environ["DAGSHUB_CLIENT_TOKEN"] = dagshub_token

# Sesuaikan dengan nama repositori DagsHub target utama kalian
REPO_OWNER = "trionohidayat3"
REPO_NAME = "my-first-repo"

print(f"[-] Menginisialisasi DagsHub untuk {REPO_OWNER}/{REPO_NAME}...")
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)