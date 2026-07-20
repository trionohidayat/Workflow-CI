import os
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing.automate_triono import prepare_data

# 1. Inisialisasi DagsHub secara online (Ganti dengan username dan nama repo GitHub kalian)
# Pastikan kalian sudah menghubungkan repo GitHub ke DagsHub terlebih dahulu
REPO_OWNER = "trionohidayat3"  
REPO_NAME = "my-first-repo"     # <-- Sesuaikan dengan nama repo kalian
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def train_and_track():
    # 2. Ambil data yang sudah di-preprocessing otomatis dari Kriteria 1
    PATH_DATA = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X_train, X_test, y_train, y_test = prepare_data(PATH_DATA)
    
    # 3. Set Nama Eksperimen di MLflow
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    # 4. Mulai Run Tracking MLflow
    with mlflow.start_run(run_name="Baseline_RandomForest"):
        print("[-] Melatih model Random Forest...")
        
        # Hyperparameter model
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Inisialisasi dan latih model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Prediksi data test
        y_pred = model.predict(X_test)
        
        # 5. Hitung Metriks Evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[+] Model selesai dilatih. Akurasi: {acc:.4f}")
        
        # 6. Manual Logging ke MLflow (Kritikal untuk level Skilled/Advanced)
        # Log Parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Log Metriks
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 7. Log Model & Artefak Tambahan (Minimal 2 artefak untuk Advanced)
        # Menyimpan model ke MLflow Artifacts
        mlflow.sklearn.log_model(model, "model")
        
        # Membuat file teks log internal sebagai artefak tambahan ke-1
        with open("training_summary.txt", "w") as f:
            f.write(f"Model: RandomForestClassifier\nAccuracy: {acc}\nF1-Score: {f1}")
        mlflow.log_artifact("training_summary.txt", artifact_path="metadata")
        
        # Menyimpan contoh struktur data input sebagai artefak tambahan ke-2
        X_test.head(5).to_csv("sample_input.csv", index=False)
        mlflow.log_artifact("sample_input.csv", artifact_path="metadata")
        
        print("[+] Semua parameter, metriks, dan artefak berhasil di-track ke DagsHub!")

if __name__ == "__main__":
    train_and_track()