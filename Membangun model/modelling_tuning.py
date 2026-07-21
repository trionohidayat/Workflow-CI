import os
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing.automate_triono import prepare_data

# 1. Inisialisasi DagsHub secara online berdasarkan akun kalian
REPO_OWNER = "trionohidayat3"  
REPO_NAME = "my-first-repo"    
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def tune_and_track():
    # 2. Ambil data hasil preprocessing otomatis
    PATH_DATA = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X_train, X_test, y_train, y_test = prepare_data(PATH_DATA)
    
    # 3. Set Nama Eksperimen di MLflow
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    # 4. Mulai Parent Run untuk Hyperparameter Tuning
    with mlflow.start_run(run_name="Tuning_RandomForest") as parent_run:
        print("[-] Memulai proses GridSearchCV untuk Hyperparameter Tuning...")
        
        # Definisikan parameter yang akan diuji
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'criterion': ['gini', 'entropy']
        }
        
        # Inisialisasi model dasar dan GridSearch
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
        
        # Jalankan proses tuning
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"[+] Kombinasi parameter terbaik ditemukan: {best_params}")
        
        # 5. Evaluasi Model Terbaik pada Data Test
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 6. Manual Logging Parameter Terbaik dan Metriks ke Parent Run
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
            
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 7. Menyimpan Artefak Model & 2 File Tambahan (Syarat Advanced)
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # File tambahan 1: Log detail seluruh hasil tuning
        tuning_results = pd.DataFrame(grid_search.cv_results_)
        tuning_results.to_csv("tuning_results.csv", index=False)
        mlflow.log_artifact("tuning_results.csv", artifact_path="metadata")
        
        # File tambahan 2: Summary performa model terbaik
        with open("best_model_summary.txt", "w") as f:
            f.write(f"Best Params: {best_params}\nAccuracy: {acc}\nF1-Score: {f1}")
        mlflow.log_artifact("best_model_summary.txt", artifact_path="metadata")
        
        print("[+] Semua parameter terbaik, metriks, dan artefak sukses di-track ke DagsHub!")

if __name__ == "__main__":
    # Pastikan pandas di-import di lingkup pengujian jika dibutuhkan ekspor data
    import pandas as pd
    tune_and_track()