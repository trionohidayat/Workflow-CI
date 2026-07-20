import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

# 1. Aktifkan autolog MLflow (Otomatis mencatat parameter, metrik, dan model)
mlflow.autolog()

# 2. Load Dataset Hasil Preprocessing
df = pd.read_csv('telco-telco-customer_preprocessing.csv')

# 3. Pemisahan Feature dan Target
X = df.drop(columns=['Churn']) if 'Churn' in df.columns else df.iloc[:, :-1]
y = df['Churn'] if 'Churn' in df.columns else df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Eksperimen & Melatih Model
mlflow.set_experiment("Basic_Modelling_Autolog")

with mlflow.start_run(run_name="Basic_Autolog_Run"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("[+] Training selesai. Log telah dicatat otomatis oleh mlflow.autolog()")