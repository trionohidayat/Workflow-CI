import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(file_path):
    """
    Fungsi otomatisasi untuk memuat, membersihkan, dan memproses dataset Telco Customer Churn.
    Mendukung pemisahan data train dan test secara konsisten.
    """
    print(f"[-] Memuat data dari: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Menangani missing value tersembunyi (spasi kosong) pada TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Menggunakan nilai median yang konsisten
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'].fillna(total_charges_median, inplace=True)
    
    # 2. Menghapus fitur yang tidak relevan untuk pemodelan
    if 'customerID' in df.columns:
        df_clean = df.drop(columns=['customerID'])
    else:
        df_clean = df.copy()
        
    # 3. Encoding Target Variabel (Churn) jika ada di dalam dataset
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        X = df_clean.drop(columns=['Churn'])
        y = df_clean['Churn']
    else:
        X = df_clean
        y = None

    # 4. Menerapkan One-Hot Encoding pada fitur kategorikal
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # 5. Membagi data menjadi Train dan Test Set jika label target tersedia
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        print("[+] Preprocessing sukses! Data Train & Test berhasil dibuat.")
        return X_train, X_test, y_train, y_test
    else:
        print("[+] Preprocessing sukses! Data siap digunakan untuk inference.")
        return X_encoded

if __name__ == "__main__":
    # Blok pengujian lokal ketika script dijalankan langsung
    PATH_DATA = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    try:
        X_train, X_test, y_train, y_test = prepare_data(PATH_DATA)
        print(f"-> Dimensi X_train: {X_train.shape}, Dimensi X_test: {X_test.shape}")
    except FileNotFoundError:
        print(f"[!] File {PATH_DATA} tidak ditemukan. Pastikan posisi file sudah benar.")