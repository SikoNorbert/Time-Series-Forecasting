import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

import subprocess
import sys

def check_tensorflow_availability():
    """
    Checks if TensorFlow can be imported without crashing (segfault).
    Returns True if available, False otherwise.
    """
    try:
        # Run a subprocess to test import. If it segfaults, return code will be non-zero (usually 139).
        result = subprocess.run(
            [sys.executable, "-c", "import tensorflow"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def load_and_preprocess_data():

    def parse_dates(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
    
    def data_after(self, time: str):
        self.data = self.data[self.data['Date'] >= pd.Timestamp(time)]

    def activate(self, time: str):
        self.parse_dates()
        self.data_after(time)
        return self.data

def load_and_preprocess_data():
    # Load data
    # Assuming dataFrame directory is in the same directory as this script or main.py
    # We might need to adjust paths if running from a different location, but for now relative paths should work if CWD is correct.
    
    files = {
        'AMD_1980_2023': 'dataFrame/AMD (1980 -11.07.2023).csv',
        'AMD_2023_2024': 'dataFrame/AMD (2023 - 08.04.2024).csv',
        'ASUS_2000_2023': 'dataFrame/ASUS (2000 - 11.07.2023).csv',
        'ASUS_2023_2024': 'dataFrame/ASUS (2023 - 08.04.2024).csv',
        'INTEL_1980_2023': 'dataFrame/INTEL (1980 - 11.07.2023).csv',
        'INTEL_2023_2024': 'dataFrame/Intel (2023 - 08.04.2024).csv',
        'MSI_2023_2024': 'dataFrame/MSI (2023 - 08.04.2024).csv',
        'NVIDIA_1999_2023': 'dataFrame/NVIDIA (1999 -11.07.2023).csv',
        'NVIDIA_2023_2024': 'dataFrame/Nvidia (2023 - 08.04.2024).csv'
    }
    
    dfs = {}
    for key, path in files.items():
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
        else:
            # Fallback or error handling if needed, for now just skip or let it fail if critical
            print(f"Warning: {path} not found.")
            return None

    # Merge DataFrames
    # Logic from original main.py
    data_frames = [
        dfs['AMD_1980_2023'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'AMD_1980_2023'}),
        dfs['AMD_2023_2024'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'AMD_2023_2024'}),
        dfs['ASUS_2000_2023'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'ASUS_2000_2023'}),
        dfs['ASUS_2023_2024'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'ASUS_2023_2024'}),
        dfs['INTEL_1980_2023'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'INTEL_1980_2023'}),
        dfs['INTEL_2023_2024'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'INTEL_2023_2024'}),
        dfs['MSI_2023_2024'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'MSI_2023_2024'}),
        dfs['NVIDIA_1999_2023'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'NVIDIA_1999_2023'}),
        dfs['NVIDIA_2023_2024'][['Date', 'Adj Close']].rename(columns={'Adj Close': 'NVIDIA_2023_2024'}),
    ]

    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = merged_df.merge(df, on='Date', how='inner')
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df = merged_df.dropna()
    
    return merged_df

def prepare_data_for_training(merged_df, adj_cols, seq_length=30):
    numeric_df = merged_df[adj_cols]
    data = numeric_df.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])

    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler
