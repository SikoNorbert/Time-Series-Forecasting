import pandas as pd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import load_and_preprocess_data, prepare_data_for_training
from utils import load_and_preprocess_data, prepare_data_for_training
# from models_tf import create_ann_tf, train_evaluate_tf

print("Loading data...")
merged_df = load_and_preprocess_data()
adj_cols = ['AMD_1980_2023', 'AMD_2023_2024', 'ASUS_2000_2023', 'ASUS_2023_2024', 'INTEL_1980_2023', 'INTEL_2023_2024', 'MSI_2023_2024', 'NVIDIA_1999_2023', 'NVIDIA_2023_2024']
seq_length = 30

print("Preparing data...")
X_train, y_train, X_test, y_test, scaler = prepare_data_for_training(merged_df, adj_cols, seq_length)
n_features = len(adj_cols)

print("Creating model...")
from models_tf import create_ann_tf, train_evaluate_tf
model = create_ann_tf(seq_length, n_features)

print("Starting training...")
try:
    rmse, y_pred, train_losses, val_losses = train_evaluate_tf(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=5)
    print(f"Training finished. RMSE: {rmse}")
    print("Done.")
except Exception as e:
    print(f"Training failed: {e}")
