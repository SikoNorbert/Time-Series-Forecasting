import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import load_and_preprocess_data, prepare_data_for_training
# Models are imported lazily to prevent segmentation faults due to library conflicts


# Page Configuration
st.set_page_config(
    page_title="Stock Market Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training"])

# Load Data (Cached)
@st.cache_data
def get_data():
    return load_and_preprocess_data()

merged_df = get_data()

if merged_df is None:
    st.error("Failed to load data. Please check if the 'dataFrame' directory exists and contains the required CSV files.")
    st.stop()

# Define columns
adj_cols = ['AMD_1980_2023', 'AMD_2023_2024', 'ASUS_2000_2023', 'ASUS_2023_2024', 'INTEL_1980_2023', 'INTEL_2023_2024', 'MSI_2023_2024', 'NVIDIA_1999_2023', 'NVIDIA_2023_2024']

# Home Page
if page == "Home":
    st.title("📈 Stock Market Analysis & Prediction")
    st.image("https://images.unsplash.com/photo-1611974765270-ca1258634369?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", use_container_width=True)
    
    st.markdown("""
    ### Welcome to the Stock Market Analysis Dashboard
    
    This application leverages advanced machine learning techniques to analyze and predict stock prices for major tech giants: **NVIDIA, AMD, Intel, ASUS, and MSI**.
    
    #### Key Features:
    - **Data Analysis**: Explore historical stock data, visualize trends, and analyze correlations.
    - **Model Training**: Train Deep Learning models (ANN, CNN, RNN) using PyTorch or TensorFlow.
    - **Predictions**: Visualize model performance and future stock price predictions.
    
    Navigate through the sidebar to get started!
    """)

# Data Analysis Page
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Stock Trends", "Correlation"])
    
    with tab1:
        st.subheader("Merged Dataset")
        st.dataframe(merged_df.head(100), use_container_width=True)
        st.write(f"Total Records: {len(merged_df)}")
        st.write(f"Columns: {merged_df.columns.tolist()}")
        
    with tab2:
        st.subheader("Stock Price Trends")
        selected_stocks = st.multiselect("Select Stocks to Compare", adj_cols, default=adj_cols[:2])
        
        if selected_stocks:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in selected_stocks:
                sns.lineplot(data=merged_df, x='Date', y=col, label=col, ax=ax)
            ax.set_title("Comparative Stock Price Trend")
            ax.set_xlabel("Date")
            ax.set_ylabel("Adjusted Close Price")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Please select at least one stock to visualize.")
            
    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(merged_df[adj_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation of Adjusted Close Prices")
        st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.title("🧠 Model Training & Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Select Target Stock (Simplified for demo, using all features to predict one or all? 
        # Original code predicted all features. Let's stick to that or allow selecting one.)
        # The original code: X is all features, y is all features shifted.
        
        model_type = st.selectbox("Select Model Type", ["ANN", "CNN", "RNN"])
        # Framework Selection with Availability Check
        from utils import check_tensorflow_availability
        
        available_frameworks = ["PyTorch"]
        if "tf_available" not in st.session_state:
            with st.spinner("Checking TensorFlow availability..."):
                st.session_state.tf_available = check_tensorflow_availability()
        
        if st.session_state.tf_available:
            available_frameworks.append("TensorFlow")
        
        framework = st.selectbox("Select Framework", available_frameworks)
        
        if framework == "TensorFlow":
            st.warning("⚠️ TensorFlow is unstable in this environment. If the app crashes, please use PyTorch.")
        elif not st.session_state.tf_available:
            st.info("ℹ️ TensorFlow is disabled because it causes crashes in this environment. Using PyTorch.")
        
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=0)
        seq_length = st.slider("Sequence Length", min_value=10, max_value=60, value=30)
        
        train_btn = st.button("Start Training")
        
    with col2:
        st.subheader("Training Progress & Results")
        
        if train_btn:
            with st.spinner(f"Training {model_type} using {framework}..."):
                # Prepare Data
                X_train, y_train, X_test, y_test, scaler = prepare_data_for_training(merged_df, adj_cols, seq_length)
                n_features = len(adj_cols)
                
                rmse = None
                y_pred = None
                train_losses = []
                val_losses = []
                
                try:
                    if framework == "PyTorch":
                        # Lazy import PyTorch modules
                        import torch
                        from torch.utils.data import DataLoader
                        from models_pt import ANN, CNN, RNN, train_evaluate_pt, TimeSeriesDataset

                        train_dataset = TimeSeriesDataset(X_train, y_train)
                        test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size)
                        
                        if model_type == "ANN":
                            model = ANN(seq_length, n_features)
                        elif model_type == "CNN":
                            model = CNN(seq_length, n_features)
                        elif model_type == "RNN":
                            model = RNN(input_size=n_features, n_features=n_features)
                            
                        rmse, y_pred, train_losses, val_losses = train_evaluate_pt(model, train_dataset, test_loader, batch_size, epochs)
                        
                    elif framework == "TensorFlow":
                        # Lazy import TensorFlow modules
                        from models_tf import create_ann_tf, create_cnn_tf, create_rnn_tf, train_evaluate_tf

                        if model_type == "ANN":
                            model = create_ann_tf(seq_length, n_features)
                        elif model_type == "CNN":
                            model = create_cnn_tf(seq_length, n_features)
                        elif model_type == "RNN":
                            model = create_rnn_tf(seq_length, n_features)
                            
                        rmse, y_pred, train_losses, val_losses = train_evaluate_tf(model, X_train, y_train, X_test, y_test, batch_size, epochs)
                    
                    st.success(f"Training Completed! RMSE: {rmse:.4f}")
                    
                    # Plot Loss
                    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
                    ax_loss.plot(train_losses, label='Train Loss')
                    ax_loss.plot(val_losses, label='Validation Loss')
                    ax_loss.set_title(f'{model_type} ({framework}) Training Loss')
                    ax_loss.set_xlabel('Epoch')
                    ax_loss.set_ylabel('Loss')
                    ax_loss.legend()
                    st.pyplot(fig_loss)
                    
                    # Plot Predictions for a few stocks
                    st.subheader("Prediction vs Actual")
                    dates_test = merged_df['Date'].iloc[-len(y_test):].values
                    
                    # Select stock to view
                    stock_to_view = st.selectbox("Select Stock to View Prediction", adj_cols)
                    stock_idx = adj_cols.index(stock_to_view)
                    
                    y_true_rescaled = scaler.inverse_transform(y_test)[:, stock_idx]
                    y_pred_rescaled = scaler.inverse_transform(y_pred)[:, stock_idx]
                    
                    fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x=dates_test, y=y_true_rescaled, label='Actual', ax=ax_pred)
                    sns.lineplot(x=dates_test, y=y_pred_rescaled, label='Predicted', linestyle='--', ax=ax_pred)
                    ax_pred.set_title(f'{stock_to_view}: Actual vs Predicted')
                    ax_pred.set_xlabel('Date')
                    ax_pred.set_ylabel('Price')
                    ax_pred.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig_pred)
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    st.exception(e)