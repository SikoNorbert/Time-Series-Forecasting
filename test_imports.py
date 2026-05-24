import sys
print("Starting imports...")

try:
    print("Importing pandas...")
    import pandas as pd
    print("Pandas imported.")
except Exception as e:
    print(f"Pandas failed: {e}")

try:
    print("Importing torch...")
    import torch
    print("Torch imported.")
except Exception as e:
    print(f"Torch failed: {e}")

try:
    print("Importing tensorflow...")
    import tensorflow as tf
    print("TensorFlow imported.")
except Exception as e:
    print(f"TensorFlow failed: {e}")

try:
    print("Importing streamlit...")
    import streamlit as st
    print("Streamlit imported.")
except Exception as e:
    print(f"Streamlit failed: {e}")

print("All imports finished.")
