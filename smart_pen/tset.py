import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# List of files you want to test
TEST_FILES = [
    r"C:\Users\shaym\Downloads\square_medium_20260113_110553.csv"
]

# The only columns the model is allowed to see
SENSOR_COLS = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# ==========================================
# 2. LOAD MODELS & SCALER
# ==========================================
print("Loading models...")
try:
    # Try loading the 'best' models first, or fallback to standard ones
    if os.path.exists("best_model_shape_detector.pkl"):
        with open('best_model_shape_detector.pkl', 'rb') as f: model_shape = pickle.load(f)
        with open('best_model_quality_detector.pkl', 'rb') as f: model_quality = pickle.load(f)
    else:
        # Fallback if you used the other script names
        with open('model_shape_detector.pkl', 'rb') as f: model_shape = pickle.load(f)
        with open('model_quality_detector.pkl', 'rb') as f: model_quality = pickle.load(f)

    # CRITICAL: You need the scaler to normalize the new data exactly like the training data
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    
except FileNotFoundError as e:
    print(f"\nERROR: Missing file! {e}")
    print("Make sure 'scaler.pkl' and your model .pkl files are in this folder.")
    exit()

# ==========================================
# 3. FEATURE EXTRACTION (Must match training exactly)
# ==========================================
def extract_features_clean(df):
    # 1. Select ONLY sensor columns (Removes HandID, Timestamp, Labels = NO CHEATING)
    data = df[SENSOR_COLS].values
    
    # 2. Extract the 30 statistical features
    features = []
    for i in range(data.shape[1]): # Loop through X, Y, Z for Accel and Gyro
        axis_data = data[:, i]
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
        features.append(np.max(axis_data))
        features.append(np.min(axis_data))
        features.append(np.sqrt(np.mean(axis_data**2))) # RMS
        
    return np.array(features).reshape(1, -1) # Reshape for a single sample

# ==========================================
# 4. PREDICTION LOOP
# ==========================================
print(f"\n{'='*60}")
print(f"{'FILE NAME':<40} | {'PREDICTED SHAPE':<15} | {'PREDICTED QUALITY':<15}")
print(f"{'='*60}")

for file_path in TEST_FILES:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # A. Load the CSV
        df = pd.read_csv(file_path)

        # B. Clean non-numeric data just in case
        for col in SENSOR_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=SENSOR_COLS)

        # C. Extract Features (Ignores 'Shape' and 'Quality' columns completely)
        features = extract_features_clean(df)

        # D. Scale Features (Apply the math from training)
        features_scaled = scaler.transform(features)

        # E. Predict
        pred_shape = model_shape.predict(features_scaled)[0]
        pred_quality = model_quality.predict(features_scaled)[0]

        # F. Print Result
        filename_only = os.path.basename(file_path)
        print(f"{filename_only:<40} | {pred_shape:<15} | {pred_quality:<15}")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

print(f"{'='*60}")