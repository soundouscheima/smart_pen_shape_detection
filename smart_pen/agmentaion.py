import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
import pickle

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = r"C:\Users\shaym\OneDrive\Desktop\collectedData\collectedData"
SENSOR_COLS = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# ==========================================
# 2. AUGMENTATION FUNCTIONS
# ==========================================
def augment_jitter(data_array, noise_sigma=0.05):
    noise = np.random.normal(loc=0, scale=noise_sigma, size=data_array.shape)
    return data_array + noise

def augment_scaling(data_array, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(1, data_array.shape[1]))
    return data_array * factor

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def extract_features(df):
    # Ensure data is float (numbers) before processing
    data = df[SENSOR_COLS].values.astype(float)
    
    features = []
    for i in range(data.shape[1]): # Iterate over columns
        axis_data = data[:, i]
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
        features.append(np.max(axis_data))
        features.append(np.min(axis_data))
        features.append(np.sqrt(np.mean(axis_data**2))) # RMS
        
    return np.array(features)

# ==========================================
# 4. MAIN PROCESSING LOOP
# ==========================================
def process_data():
    print("Starting Data Processing...")
    
    X_list = []
    y_shape_list = []
    y_quality_list = []
    
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    print(f"Found {len(csv_files)} files.")

    for filename in csv_files:
        try:
            # A. Load Data
            df = pd.read_csv(filename)
            
            # --- THE FIX IS HERE ---
            # Force columns to be numbers. If it's text, turn it into NaN
            for col in SENSOR_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows where conversion failed (bad data)
            df = df.dropna(subset=SENSOR_COLS)
            
            if len(df) < 10: 
                continue

            # Get raw numpy array for augmentation
            raw_data = df[SENSOR_COLS].values

            # --- ORIGINAL DATA ---
            feats_original = extract_features(df)
            
            # Parse Label
            basename = os.path.basename(filename)
            parts = basename.split('_')
            label_shape = parts[0]
            label_quality = parts[1]

            X_list.append(feats_original)
            y_shape_list.append(label_shape)
            y_quality_list.append(label_quality)

            # --- AUGMENTATION 1: JITTER ---
            aug_data_1 = augment_jitter(raw_data)
            df_aug_1 = pd.DataFrame(aug_data_1, columns=SENSOR_COLS)
            feats_aug_1 = extract_features(df_aug_1)
            
            X_list.append(feats_aug_1)
            y_shape_list.append(label_shape)
            y_quality_list.append(label_quality)

            # --- AUGMENTATION 2: SCALING ---
            aug_data_2 = augment_scaling(raw_data)
            df_aug_2 = pd.DataFrame(aug_data_2, columns=SENSOR_COLS)
            feats_aug_2 = extract_features(df_aug_2)
            
            X_list.append(feats_aug_2)
            y_shape_list.append(label_shape)
            y_quality_list.append(label_quality)

        except Exception as e:
            # This will print which file is still bad, but won't crash the script
            print(f"Skipping {os.path.basename(filename)}: {e}")

    # Convert to Numpy Arrays
    X = np.array(X_list)
    y_s = np.array(y_shape_list)
    y_q = np.array(y_quality_list)
    
    print(f"\nProcessing Complete!")
    print(f"Total samples: {len(X)}")
    
    return X, y_s, y_q

# ==========================================
# 5. EXECUTION & SAVING
# ==========================================
if __name__ == "__main__":
    X, y_shape, y_quality = process_data()

    if X is not None and len(X) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Saving processed data...")
        with open('X_data.pkl', 'wb') as f: pickle.dump(X_scaled, f)
        with open('y_shape.pkl', 'wb') as f: pickle.dump(y_shape, f)
        with open('y_quality.pkl', 'wb') as f: pickle.dump(y_quality, f)
        with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
        
        print("Done! Files saved.")