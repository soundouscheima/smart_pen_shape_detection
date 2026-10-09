import socket
import csv
import threading
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import time
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# ===== Global Variables =====
class DataRecorder:
    def __init__(self):
        self.is_recording = False
        self.current_data = []
        self.current_filename = None
        self.shape = None
        self.quality = None
        self.server_socket = None
        self.client_socket = None
        self.recording_thread = None
        self.server_running = False
        self.predicted_shape = None
        self.predicted_quality = None
        self.prediction_ready = False
        self.connected = False
        self.client_address = None

data_recorder = DataRecorder()

# ===== ML Models Loading =====
print("[SYSTEM] Loading ML models...")
SENSOR_COLS = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# Try to load models
try:
    # Check for best models first
    if os.path.exists("best_model_shape_detector.pkl"):
        with open('best_model_shape_detector.pkl', 'rb') as f:
            model_shape = pickle.load(f)
        print("[MODELS] Loaded best shape model")
    elif os.path.exists("model_shape_detector.pkl"):
        with open('model_shape_detector.pkl', 'rb') as f:
            model_shape = pickle.load(f)
        print("[MODELS] Loaded standard shape model")
    else:
        model_shape = None
        print("[MODELS] No shape model found")
    
    if os.path.exists("best_model_quality_detector.pkl"):
        with open('best_model_quality_detector.pkl', 'rb') as f:
            model_quality = pickle.load(f)
        print("[MODELS] Loaded best quality model")
    elif os.path.exists("model_quality_detector.pkl"):
        with open('model_quality_detector.pkl', 'rb') as f:
            model_quality = pickle.load(f)
        print("[MODELS] Loaded standard quality model")
    else:
        model_quality = None
        print("[MODELS] No quality model found")
    
    if os.path.exists("scaler.pkl"):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("[MODELS] Loaded scaler")
    else:
        scaler = None
        print("[MODELS] No scaler found")
        
except Exception as e:
    print(f"[ERROR] Loading models: {e}")
    model_shape = None
    model_quality = None
    scaler = None

# ===== Feature Extraction Function =====
def extract_features_clean(df):
    """Extract features from sensor data exactly like training"""
    try:
        if len(df) < 10:
            return None
        
        # Select ONLY sensor columns
        data = df[SENSOR_COLS].values
        
        # Extract the 30 statistical features
        features = []
        for i in range(data.shape[1]):  # Loop through X, Y, Z for Accel and Gyro
            axis_data = data[:, i]
            features.append(np.mean(axis_data))
            features.append(np.std(axis_data))
            features.append(np.max(axis_data))
            features.append(np.min(axis_data))
            features.append(np.sqrt(np.mean(axis_data**2)))  # RMS
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        print(f"[ERROR] Feature extraction: {e}")
        return None

# ===== Prediction Function =====
def predict_from_data(file_path):
    """Predict shape and quality from recorded data"""
    try:
        if not os.path.exists(file_path):
            return None, None, "File not found"
        
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Check if we have the required columns
        missing_cols = [col for col in SENSOR_COLS if col not in df.columns]
        if missing_cols:
            print(f"[PREDICTION] Missing columns: {missing_cols}")
            # Try to rename columns if they exist with different names
            column_mapping = {
                'accel_x': 'Accel_X', 'accel_y': 'Accel_Y', 'accel_z': 'Accel_Z',
                'gyro_x': 'Gyro_X', 'gyro_y': 'Gyro_Y', 'gyro_z': 'Gyro_Z'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
        
        # Clean non-numeric data
        for col in SENSOR_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in sensor columns
        df = df.dropna(subset=SENSOR_COLS)
        
        # Check if we have enough data
        if len(df) < 10:
            return None, None, f"Insufficient data for prediction ({len(df)} points)"
        
        # Extract Features
        features = extract_features_clean(df)
        if features is None:
            return None, None, "Failed to extract features"
        
        # Scale Features
        features_scaled = scaler.transform(features)
        
        # Predict
        pred_shape = model_shape.predict(features_scaled)[0]
        pred_quality = model_quality.predict(features_scaled)[0]
        
        return pred_shape, pred_quality, "Success"
        
    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")
        return None, None, str(e)

# ===== Socket Server Functions =====
def start_socket_server():
    """Start the socket server in a separate thread"""
    try:
        data_recorder.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_recorder.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_recorder.server_socket.bind(("0.0.0.0", 5000))
        data_recorder.server_socket.listen(5)  # Increased backlog
        data_recorder.server_running = True
        print(f"[SERVER] Socket server listening on 0.0.0.0:5000")
        print(f"[SERVER] Waiting for MPU device connection...")
        
        while data_recorder.server_running:
            try:
                client_socket, addr = data_recorder.server_socket.accept()
                print(f"[CONNECTED] Device connected from {addr}")
                data_recorder.client_socket = client_socket
                data_recorder.connected = True
                data_recorder.client_address = addr
                
                # Send welcome message
                client_socket.send(b"CONNECTED_TO_SERVER\n")
                
                # Handle client in separate thread
                client_handler = threading.Thread(target=handle_client, args=(client_socket, addr))
                client_handler.daemon = True
                client_handler.start()
                
            except Exception as e:
                if data_recorder.server_running:
                    print(f"[ERROR] Socket accept error: {e}")
                    break
                    
    except Exception as e:
        print(f"[ERROR] Socket server error: {e}")
    finally:
        if data_recorder.server_socket:
            data_recorder.server_socket.close()

def handle_client(client_socket, addr):
    """Handle individual client connection"""
    try:
        while data_recorder.server_running:
            try:
                # Check if socket is still connected
                client_socket.settimeout(5.0)  # 5 second timeout
                
                # Receive data
                data = client_socket.recv(1024)
                if not data:
                    print(f"[INFO] Client {addr} disconnected")
                    break
                
                # Process data
                decoded_data = data.decode('utf-8', errors='ignore').strip()
                if decoded_data:
                    process_incoming_data(decoded_data, client_socket)
                    
            except socket.timeout:
                continue  # Just continue if timeout
            except socket.error as e:
                print(f"[ERROR] Socket error from {addr}: {e}")
                break
            except Exception as e:
                print(f"[ERROR] Client handler error: {e}")
                continue
                
    except Exception as e:
        print(f"[ERROR] Client handler crashed: {e}")
    finally:
        client_socket.close()
        data_recorder.connected = False
        data_recorder.client_socket = None
        print(f"[INFO] Client {addr} handler stopped")

def process_incoming_data(data, client_socket):
    """Process incoming data from MPU"""
    lines = data.strip().split('\n')
    
    for line in lines:
        if not line:
            continue
            
        try:
            # Parse the data line
            parts = line.strip().split(',')
            
            if len(parts) >= 8:
                # Create data point
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'hand_id': parts[0],
                    'timestamp_ms': parts[1] if len(parts) > 1 else '0',
                    'accel_x': float(parts[2]) if len(parts) > 2 else 0,
                    'accel_y': float(parts[3]) if len(parts) > 3 else 0,
                    'accel_z': float(parts[4]) if len(parts) > 4 else 0,
                    'gyro_x': float(parts[5]) if len(parts) > 5 else 0,
                    'gyro_y': float(parts[6]) if len(parts) > 6 else 0,
                    'gyro_z': float(parts[7]) if len(parts) > 7 else 0,
                    'shape': data_recorder.shape,
                    'quality': data_recorder.quality
                }
                
                # Store in memory for display
                data_recorder.current_data.append(data_point)
                
                # Keep only last 50 points for display
                if len(data_recorder.current_data) > 50:
                    data_recorder.current_data.pop(0)
                
                # Save to file if recording
                if data_recorder.is_recording and data_recorder.current_filename:
                    save_data_point(data_point)
                
                # Optional: Send acknowledgment back
                # client_socket.send(b"OK\n")
                
        except Exception as e:
            print(f"[ERROR] Processing data line '{line}': {e}")

def save_data_point(data_point):
    """Save a single data point to CSV"""
    try:
        # Check if file exists to write header
        file_exists = os.path.exists(data_recorder.current_filename)
        
        with open(data_recorder.current_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'HandID', 'Timestamp_ms', 'Accel_X', 'Accel_Y', 'Accel_Z',
                    'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Shape', 'Quality'
                ])
            
            # Write data row
            writer.writerow([
                data_point['timestamp'],
                data_point['hand_id'],
                data_point['timestamp_ms'],
                data_point['accel_x'],
                data_point['accel_y'],
                data_point['accel_z'],
                data_point['gyro_x'],
                data_point['gyro_y'],
                data_point['gyro_z'],
                data_point['shape'],
                data_point['quality']
            ])
            
    except Exception as e:
        print(f"[ERROR] Saving data point: {e}")

# ===== Flask Routes =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real')
def real():
    return render_template('real.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    if data_recorder.is_recording:
        return jsonify({'status': 'error', 'message': 'Recording already in progress'})
    
    shape = request.json.get('shape', 'unknown')
    quality = request.json.get('quality', 'unknown')
    
    if not data_recorder.connected:
        return jsonify({'status': 'error', 'message': 'No MPU device connected'})
    
    data_recorder.shape = shape
    data_recorder.quality = quality
    data_recorder.is_recording = True
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_recorder.current_filename = f"data/{shape}_{quality}_{timestamp}.csv"
    
    # Clear previous data
    data_recorder.current_data = []
    
    return jsonify({
        'status': 'success', 
        'message': f'Recording started - Shape: {shape}, Quality: {quality}',
        'filename': data_recorder.current_filename
    })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    if not data_recorder.is_recording:
        return jsonify({'status': 'error', 'message': 'No recording in progress'})
    
    data_recorder.is_recording = False
    
    # Wait a moment for any remaining data
    time.sleep(0.2)
    
    recorded_points = len(data_recorder.current_data)
    
    return jsonify({
        'status': 'success', 
        'message': f'Recording stopped. {recorded_points} data points recorded.',
        'data_points': recorded_points,
        'filename': data_recorder.current_filename
    })

@app.route('/predict_last', methods=['POST'])
def predict_last():
    """Predict shape and quality from the last recording"""
    if not data_recorder.current_filename or not os.path.exists(data_recorder.current_filename):
        return jsonify({'status': 'error', 'message': 'No recording found to predict'})
    
    if model_shape is None or model_quality is None or scaler is None:
        return jsonify({'status': 'error', 'message': 'ML models not loaded properly'})
    
    try:
        pred_shape, pred_quality, message = predict_from_data(data_recorder.current_filename)
        
        if pred_shape is not None and pred_quality is not None:
            data_recorder.predicted_shape = pred_shape
            data_recorder.predicted_quality = pred_quality
            data_recorder.prediction_ready = True
            
            return jsonify({
                'status': 'success',
                'predicted_shape': pred_shape,
                'predicted_quality': pred_quality,
                'message': message,
                'filename': os.path.basename(data_recorder.current_filename)
            })
        else:
            return jsonify({'status': 'error', 'message': message})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """Predict shape and quality from uploaded file"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Only CSV files are allowed'})
    
    if model_shape is None or model_quality is None or scaler is None:
        return jsonify({'status': 'error', 'message': 'ML models not loaded'})
    
    try:
        # Save uploaded file temporarily
        os.makedirs('data/uploads', exist_ok=True)
        temp_path = f"data/uploads/upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file.save(temp_path)
        
        # Predict
        pred_shape, pred_quality, message = predict_from_data(temp_path)
        
        # Clean up temp file (optional)
        # os.remove(temp_path)
        
        if pred_shape is not None and pred_quality is not None:
            # Update global variables
            data_recorder.predicted_shape = pred_shape
            data_recorder.predicted_quality = pred_quality
            data_recorder.prediction_ready = True
            
            return jsonify({
                'status': 'success',
                'predicted_shape': pred_shape,
                'predicted_quality': pred_quality,
                'message': message,
                'filename': file.filename
            })
        else:
            return jsonify({'status': 'error', 'message': message})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_status')
def get_status():
    return jsonify({
        'is_recording': data_recorder.is_recording,
        'is_connected': data_recorder.connected,
        'client_address': data_recorder.client_address,
        'shape': data_recorder.shape,
        'quality': data_recorder.quality,
        'data_points': len(data_recorder.current_data),
        'current_filename': data_recorder.current_filename,
        'prediction_ready': data_recorder.prediction_ready,
        'predicted_shape': data_recorder.predicted_shape,
        'predicted_quality': data_recorder.predicted_quality,
        'ml_models_loaded': model_shape is not None and scaler is not None
    })

@app.route('/get_recent_data')
def get_recent_data():
    """Return recent data for display"""
    recent_data = data_recorder.current_data[-10:] if data_recorder.current_data else []
    return jsonify({
        'data': recent_data,
        'total_points': len(data_recorder.current_data)
    })

@app.route('/save_data', methods=['POST'])
def save_data():
    """Force save current data"""
    if not data_recorder.current_filename or not os.path.exists(data_recorder.current_filename):
        return jsonify({'status': 'error', 'message': 'No data to save'})
    
    filename = data_recorder.current_filename
    file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
    
    return jsonify({
        'status': 'success', 
        'message': f'Data saved as {os.path.basename(filename)}',
        'filename': os.path.basename(filename),
        'file_path': filename,
        'data_points': len(data_recorder.current_data),
        'file_size': file_size
    })

@app.route('/download/<filename>')
def download_file(filename):
    """Download the recorded CSV file"""
    try:
        file_path = os.path.join('data', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            # Try in uploads folder
            upload_path = os.path.join('data/uploads', filename)
            if os.path.exists(upload_path):
                return send_file(upload_path, as_attachment=True)
            else:
                return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/list_files')
def list_files():
    """List all recorded files"""
    try:
        files = []
        if os.path.exists('data'):
            for f in os.listdir('data'):
                if f.endswith('.csv'):
                    file_path = os.path.join('data', f)
                    files.append({
                        'name': f,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    })
        
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/test_connection')
def test_connection():
    """Test if server is running"""
    return jsonify({
        'status': 'success',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })

# ===== Application Startup =====
def initialize_server():
    """Initialize the socket server when the application starts"""
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/uploads', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        
        # Start socket server in background thread
        server_thread = threading.Thread(target=start_socket_server)
        server_thread.daemon = True
        server_thread.start()
        
        print("[SYSTEM] IMU Data Collection Server Initialized")
        print("[SYSTEM] Socket server started on 0.0.0.0:5000")
        print("[SYSTEM] Web server starting on 0.0.0.0:5001")
        
    except Exception as e:
        print(f"[ERROR] Server initialization failed: {e}")

# Initialize when module is imported
initialize_server()

if __name__ == '__main__':
    print("[SYSTEM] Starting Flask application...")
    print("[SYSTEM] Access the web interface at http://localhost:5001")
    print("[SYSTEM] Access prediction page at http://localhost:5001/real")
    print("[SYSTEM] Testing connection at http://localhost:5001/test_connection")
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)