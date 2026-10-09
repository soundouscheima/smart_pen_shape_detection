import socket
import csv
import threading
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import time

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

data_recorder = DataRecorder()

# ===== Socket Server Functions =====
def start_socket_server():
    """Start the socket server in a separate thread"""
    try:
        data_recorder.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_recorder.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_recorder.server_socket.bind(("0.0.0.0", 5000))
        data_recorder.server_socket.listen(1)
        data_recorder.server_running = True
        print(f"[SERVER] Socket server listening on 0.0.0.0:5000")
        
        while data_recorder.server_running:
            try:
                client_socket, addr = data_recorder.server_socket.accept()
                print(f"[CONNECTED] Device connected from {addr}")
                data_recorder.client_socket = client_socket
            except Exception as e:
                if data_recorder.server_running:
                    print(f"[ERROR] Socket accept error: {e}")
    except Exception as e:
        print(f"[ERROR] Socket server error: {e}")

def record_data():
    """Record data from the connected client"""
    if not data_recorder.client_socket:
        return
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate filename with shape and quality
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_recorder.current_filename = f"data/{data_recorder.shape}_{data_recorder.quality}_{timestamp}.csv"
    
    try:
        with open(data_recorder.current_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            # Enhanced CSV Header with shape and quality information
            writer.writerow([
                "HandID",
                "Timestamp_ms", 
                "Accel_X",
                "Accel_Y", 
                "Accel_Z",
                "Gyro_X",
                "Gyro_Y",
                "Gyro_Z",
                "Shape",
                "Quality",
                "Recording_Start_Time"
            ])
            
            recording_start_time = datetime.now().isoformat()
            data_recorder.current_data = []  # Reset current data
            
            while data_recorder.is_recording and data_recorder.client_socket:
                try:
                    data = data_recorder.client_socket.recv(1024).decode("utf-8")
                    if not data:
                        print("[INFO] Client disconnected")
                        break
                    
                    for line in data.strip().splitlines():
                        fields = line.split(",")
                        
                        # Validate and enhance data with shape and quality
                        if len(fields) == 8:
                            # Add shape, quality, and recording start time to each data point
                            enhanced_fields = fields + [
                                data_recorder.shape, 
                                data_recorder.quality,
                                recording_start_time
                            ]
                            writer.writerow(enhanced_fields)
                            file.flush()  # Ensure data is written immediately
                            
                            # Store in memory for real-time display
                            data_recorder.current_data.append({
                                'hand_id': fields[0],
                                'timestamp': fields[1],
                                'accel_x': fields[2],
                                'accel_y': fields[3],
                                'accel_z': fields[4],
                                'gyro_x': fields[5],
                                'gyro_y': fields[6],
                                'gyro_z': fields[7],
                                'shape': data_recorder.shape,
                                'quality': data_recorder.quality
                            })
                            
                            # Keep only recent data for display (last 100 points)
                            if len(data_recorder.current_data) > 100:
                                data_recorder.current_data.pop(0)
                            
                            print(f"Recorded: {line}")
                            
                except socket.error as e:
                    print(f"[ERROR] Socket error during recording: {e}")
                    break
                except Exception as e:
                    print(f"[ERROR] Data processing error: {e}")
                    continue
                    
    except Exception as e:
        print(f"[ERROR] File writing error: {e}")

# ===== Flask Routes =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    if data_recorder.is_recording:
        return jsonify({'status': 'error', 'message': 'Recording already in progress'})
    
    shape = request.json.get('shape')
    quality = request.json.get('quality')
    
    if not shape or not quality:
        return jsonify({'status': 'error', 'message': 'Shape and quality are required'})
    
    if not data_recorder.client_socket:
        return jsonify({'status': 'error', 'message': 'No device connected'})
    
    data_recorder.shape = shape
    data_recorder.quality = quality
    data_recorder.is_recording = True
    
    # Start recording in a separate thread
    data_recorder.recording_thread = threading.Thread(target=record_data)
    data_recorder.recording_thread.daemon = True
    data_recorder.recording_thread.start()
    
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
    time.sleep(0.1)  # Allow recording thread to finish
    
    recorded_points = len(data_recorder.current_data)
    
    return jsonify({
        'status': 'success', 
        'message': f'Recording stopped. {recorded_points} data points recorded.',
        'data_points': recorded_points,
        'filename': data_recorder.current_filename
    })

@app.route('/get_status')
def get_status():
    return jsonify({
        'is_recording': data_recorder.is_recording,
        'is_connected': data_recorder.client_socket is not None,
        'shape': data_recorder.shape,
        'quality': data_recorder.quality,
        'data_points': len(data_recorder.current_data),
        'current_filename': data_recorder.current_filename
    })

@app.route('/get_recent_data')
def get_recent_data():
    """Return recent data for display"""
    return jsonify({
        'data': data_recorder.current_data[-10:],  # Last 10 data points
        'total_points': len(data_recorder.current_data)
    })

@app.route('/save_data', methods=['POST'])
def save_data():
    """Force save current data (already saved continuously, but provides confirmation)"""
    if not data_recorder.current_filename or not os.path.exists(data_recorder.current_filename):
        return jsonify({'status': 'error', 'message': 'No data to save'})
    
    filename = data_recorder.current_filename
    return jsonify({
        'status': 'success', 
        'message': f'Data saved as {os.path.basename(filename)}',
        'filename': os.path.basename(filename),
        'file_path': filename,
        'data_points': len(data_recorder.current_data)
    })

@app.route('/download/<filename>')
def download_file(filename):
    """Download the recorded CSV file"""
    try:
        file_path = os.path.join('data', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ===== Application Startup =====
def initialize_server():
    """Initialize the socket server when the application starts"""
    server_thread = threading.Thread(target=start_socket_server)
    server_thread.daemon = True
    server_thread.start()
    print("[SYSTEM] IMU Data Collection Server Initialized")

# Initialize when module is imported
initialize_server()

if __name__ == '__main__':
    print("[SYSTEM] Starting Flask application...")
    print("[SYSTEM] Access the web interface at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)