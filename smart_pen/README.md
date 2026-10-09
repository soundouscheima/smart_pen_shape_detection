# Smart Pen — IMU Handwriting Shape & Quality Recognition

An end-to-end system that turns a **pen equipped with an IMU sensor** into a
"smart pen": as the user draws/writes a shape in the air (or on paper), the
sensor captures the hand motion, the data is streamed over WiFi to a Python
server, and a Machine Learning model predicts **which shape** was drawn and
**how good** the drawing quality is.

The project combines an embedded hardware prototype (ESP32 + MPU6050), a
real-time data-collection web app (Flask + TCP socket server), a signal
processing / feature extraction pipeline, and two trained SVM classifiers.

---

## 1. The Idea — "It's a Pen"

Drawing and handwriting are rich sources of motion data. When a human draws a
shape, the hand produces a characteristic pattern of acceleration and rotation
that is different for a square, a circle, a triangle, or a star — and also
different depending on how cleanly the shape was drawn.

This project captures that intuition with a **pen-shaped device**:

- An **MPU6050** 6-axis IMU (3-axis accelerometer + 3-axis gyroscope) is fitted
  inside a pen body (the "Arduino/ESP32 pen prototype").
- A **ESP32** microcontroller reads the IMU at ~20 Hz and streams the readings
  over WiFi to a PC.
- The PC records the stream and runs **two SVM classifiers**:
  1. **Shape Detector** — classifies the drawn shape (`square`, `circle`, `triangle`, `star`, `unknown`).
  2. **Quality Detector** — classifies the drawing quality (`good`, `medium`, `bad`, `unknown`).

The result is a pen that can understand *what* you drew and *how well* you drew it.

> Possible applications: handwriting/drawing tutoring for children, signature and
> gesture recognition, digitizing air-drawn shapes, tremor/rehabilitation
> monitoring, and human–computer interaction research.

---

## 2. Hardware Prototype

### 2.1 The Arduino / ESP32 Pen

The pen prototype is a hand-held (pen-shaped) enclosure hosting the electronics.
A battery powers the ESP32 while the IMU sits near the pen tip so it captures the
motion of the drawing stroke. Because the ESP32 has built-in WiFi, the pen is
fully wireless and sends samples directly to the Python server.

```
        ┌─────────────────────────────┐
        │        SMART PEN            │
        │                             │
        │   [MPU6050 IMU]  (tip)      │
        │        │ I2C                │
        │   [ESP32 MCU] ── WiFi ──►   │  TCP :5000
        │        │                    │
        │   [Li-ion battery]          │
        └─────────────────────────────┘
```

### 2.2 Components

| Component      | Role                                                        |
|----------------|-------------------------------------------------------------|
| **ESP32**      | Reads the IMU and streams data over WiFi/TCP                 |
| **MPU6050**    | 6-axis IMU — accelerometer (X, Y, Z) + gyroscope (X, Y, Z)  |
| **Li-ion cell**| Powers the wireless pen                                     |
| **Pen body**   | Housing / ergonomic enclosure for the electronics           |

### 2.3 Wiring (I2C)

| MPU6050 | ESP32        |
|---------|--------------|
| VCC     | 3V3          |
| GND     | GND          |
| SDA     | GPIO21       |
| SCL     | GPIO22       |

### 2.4 Firmware

The ESP32 firmware lives in [`arduino/pen_imu/pen_imu.ino`](arduino/pen_imu/pen_imu.ino).
It:

1. Connects to WiFi.
2. Opens a TCP connection to the Python server on port `5000`.
3. Reads accel + gyro from the MPU6050 every `50 ms` (20 Hz).
4. Sends one CSV line per sample:

```
HandID, Timestamp_ms, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
```

Before flashing, edit the top of the sketch:

```cpp
const char* ssid     = "--";            // your WiFi SSID
const char* password = "--";            // your WiFi password
const char* host     = "192.168.246.227";  // your PC IPv4 address
const uint16_t port  = 5000;            // Python server port
```

Libraries: `WiFi` and `Wire` ship with the ESP32 board package; install the
`MPU6050` library from the Arduino Library Manager.

---

## 3. System Architecture

```
  ┌────────────┐   I2C    ┌─────────┐   WiFi / TCP   ┌──────────────────────────┐
  │  MPU6050   │─────────►│  ESP32  │───────────────►│  Python server           │
  │ (in pen)   │          │ (pen)   │   port 5000    │  • TCP socket listener   │
  └────────────┘          └─────────┘                │  • Flask web app :5001   │
                                                     └────────────┬─────────────┘
                                                                  │
                              ┌───────────────────────────────────┼───────────────────────────┐
                              │                                   │                           │
                     ┌────────▼─────────┐              ┌──────────▼─────────┐        ┌─────────▼─────────┐
                     │ Data collection  │              │ Feature extraction │        │  CSV files        │
                     │ (start/stop req) │              │ 30 stats features  │        │  data/*.csv       │
                     └──────────────────┘              └──────────┬─────────┘        └───────────────────┘
                                                                  │
                                                     ┌────────────▼─────────────┐
                                                     │ StandardScaler +         │
                                                     │ SVM (shape) + SVM(qual.) │
                                                     └────────────┬─────────────┘
                                                                  │
                                                     ┌────────────▼─────────────┐
                                                     │  Predicted Shape +       │
                                                     │  Quality (web UI)        │
                                                     └──────────────────────────┘
```

Data never leaves the local network: the ESP32 talks directly to the PC over TCP.

---

## 4. Repository Structure

```
newfai/
├── README.md                        # this file
├── arduino/
│   └── pen_imu/
│       └── pen_imu.ino              # ESP32 + MPU6050 firmware (the smart pen)
│
├── app.py                          # Flask app: data collection only (socket :5000, web :5001)
├── ser_4_realtime.py               # Main app: data collection + real-time ML prediction
│
├── agmentaion.py                   # Preprocessing + augmentation + feature extraction
├── train.py                        # Train the two SVM classifiers + plots
├── tset.py                         # Standalone test on a single CSV file
├── testmodel.py                    # (empty placeholder)
│
├── templates/
│   ├── index.html                  # Data-collection web UI
│   ├── real.html                   # Real-time prediction web UI
│   ├── test.html                   # Combined drawing/prediction UI (experimental)
│   └── results.html                # Simple server-rendered result page
│
├── model_shape_detector.pkl        # Trained SVM — shape classifier
├── model_quality_detector.pkl      # Trained SVM — quality classifier
├── scaler.pkl                      # Fitted StandardScaler (must match training)
├── X_data.pkl, y_shape.pkl, y_quality.pkl   # Preprocessed training data
│
├── Confusion_Matrix_-_Shape_Detector.png
├── Confusion_Matrix_-_Quality_Detector.png
├── Learning_Curve_-_Shape_Detector.png
├── Learning_Curve_-_Quality_Detector.png
│
├── data/                           # Labelled recordings (shape_quality_timestamp.csv)
│   └── uploads/                    # CSVs uploaded through the web UI
├── data_in/                        # Misc. input recordings
├── recordings/                     # Raw recordings
└── temp_data/                      # Temporary prediction recordings
```

> **Note:** `data/`, `data_in/`, `recordings/`, `temp_data/`, the `*.pkl` models,
> and the training scripts are treated as locked artifacts and are not modified
> here.

---

## 5. Data Format

### 5.1 Wire format (from the pen)

One CSV line per sample, no header:

```
HandID, Timestamp_ms, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
```

### 5.2 Stored format (server-side CSV)

When recording, the server augments each sample with the label and metadata:

```
HandID, Timestamp_ms, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z, Shape, Quality, Recording_Start_Time
```

Example (`data/circle_medium_20260113_110227.csv`):

```csv
HandID,Timestamp_ms,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z,Shape,Quality,Recording_Start_Time
1,398773,4596,-168,-16080,-116,166,61,circle,medium,2026-01-13T11:02:27.228269
1,398826,4596,-120,-16052,-99,168,30,circle,medium,2026-01-13T11:02:27.228269
```

The **label is encoded in the filename** as `<Shape>_<Quality>_<timestamp>.csv`,
which is how the preprocessing script derives the ground truth.

---

## 6. Machine Learning Pipeline

The ML pipeline has three stages: **collect → preprocess/augment → train**.

### 6.1 Feature Extraction (`agmentaion.py`, `tset.py`, `ser_4_realtime.py`)

Only the six sensor columns are used — `HandID`, timestamps, and labels are
**never** fed to the model (this prevents label leakage). For each of the six
axes the following **5 statistical features** are computed:

1. Mean
2. Standard deviation
3. Maximum
4. Minimum
5. RMS (`sqrt(mean(x^2))`)

That is `6 axes × 5 stats = 30 features` per recording.

### 6.2 Augmentation (`agmentaion.py`)

To increase the effective dataset size, each recording is augmented with two
transforms before features are extracted:

- **Jitter** — add Gaussian noise (`sigma = 0.05`) to raw samples.
- **Scaling** — multiply each axis by a random factor (`sigma = 0.1`).

Each original recording therefore yields 3 feature rows (original, jittered,
scaled), all sharing the same shape/quality labels.

### 6.3 Training (`train.py`)

- Splits data 80% train / 20% test.
- Trains two **SVM classifiers** with an RBF kernel (`C=10`, `gamma='scale'`).
- Runs **5-fold cross-validation**, reports test accuracy and a full
  precision/recall/F1 classification report.
- Saves a **learning curve** (over/under-fitting check) and a **confusion
  matrix** heatmap as PNGs.
- Persists the models as `model_shape_detector.pkl` and
  `model_quality_detector.pkl`.

### 6.4 Inference (`ser_4_realtime.py`, `tset.py`)

New recordings are cleaned (non-numeric → NaN → dropped), the same 30 features
are extracted, the saved `scaler.pkl` is applied, and both SVMs return a
prediction.

---

## 7. Software Components

| File | Purpose |
|------|---------|
| `app.py` | Flask **data-collection** app. A background TCP socket server listens on `0.0.0.0:5000`; the web UI runs on port `5001`. Lets the user label a recording (shape + quality), start/stop recording, and writes labelled CSVs to `data/`. |
| `ser_4_realtime.py` | The **main application**. Same socket server + collection flow as `app.py`, but also loads the SVM models and scaler and exposes prediction endpoints (`/predict_last`, `/predict_upload`) and the `/real` UI. |
| `agmentaion.py` | Preprocessing + augmentation. Reads all `*.csv` from a data folder, extracts 30 features, saves `X_data.pkl`, `y_shape.pkl`, `y_quality.pkl`, and fits/saves `scaler.pkl`. |
| `train.py` | Trains both SVM models, generates evaluation plots, and saves the `.pkl` models. |
| `tset.py` | Quick sanity check — runs a prediction on a single hard-coded CSV file. |
| `testmodel.py` | Empty placeholder. |
| `arduino/pen_imu/pen_imu.ino` | ESP32 firmware for the smart pen (WiFi + MPU6050 → TCP CSV). |
| `templates/index.html` | Collection dashboard: connection/recording status, shape & quality selectors, start/stop/save. |
| `templates/real.html` | Prediction dashboard: start/stop recording, predict-last, upload-and-predict, animated result card. |
| `templates/test.html` | Larger experimental combined drawing/prediction UI. |
| `templates/results.html` | Minimal server-rendered result page. |

### HTTP / network endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Collection UI (`index.html`) |
| `/real` | GET | Real-time prediction UI (`real.html`) |
| `/start_recording` | POST | Begin recording (JSON: `shape`, `quality`) |
| `/stop_recording` | POST | Stop recording, returns point count |
| `/get_status` | GET | Connection / recording / model status |
| `/get_recent_data` | GET | Last few data points for live display |
| `/save_data` | POST | Confirm/describe the saved file |
| `/download/<filename>` | GET | Download a recorded CSV |
| `/list_files` | GET | List recorded CSVs (`ser_4_realtime.py`) |
| `/predict_last` | POST | Predict from the most recent recording |
| `/predict_upload` | POST | Predict from an uploaded CSV |
| `/test_connection` | GET | Health check |
| `tcp://<pc-ip>:5000` | TCP | Raw sensor stream from the pen |

---

## 8. Getting Started

### 8.1 Prerequisites

- Python 3.8+ with: `flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
  `seaborn`.
- Arduino IDE with the **ESP32 board package** and the **MPU6050** library.

Install the Python dependencies:

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### 8.2 Flash the pen (ESP32)

1. Open `arduino/pen_imu/pen_imu.ino` in the Arduino IDE.
2. Set `ssid`, `password`, and `host` (your PC's IPv4 address) — keep `port = 5000`.
3. Select your ESP32 board and upload.
4. Confirm the Serial Monitor shows `WiFi connected!`, `Connected to server!`, `IMU ready!`.

> Find your PC IPv4 with `ipconfig` (Windows). The ESP32 and PC must be on the
> same WiFi network.

### 8.3 Run the server

For **collection + live prediction**:

```bash
python ser_4_realtime.py
```

Then open:

- Collection UI: <http://localhost:5001/>
- Prediction UI: <http://localhost:5001/real>

For **collection only**, use `python app.py` (web UI on port `5001`).

### 8.4 Record a drawing

1. Wait until the UI shows the pen **Connected**.
2. Pick a **shape** and a **quality** label.
3. Click **Start Recording**, draw the shape with the pen, then **Stop Recording**.
4. The recording is saved to `data/<shape>_<quality>_<timestamp>.csv`.

### 8.5 Train / retrain the models

```bash
# 1. Preprocess + augment (edit DATA_PATH in agmentaion.py to point at your data folder)
python agmentaion.py

# 2. Train both SVMs and produce plots
python train.py

# 3. Optional: quick single-file sanity check (edit TEST_FILES in tset.py)
python tset.py
```

### 8.6 Predict

Use the **/real** page (predict last recording / upload a CSV), or run `tset.py`.
The server loads `model_shape_detector.pkl`, `model_quality_detector.pkl`, and
`scaler.pkl` at startup.

---

## 9. Notes, Caveats & Known Issues

- **Hard-coded paths:** `agmentaion.py` uses an absolute `DATA_PATH` and `tset.py`
  has a hard-coded `TEST_FILES` list — update both for your machine.
- **Model/scaler must match:** always use the `scaler.pkl` produced by the same
  preprocessing run that trained the models. A scaler/model mismatch silently
  degrades accuracy.
- **scikit-learn version:** `.pkl` models are version-sensitive; unpickle them
  with a compatible scikit-learn version.
- **`best_model_*.pkl`:** the runtime prefers `best_model_shape_detector.pkl` /
  `best_model_quality_detector.pkl` if present, otherwise falls back to the
  standard `model_*.pkl` files.
- **Network config:** the ESP32 `host` must be the PC's current IPv4; it changes
  with the network. `ssid`, `password`, and `host` in the `.ino` still contain
  placeholders and must be filled in before flashing.
- **Sample rate:** the firmware sends at 20 Hz (`delay(50)`); recordings shorter
  than ~10 samples are rejected by the feature extractor.
- **Real-time display** keeps only the last ~50 samples in memory.

---

## 10. Quick Reference — Record → Predict

```
ESP32 pen ──TCP:5000──► Python server ──► data/<shape>_<quality>_*.csv
                                              │
                 agmentaion.py  ──►  X_data.pkl, y_*.pkl, scaler.pkl
                                              │
                       train.py  ──►  model_*.pkl (+ plots)
                                              │
        /real page  or  tset.py  ──►  predicted Shape + Quality
```
