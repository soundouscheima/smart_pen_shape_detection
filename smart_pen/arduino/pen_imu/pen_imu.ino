/*
 * ============================================================
 *  SMART PEN - IMU Drawing Data Transmitter
 * ============================================================
 *  Board  : ESP32 (any WiFi-capable dev board)
 *  Sensor : MPU6050 (6-axis Accelerometer + Gyroscope, I2C)
 *
 *  What it does:
 *    The MPU6050 is mounted inside a pen body. While the user
 *    "draws" a shape in the air (or on paper), the IMU captures
 *    the hand motion and this sketch streams the raw readings
 *    over WiFi/TCP to the Python server (ser_4_realtime.py or
 *    app.py) as a CSV line.
 *
 *  CSV format sent to the server (port 5000):
 *    HandID, Timestamp_ms, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
 *
 *  Libraries needed (Arduino Library Manager):
 *    - WiFi          (bundled with ESP32 core)
 *    - Wire          (bundled with ESP32 core)
 *    - MPU6050       by Electronic Cats / Adafruit (class: MPU6050)
 *
 *  Wiring (I2C):
 *    MPU6050 VCC -> ESP32 3V3
 *    MPU6050 GND -> ESP32 GND
 *    MPU6050 SDA -> ESP32 GPIO21  (default SDA)
 *    MPU6050 SCL -> ESP32 GPIO22  (default SCL)
 * ============================================================
 */

#include <WiFi.h>
#include <Wire.h>
#include <MPU6050.h>

MPU6050 imu;

// ===== WiFi Config =====
const char* ssid     = "--";            // <-- your WiFi SSID
const char* password = "--";            // <-- your WiFi password

const char* host = "192.168.246.227";   // <-- your PC IPv4 address
const uint16_t port = 5000;             // Python server port

WiFiClient client;

// ===== IMU Config =====
const int HAND_ID = 1;   // Change this for left/right hand

void setup() {
  Serial.begin(115200);
  delay(1000);

  // ===== WiFi =====
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // ===== Connect to Server =====
  Serial.println("Connecting to Python server...");
  if (client.connect(host, port)) {
    Serial.println("Connected to server!");
  } else {
    Serial.println("FAILED to connect!");
  }

  // ===== IMU Initialization =====
  Wire.begin();
  imu.initialize();

  if (!imu.testConnection()) {
    Serial.println("IMU connection failed!");
  } else {
    Serial.println("IMU ready!");
  }
}

void loop() {
  if (!client.connected()) {
    Serial.println("Disconnected! Reconnecting...");
    client.connect(host, port);
    delay(1000);
    return;
  }

  // ===== Read IMU =====
  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  imu.getAcceleration(&ax, &ay, &az);
  imu.getRotation(&gx, &gy, &gz);

  unsigned long t = millis();

  // ===== Send in CSV Format =====
  String data = String(HAND_ID) + "," +
                String(t) + "," +
                String(ax) + "," +
                String(ay) + "," +
                String(az) + "," +
                String(gx) + "," +
                String(gy) + "," +
                String(gz);

  client.println(data);  // send to Python
  Serial.println("Sent: " + data);

  delay(50);  // 20Hz sending frequency
}
