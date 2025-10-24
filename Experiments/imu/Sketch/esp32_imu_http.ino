#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>

// ===============================
// WiFi Config
// ===============================
const char* ssid = "YOUR_WIFI_SSID";          // 🔹 Replace with your WiFi SSID
const char* password = "YOUR_WIFI_PASSWORD";  // 🔹 Replace with your WiFi password
const char* serverName = "http://<your-laptop-ip>:3237/imu"; // 🔹 Replace with your laptop IP

// ===============================
// Pin Config
// ===============================
const int buttonPin = 14;
const int buzzerPin = 26;
const int ledPin = 13;

// ===============================
// Debounce & State
// ===============================
const unsigned long debounceDelay = 80;
int stableState = HIGH;
int lastReading = HIGH;
unsigned long lastDebounceTime = 0;

bool buzzerOn = false;
bool isRecording = false;
int actionId = 0;
unsigned long recordStartTime = 0;

// ===============================
// MPU6050 & WiFi Objects
// ===============================
Adafruit_MPU6050 mpu;

// ===============================
// WiFi Setup
// ===============================
void setup_wifi() {
  Serial.print("[WiFi] Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  unsigned long startAttempt = millis();

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
    if (millis() - startAttempt > 20000) {
      Serial.println("\n[WiFi] Failed to connect. Rebooting...");
      ESP.restart();
    }
  }

  Serial.println("\n[WiFi] Connected");
  Serial.print("[WiFi] IP: ");
  Serial.println(WiFi.localIP());
}

// ===============================
// IMU Setup
// ===============================
void setupIMU() {
  Wire.begin(21, 22);  // SDA, SCL (
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERROR: MPU6050 connection failed!");
    while (1) {
      tone(buzzerPin, 2000, 200);
      delay(500);
    }
  }

  Serial.println("[IMU] MPU6050 initialized");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  delay(100);
}

// ===============================
// HTTP Send Function
// ===============================
void sendIMUDataHTTP(float accelX, float accelY, float accelZ,
                     float gyroX, float gyroY, float gyroZ,
                     unsigned long timestamp, int actionId) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected, skipping send");
    return;
  }

  StaticJsonDocument<256> doc;
  doc["timestamp"] = timestamp;
  doc["accel_x"] = accelX;
  doc["accel_y"] = accelY;
  doc["accel_z"] = accelZ;
  doc["gyro_x"] = gyroX;
  doc["gyro_y"] = gyroY;
  doc["gyro_z"] = gyroZ;
  doc["action_id"] = actionId; 

  String jsonString;
  serializeJson(doc, jsonString);

  HTTPClient http;
  http.begin(serverName);
  http.addHeader("Content-Type", "application/json");

  int httpResponseCode = http.POST(jsonString);

  if (httpResponseCode > 0) {
    Serial.println("[HTTP] Sent: " + jsonString);
  } else {
    Serial.println("[HTTP] Error sending: " + String(httpResponseCode));
  }

  http.end();
}

// ===============================
// Setup
// ===============================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("========= ESP32 IMU HTTP Publisher =========");

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  setupIMU();
  setup_wifi();

  Serial.println("[SYSTEM] Ready - Press and hold button to record");
}

// ===============================
// Main Loop
// ===============================
void loop() {
  int reading = digitalRead(buttonPin);

  if (reading != lastReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != stableState) {
      stableState = reading;

      if (stableState == LOW && !buzzerOn) {
        Serial.println("STATUS:Recording started");
        digitalWrite(ledPin, HIGH);
        tone(buzzerPin, 1000, 120);
        buzzerOn = true;
        isRecording = true;
        actionId++;
        recordStartTime = millis();

      } else if (stableState == HIGH && buzzerOn) {
        Serial.print("STATUS:Recording stopped - Action ID ");
        Serial.println(actionId);
        digitalWrite(ledPin, LOW);
        tone(buzzerPin, 300, 80);
        buzzerOn = false;
        isRecording = false;
      }
    }
  }

  if (isRecording) {
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

    unsigned long timestamp = millis() - recordStartTime;

    sendIMUDataHTTP(
      accel.acceleration.x,
      accel.acceleration.y,
      accel.acceleration.z,
      gyro.gyro.x,
      gyro.gyro.y,
      gyro.gyro.z,
      timestamp,
      actionId
    );

    delay(10); // ~100Hz sample rate 
  }

  lastReading = reading;
}
