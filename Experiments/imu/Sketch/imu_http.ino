// #include <WiFi.h>
// #include <HTTPClient.h>
// #include <Wire.h>
// #include <Adafruit_MPU6050.h>
// #include <Adafruit_Sensor.h>
// #include <math.h>

// // ====== CONFIG ======
// const char* WIFI_SSID = "";
// const char* WIFI_PASS = "";

// String SERVER_URL = "http://<ip_address>:5000/predict?log=1&session=esp32";

// const char* SERVER_HOST = "<ip_address>";
// const int SERVER_PORT = 5000;
// String SERVER_PREDICT_URL =
//     String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/predict?log=1&session=esp32";
// String SERVER_PING_URL =
//     String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/ping";

// // IMU: Adafruit MPU6050
// Adafruit_MPU6050 mpu;

// // Streaming params
// const int WINDOW = 244;          // must match your model window
// const int SAMPLE_HZ = 50;        // sensor sample rate
// const int POST_EVERY_MS = 200;   // send a window every 200 ms
// // =====================

// struct Sample { float gx, gy, gz, ax, ay, az; };
// Sample ring[WINDOW];
// int ringPos = 0;
// int sampleCount = 0;

// unsigned long lastSampleMs = 0;
// unsigned long lastPostMs = 0;

// // gyro bias (computed at startup)
// float gyro_bias_x = 0.0f, gyro_bias_y = 0.0f, gyro_bias_z = 0.0f;
// // axis sign adjustment if needed (1 or -1)
// const int AXIS_SIGN_GX = 1;
// const int AXIS_SIGN_GY = 1;
// const int AXIS_SIGN_GZ = 1;

// // default quick mapping (used in normal buildWindowJSON)
// const int MAP_GYRO_IDX[3] = {0, 2, 1}; // maps output [gx,gy,gz] = input [g[0], g[2], g[1]]

// // ===== DEBUG / MAPPING TEST =====
// // Set to true to cycle through candidate mappings and POST each (prints payload + server resp)
// const bool TEST_MAPPING_MODE = true;

// // Candidate mappings to try (top-3 from find_best_mapping)
// struct Mapping {
//   int gperm[3];
//   int gsign[3];
//   int aperm[3];
//   int asign[3];
// };

// // Populate candidates (adjust/add as needed)
// Mapping candidates[] = {
//   // candidate 0: gyro perm (0,2,1), all signs +, accel identity
//   {{0,2,1}, {1,1,1}, {0,1,2}, {1,1,1}},
//   // candidate 1: same perm, flip gx sign
//   {{0,2,1}, {-1,1,1}, {0,1,2}, {1,1,1}},
//   // candidate 2: same perm, flip az sign
//   {{0,2,1}, {1,1,1}, {0,1,2}, {1,1,-1}}
// };
// const int NUM_CANDIDATES = sizeof(candidates)/sizeof(candidates[0]);
// // ===== END DEBUG / MAPPING TEST =====

// bool calibrateGyroBias(unsigned int samples = 200, unsigned int sample_ms = 10) {
//   float bx=0, by=0, bz=0;
//   sensors_event_t a, g, temp;
//   delay(50);
//   for (unsigned int i=0;i<samples;i++) {
//     mpu.getEvent(&a, &g, &temp);
//     bx += g.gyro.x;
//     by += g.gyro.y;
//     bz += g.gyro.z;
//     delay(sample_ms);
//   }
//   gyro_bias_x = bx / (float)samples;
//   gyro_bias_y = by / (float)samples;
//   gyro_bias_z = bz / (float)samples;
//   Serial.printf("GYRO_BIAS: gx=%.6f gy=%.6f gz=%.6f (rad/s)\n", gyro_bias_x, gyro_bias_y, gyro_bias_z);
//   return true;
// }

// bool readIMU(Sample &s) {
//   // Use Adafruit MPU6050 API so units match training:
//   // - g.gyro.* returns radians/sec
//   // - a.acceleration.* returns m/s^2
//   sensors_event_t a, g, temp;
//   mpu.getEvent(&a, &g, &temp);

//   // assign and apply bias + axis sign
//   s.gx = (g.gyro.x - gyro_bias_x) * AXIS_SIGN_GX;
//   s.gy = (g.gyro.y - gyro_bias_y) * AXIS_SIGN_GY;
//   s.gz = (g.gyro.z - gyro_bias_z) * AXIS_SIGN_GZ;

//   s.ax = a.acceleration.x;
//   s.ay = a.acceleration.y;
//   s.az = a.acceleration.z;

//   // DEBUG print (concise)
//   Serial.printf("IMU ax=%.3f ay=%.3f az=%.3f gx=%.3f gy=%.3f gz=%.3f\n",
//                 s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
//   return true;
// }

// void appendSample(const Sample &s) {
//   ring[ringPos] = s;
//   ringPos = (ringPos + 1) % WINDOW;
//   if (sampleCount < WINDOW) sampleCount++;
// }

// void buildWindowJSON(String &out) {
//   out.reserve(12000);
//   out = "{\"frames\":[";
//   int start = (sampleCount < WINDOW) ? 0 : ringPos;
//   int count = (sampleCount < WINDOW) ? sampleCount : WINDOW;
//   for (int i = 0; i < count; i++) {
//     int idx = (start + i) % WINDOW;
//     const Sample &s = ring[idx];

//     // apply mapping: create array of input gyros and accels then pick mapped order
//     float g_in[3] = { s.gx, s.gy, s.gz };
//     float a_in[3] = { s.ax, s.ay, s.az };
//     // mapped gyro: out_gx = g_in[ MAP_GYRO_IDX[0] ], out_gy = g_in[ MAP_GYRO_IDX[1] ], ...
//     float out_gx = g_in[ MAP_GYRO_IDX[0] ];
//     float out_gy = g_in[ MAP_GYRO_IDX[1] ];
//     float out_gz = g_in[ MAP_GYRO_IDX[2] ];
//     // accel unchanged
//     float out_ax = a_in[0];
//     float out_ay = a_in[1];
//     float out_az = a_in[2];

//     out += "[";
//     out += String(out_gx, 5) + "," + String(out_gy, 5) + "," + String(out_gz, 5) + ",";
//     out += String(out_ax, 5) + "," + String(out_ay, 5) + "," + String(out_az, 5);
//     out += "]";
//     if (i != count - 1) out += ",";
//   }
//   out += "]}";
// }

// // Helper: build JSON using an explicit mapping
// String buildWindowJSONWithMapping(const Mapping &m) {
//   String out;
//   out.reserve(12000);
//   out = "{\"frames\":[";
//   int start = (sampleCount < WINDOW) ? 0 : ringPos;
//   int count = (sampleCount < WINDOW) ? sampleCount : WINDOW;
//   for (int i = 0; i < count; i++) {
//     int idx = (start + i) % WINDOW;
//     const Sample &s = ring[idx];
//     float g_in[3] = { (s.gx) * AXIS_SIGN_GX, (s.gy) * AXIS_SIGN_GY, (s.gz) * AXIS_SIGN_GZ };
//     float a_in[3] = { s.ax, s.ay, s.az };
//     // apply perm & sign
//     float out_gx = g_in[m.gperm[0]] * m.gsign[0];
//     float out_gy = g_in[m.gperm[1]] * m.gsign[1];
//     float out_gz = g_in[m.gperm[2]] * m.gsign[2];
//     float out_ax = a_in[m.aperm[0]] * m.asign[0];
//     float out_ay = a_in[m.aperm[1]] * m.asign[1];
//     float out_az = a_in[m.aperm[2]] * m.asign[2];
//     out += "[";
//     out += String(out_gx, 5) + "," + String(out_gy, 5) + "," + String(out_gz, 5) + ",";
//     out += String(out_ax, 5) + "," + String(out_ay, 5) + "," + String(out_az, 5);
//     out += "]";
//     if (i != count - 1) out += ",";
//   }
//   out += "]}";
//   return out;
// }

// void computeWindowStats(float mean_out[6], float std_out[6]) {
//   int start = (sampleCount < WINDOW) ? 0 : ringPos;
//   int count = (sampleCount < WINDOW) ? sampleCount : WINDOW;
//   // init
//   for (int k=0;k<6;k++){ mean_out[k]=0; std_out[k]=0; }
//   // accumulate means
//   for (int i=0;i<count;i++){
//     int idx = (start + i) % WINDOW;
//     Sample &s = ring[idx];
//     float vals[6] = {s.gx, s.gy, s.gz, s.ax, s.ay, s.az};
//     for (int k=0;k<6;k++) mean_out[k] += vals[k];
//   }
//   for (int k=0;k<6;k++) mean_out[k] /= (float)count;
//   // accumulate std
//   for (int i=0;i<count;i++){
//     int idx = (start + i) % WINDOW;
//     Sample &s = ring[idx];
//     float vals[6] = {s.gx, s.gy, s.gz, s.ax, s.ay, s.az};
//     for (int k=0;k<6;k++){
//       float d = vals[k] - mean_out[k];
//       std_out[k] += d*d;
//     }
//   }
//   for (int k=0;k<6;k++) std_out[k] = sqrt(std_out[k]/(float)count);
// }

// bool windowIsDynamic(float dyn_thresh = 0.7f) {
//   float mean[6], stdev[6];
//   computeWindowStats(mean, stdev);
//   float max_ang = fabs(mean[0]) + fabs(mean[1]) + fabs(mean[2]);
//   // print stats every now and then for debugging
//   static unsigned long lastPrint = 0;
//   if (millis() - lastPrint > 2000) {
//     lastPrint = millis();
//     Serial.printf("WINDOW_MEANS gx=%.3f gy=%.3f gz=%.3f ax=%.3f ay=%.3f az=%.3f\n",
//                   mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]);
//     Serial.printf("WINDOW_STD   gx=%.3f gy=%.3f gz=%.3f ax=%.3f ay=%.3f az=%.3f\n",
//                   stdev[0], stdev[1], stdev[2], stdev[3], stdev[4], stdev[5]);
//   }
//   return max_ang > dyn_thresh;
// }

// void postWindow() {
//   if (sampleCount < WINDOW) return;
//   if (!windowIsDynamic(0.5f)) { // tweak threshold
//     Serial.println("Window static — skip POST");
//     return;
//   }

//   // Normal behavior: build mapped payload and print it (so you can copy/save it)
//   if (!TEST_MAPPING_MODE) {
//     HTTPClient http;
//     http.begin(SERVER_PREDICT_URL);
//     http.addHeader("Content-Type", "application/json");
//     String payload;
//     buildWindowJSON(payload);
//     // Print exact JSON to Serial BEFORE POST so you can save it as live_window.json
//     Serial.println("JSON_PAYLOAD_START");
//     Serial.println(payload);
//     Serial.println("JSON_PAYLOAD_END");
//     int code = http.POST((uint8_t*)payload.c_str(), payload.length());
//     if (code > 0) {
//       String resp = http.getString();
//       Serial.printf("POST %d, bytes=%d, resp=%s\n", code, payload.length(), resp.c_str());
//     } else {
//       Serial.printf("POST failed, code=%d\n", code);
//     }
//     http.end();
//     return;
//   }

//   // TEST_MAPPING_MODE: cycle candidate mappings and POST each one (debug only)
//   for (int ci = 0; ci < NUM_CANDIDATES; ++ci) {
//     const Mapping &m = candidates[ci];
//     String payload = buildWindowJSONWithMapping(m);
//     // Print marker and payload so you can save the exact JSON
//     Serial.printf("MAPPING_CANDIDATE %d START\n", ci);
//     Serial.println(payload); // copy this JSON to live_window.json for offline analysis
//     Serial.printf("MAPPING_CANDIDATE %d END\n", ci);

//     HTTPClient http;
//     http.begin(SERVER_PREDICT_URL);
//     http.addHeader("Content-Type", "application/json");
//     int code = http.POST((uint8_t*)payload.c_str(), payload.length());
//     if (code > 0) {
//       String resp = http.getString();
//       Serial.printf("MAPPING %d -> POST %d, resp=%s\n", ci, code, resp.c_str());
//     } else {
//       Serial.printf("MAPPING %d -> POST failed, code=%d\n", ci, code);
//     }
//     http.end();
//     delay(200); // brief pause between candidate posts
//   }
// }

// bool testPing(const char *pingUrlFull) {
//   HTTPClient http;
//   http.begin(pingUrlFull);
//   http.setTimeout(5000);
//   int code = http.GET();
//   Serial.printf("PING %s -> code=%d\n", pingUrlFull, code);
//   if (code > 0) {
//     String body = http.getString();
//     Serial.printf("PING body: %s\n", body.c_str());
//   }
//   http.end();
//   return (code > 0);
// }

// bool testSmallPost(const char *predictUrlFull) {
//   HTTPClient http;
//   http.begin(predictUrlFull);
//   http.addHeader("Content-Type", "application/json");
//   http.setTimeout(5000);
//   String small = "{\"frames\":[[0,0,0,0,0,0]]}";
//   int code = http.POST(small);
//   String resp = (code > 0) ? http.getString() : String();
//   Serial.printf("POST test -> code=%d, resp=%s\n", code, resp.c_str());
//   http.end();
//   return (code > 0);
// }

// void setup() {
//   Serial.begin(115200);
//   Wire.begin(); // default SDA/SCL

//   // initialize Adafruit MPU (match imu_data_collect.ino settings)
//   Serial.println("INIT: Initializing MPU6050...");
//   if (!mpu.begin(0x68)) {
//     Serial.println("ERROR: MPU6050 not found");
//     while (1) delay(10);
//   }
//   // set same ranges used during data collection
//   mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
//   mpu.setGyroRange(MPU6050_RANGE_500_DEG);
//   mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
//   Serial.println("INIT: MPU6050 ready (Adafruit lib)");

//   // calibrate gyro bias (keep device stationary during boot)
//   Serial.println("CALIBRATE: computing gyro bias, keep IMU stationary...");
//   calibrateGyroBias(200, 10);

//   WiFi.mode(WIFI_STA);
//   WiFi.begin(WIFI_SSID, WIFI_PASS);
//   Serial.printf("Connecting to WiFi %s", WIFI_SSID);
//   int tries = 0;
//   while (WiFi.status() != WL_CONNECTED && tries < 60) {
//     delay(500); Serial.print(".");
//     tries++;
//   }
//   Serial.println();
//   if (WiFi.status() == WL_CONNECTED) {
//     Serial.print("WiFi OK, IP: "); Serial.println(WiFi.localIP());
//     testPing(SERVER_PING_URL.c_str());
//     testSmallPost(SERVER_PREDICT_URL.c_str());
//   } else {
//     Serial.println("WiFi failed, will continue and retry later...");
//   }

//   lastSampleMs = millis();
//   lastPostMs = millis();
// }

// void loop() {
//   unsigned long now = millis();
//   unsigned long samplePeriodMs = 1000UL / SAMPLE_HZ;
//   if (now - lastSampleMs >= samplePeriodMs) {
//     lastSampleMs = now;
//     Sample s;
//     if (readIMU(s)) {
//       appendSample(s);
//     }

//     // dynamic detector in rad/s (model expects rad/s)
//     float ang_mag = fabs(s.gx) + fabs(s.gy) + fabs(s.gz);
//     const float DYN_THRESH = 0.7f; // ~40 deg/s = 0.7 rad/s, adjust if needed
//     if (ang_mag > DYN_THRESH) {
//       Serial.printf("DYNAMIC ang=%.3f gx=%.3f gy=%.3f gz=%.3f\n", ang_mag, s.gx, s.gy, s.gz);
//     } else {
//       Serial.printf("STATIC  ang=%.3f gx=%.3f gy=%.3f gz=%.3f\n", ang_mag, s.gx, s.gy, s.gz);
//     }
//   }

//   if (now - lastPostMs >= POST_EVERY_MS) {
//     lastPostMs = now;
//     if (WiFi.status() != WL_CONNECTED) {
//       WiFi.reconnect();
//     } else {
//       postWindow();
//     }
//   }

//   delay(1);
// }

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ====== CONFIG ======
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";

String SERVER_URL = "http://<ip_address>:5000/predict?log=1&session=esp32";

const char* SERVER_HOST = "<ip_address>";
const int SERVER_PORT = 5000;
String SERVER_PREDICT_URL =
    String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/predict?log=1&session=esp32";
String SERVER_PING_URL =
    String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/ping";

// IMU: Adafruit MPU6050
Adafruit_MPU6050 mpu;

// Streaming params
const int WINDOW = 244;          // must match your model window
const int SAMPLE_HZ = 50;        // sensor sample rate
const int POST_EVERY_MS = 200;   // send a window every 200 ms
// =====================

struct Sample { float gx, gy, gz, ax, ay, az; };
Sample ring[WINDOW];
int ringPos = 0;
int sampleCount = 0;

unsigned long lastSampleMs = 0;
unsigned long lastPostMs = 0;

// gyro bias (computed at startup)
float gyro_bias_x = 0.0f, gyro_bias_y = 0.0f, gyro_bias_z = 0.0f;
// axis sign adjustment if needed (1 or -1)
const int AXIS_SIGN_GX = 1;
const int AXIS_SIGN_GY = 1;
const int AXIS_SIGN_GZ = 1;

// default quick mapping (used in normal buildWindowJSON)
const int MAP_GYRO_IDX[3] = {0, 2, 1}; // maps output [gx,gy,gz] = input [g[0], g[2], g[1]]

// ===== DEBUG / MAPPING TEST =====
// Set to true to cycle through candidate mappings and POST each (prints payload + server resp)
const bool TEST_MAPPING_MODE = true;

// Candidate mappings to try (top-3 from find_best_mapping)
struct Mapping {
  int gperm[3];
  int gsign[3];
  int aperm[3];
  int asign[3];
};

// Populate candidates (adjust/add as needed)
Mapping candidates[] = {
  // candidate 0: gyro perm (0,2,1), all signs +, accel identity
  {{0,2,1}, {1,1,1}, {0,1,2}, {1,1,1}},
  // candidate 1: same perm, flip gx sign
  {{0,2,1}, {-1,1,1}, {0,1,2}, {1,1,1}},
  // candidate 2: same perm, flip az sign
  {{0,2,1}, {1,1,1}, {0,1,2}, {1,1,-1}}
};
const int NUM_CANDIDATES = sizeof(candidates)/sizeof(candidates[0]);
// ===== END DEBUG / MAPPING TEST =====

bool calibrateGyroBias(unsigned int samples = 200, unsigned int sample_ms = 10) {
  float bx=0, by=0, bz=0;
  sensors_event_t a, g, temp;
  delay(50);
  for (unsigned int i=0;i<samples;i++) {
    mpu.getEvent(&a, &g, &temp);
    bx += g.gyro.x;
    by += g.gyro.y;
    bz += g.gyro.z;
    delay(sample_ms);
  }
  gyro_bias_x = bx / (float)samples;
  gyro_bias_y = by / (float)samples;
  gyro_bias_z = bz / (float)samples;
  Serial.printf("GYRO_BIAS: gx=%.6f gy=%.6f gz=%.6f (rad/s)\n", gyro_bias_x, gyro_bias_y, gyro_bias_z);
  return true;
}

bool readIMU(Sample &s) {
  // Use Adafruit MPU6050 API so units match training:
  // - g.gyro.* returns radians/sec
  // - a.acceleration.* returns m/s^2
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // assign and apply bias + axis sign
  s.gx = (g.gyro.x - gyro_bias_x) * AXIS_SIGN_GX;
  s.gy = (g.gyro.y - gyro_bias_y) * AXIS_SIGN_GY;
  s.gz = (g.gyro.z - gyro_bias_z) * AXIS_SIGN_GZ;

  s.ax = a.acceleration.x;
  s.ay = a.acceleration.y;
  s.az = a.acceleration.z;

  // DEBUG print (concise)
  Serial.printf("IMU ax=%.3f ay=%.3f az=%.3f gx=%.3f gy=%.3f gz=%.3f\n",
                s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
  return true;
}

void appendSample(const Sample &s) {
  ring[ringPos] = s;
  ringPos = (ringPos + 1) % WINDOW;
  if (sampleCount < WINDOW) sampleCount++;
}

void buildWindowJSON(String &out) {
  out.reserve(12000);
  out = "{\"frames\":[";
  int start = (sampleCount < WINDOW) ? 0 : ringPos;
  int count = (sampleCount < WINDOW) ? sampleCount : WINDOW;
  for (int i = 0; i < count; i++) {
    int idx = (start + i) % WINDOW;
    const Sample &s = ring[idx];

    // apply mapping: create array of input gyros and accels then pick mapped order
    float g_in[3] = { s.gx, s.gy, s.gz };
    float a_in[3] = { s.ax, s.ay, s.az };
    // mapped gyro: out_gx = g_in[ MAP_GYRO_IDX[0] ], out_gy = g_in[ MAP_GYRO_IDX[1] ], ...
    float out_gx = g_in[ MAP_GYRO_IDX[0] ];
    float out_gy = g_in[ MAP_GYRO_IDX[1] ];
    float out_gz = g_in[ MAP_GYRO_IDX[2] ];
    // accel unchanged
    float out_ax = a_in[0];
    float out_ay = a_in[1];
    float out_az = a_in[2];

    out += "[";
    out += String(out_gx, 5) + "," + String(out_gy, 5) + "," + String(out_gz, 5) + ",";
    out += String(out_ax, 5) + "," + String(out_ay, 5) + "," + String(out_az, 5);
    out += "]";
    if (i != count - 1) out += ",";
  }
  out += "]}";
}

// Helper: build JSON using an explicit mapping
String buildWindowJSONWithMapping(const Mapping &m) {
  String out;
  out.reserve(12000);
  out = "{\"frames\":[";
  int start = (sampleCount < WINDOW) ? 0 : ringPos;
  int count = (sampleCount < WINDOW) ? sampleCount : WINDOW;
  for (int i = 0; i < count; i++) {
    int idx = (start + i) % WINDOW;
    const Sample &s = ring[idx];
    float g_in[3] = { (s.gx) * AXIS_SIGN_GX, (s.gy) * AXIS_SIGN_GY, (s.gz) * AXIS_SIGN_GZ };
    float a_in[3] = { s.ax, s.ay, s.az };
    // apply perm & sign
    float out_gx = g_in[m.gperm[0]] * m.gsign[0];
    float out_gy = g_in[m.gperm[1]] * m.gsign[1];
    float out_gz = g_in[m.gperm[2]] * m.gsign[2];
    float out_ax = a_in[m.aperm[0]] * m.asign[0];
    float out_ay = a_in[m.aperm[1]] * m.asign[1];
    float out_az = a_in[m.aperm[2]] * m.asign[2];
    out += "[";
    out += String(out_gx, 5) + "," + String(out_gy, 5) + "," + String(out_gz, 5) + ",";
    out += String(out_ax, 5) + "," + String(out_ay, 5) + "," + String(out_az, 5);
    out += "]";
    if (i != count - 1) out += ",";
  }
  out += "]}";
  return out;
}

void computeWindowStats(float mean_out[6], float std_out[6]) {
  int start = (sampleCount < WINDOW) ? 0 : ringPos;
  int count = (sampleCount < WINDOW) ? sampleCount : WINDOW;
  // init
  for (int k=0;k<6;k++){ mean_out[k]=0; std_out[k]=0; }
  // accumulate means
  for (int i=0;i<count;i++){
    int idx = (start + i) % WINDOW;
    Sample &s = ring[idx];
    float vals[6] = {s.gx, s.gy, s.gz, s.ax, s.ay, s.az};
    for (int k=0;k<6;k++) mean_out[k] += vals[k];
  }
  for (int k=0;k<6;k++) mean_out[k] /= (float)count;
  // accumulate std
  for (int i=0;i<count;i++){
    int idx = (start + i) % WINDOW;
    Sample &s = ring[idx];
    float vals[6] = {s.gx, s.gy, s.gz, s.ax, s.ay, s.az};
    for (int k=0;k<6;k++){
      float d = vals[k] - mean_out[k];
      std_out[k] += d*d;
    }
  }
  for (int k=0;k<6;k++) std_out[k] = sqrt(std_out[k]/(float)count);
}

bool windowIsDynamic(float dyn_thresh = 0.7f) {
  float mean[6], stdev[6];
  computeWindowStats(mean, stdev);
  float max_ang = fabs(mean[0]) + fabs(mean[1]) + fabs(mean[2]);
  // print stats every now and then for debugging
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 2000) {
    lastPrint = millis();
    Serial.printf("WINDOW_MEANS gx=%.3f gy=%.3f gz=%.3f ax=%.3f ay=%.3f az=%.3f\n",
                  mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]);
    Serial.printf("WINDOW_STD   gx=%.3f gy=%.3f gz=%.3f ax=%.3f ay=%.3f az=%.3f\n",
                  stdev[0], stdev[1], stdev[2], stdev[3], stdev[4], stdev[5]);
  }
  return max_ang > dyn_thresh;
}

void postWindow() {
  if (sampleCount < WINDOW) return;
  if (!windowIsDynamic(0.5f)) { // tweak threshold
    Serial.println("Window static — skip POST");
    return;
  }

  // Normal behavior: build mapped payload and print it (so you can copy/save it)
  if (!TEST_MAPPING_MODE) {
    HTTPClient http;
    http.begin(SERVER_PREDICT_URL);
    http.addHeader("Content-Type", "application/json");
    String payload;
    buildWindowJSON(payload);
    // Print exact JSON to Serial BEFORE POST so you can save it as live_window.json
    Serial.println("JSON_PAYLOAD_START");
    Serial.println(payload);
    Serial.println("JSON_PAYLOAD_END");
    int code = http.POST((uint8_t*)payload.c_str(), payload.length());
    if (code > 0) {
      String resp = http.getString();
      Serial.printf("POST %d, bytes=%d, resp=%s\n", code, payload.length(), resp.c_str());
    } else {
      Serial.printf("POST failed, code=%d\n", code);
    }
    http.end();
    return;
  }

  // TEST_MAPPING_MODE: cycle candidate mappings and POST each one (debug only)
  for (int ci = 0; ci < NUM_CANDIDATES; ++ci) {
    const Mapping &m = candidates[ci];
    String payload = buildWindowJSONWithMapping(m);
    // Print marker and payload so you can save the exact JSON
    Serial.printf("MAPPING_CANDIDATE %d START\n", ci);
    Serial.println(payload); // copy this JSON to live_window.json for offline analysis
    Serial.printf("MAPPING_CANDIDATE %d END\n", ci);

    HTTPClient http;
    http.begin(SERVER_PREDICT_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST((uint8_t*)payload.c_str(), payload.length());
    if (code > 0) {
      String resp = http.getString();
      Serial.printf("MAPPING %d -> POST %d, resp=%s\n", ci, code, resp.c_str());
    } else {
      Serial.printf("MAPPING %d -> POST failed, code=%d\n", ci, code);
    }
    http.end();
    delay(200); // brief pause between candidate posts
  }
}

bool testPing(const char *pingUrlFull) {
  HTTPClient http;
  http.begin(pingUrlFull);
  http.setTimeout(5000);
  int code = http.GET();
  Serial.printf("PING %s -> code=%d\n", pingUrlFull, code);
  if (code > 0) {
    String body = http.getString();
    Serial.printf("PING body: %s\n", body.c_str());
  }
  http.end();
  return (code > 0);
}

bool testSmallPost(const char *predictUrlFull) {
  HTTPClient http;
  http.begin(predictUrlFull);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);
  String small = "{\"frames\":[[0,0,0,0,0,0]]}";
  int code = http.POST(small);
  String resp = (code > 0) ? http.getString() : String();
  Serial.printf("POST test -> code=%d, resp=%s\n", code, resp.c_str());
  http.end();
  return (code > 0);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(); // default SDA/SCL

  // initialize Adafruit MPU (match imu_data_collect.ino settings)
  Serial.println("INIT: Initializing MPU6050...");
  if (!mpu.begin(0x68)) {
    Serial.println("ERROR: MPU6050 not found");
    while (1) delay(10);
  }
  // set same ranges used during data collection
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("INIT: MPU6050 ready (Adafruit lib)");

  // calibrate gyro bias (keep device stationary during boot)
  Serial.println("CALIBRATE: computing gyro bias, keep IMU stationary...");
  calibrateGyroBias(200, 10);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Connecting to WiFi %s", WIFI_SSID);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 60) {
    delay(500); Serial.print(".");
    tries++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi OK, IP: "); Serial.println(WiFi.localIP());
    testPing(SERVER_PING_URL.c_str());
    testSmallPost(SERVER_PREDICT_URL.c_str());
  } else {
    Serial.println("WiFi failed, will continue and retry later...");
  }

  lastSampleMs = millis();
  lastPostMs = millis();
}

void loop() {
  unsigned long now = millis();
  unsigned long samplePeriodMs = 1000UL / SAMPLE_HZ;
  if (now - lastSampleMs >= samplePeriodMs) {
    lastSampleMs = now;
    Sample s;
    if (readIMU(s)) {
      appendSample(s);
    }

    // dynamic detector in rad/s (model expects rad/s)
    float ang_mag = fabs(s.gx) + fabs(s.gy) + fabs(s.gz);
    const float DYN_THRESH = 0.7f; // ~40 deg/s = 0.7 rad/s, adjust if needed
    if (ang_mag > DYN_THRESH) {
      Serial.printf("DYNAMIC ang=%.3f gx=%.3f gy=%.3f gz=%.3f\n", ang_mag, s.gx, s.gy, s.gz);
    } else {
      Serial.printf("STATIC  ang=%.3f gx=%.3f gy=%.3f gz=%.3f\n", ang_mag, s.gx, s.gy, s.gz);
    }
  }

  if (now - lastPostMs >= POST_EVERY_MS) {
    lastPostMs = now;
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.reconnect();
    } else {
      postWindow();
    }
  }

  delay(1);
}

