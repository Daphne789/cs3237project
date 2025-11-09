import cv2
import requests
import numpy as np
from tensorflow.keras.models import load_model
from distance.estimate_dist import calc_dist
import time 
from scipy import stats
from PIL import Image

def preprocess_image(img):
    IMG_W, IMG_H = 64, 64
    # img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, (0, -1))
    return img_input # (1, 64, 64, 1)

def run_cnn_model(cnn_model="apriltag_regressor_finetuned.keras"):
    OG_W, OG_H = 240, 240
    url = "http://192.168.4.1:81/stream"
    stream = requests.get(url, stream=True)
    corners_model = load_model(cnn_model)
    # img_input = preprocess_image(input("Input img filepath: "))
    # print(corners_model.predict(img_input))
    
    bytes_data = b''

    last_check_time = time.time()
    detections_in_window = []

    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1 and a < b:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                cv2.imshow('ESP32 Stream', img)
                img_input = preprocess_image(img)
                corner_pred = corners_model.predict(img_input, verbose=0)[0]
                pred_corners_px = corner_pred.copy()
                pred_corners_px[0::2] *= OG_W
                pred_corners_px[1::2] *= OG_H
                corners = pred_corners_px.reshape(4, 2)

                forward_distance = calc_dist(corners)
                detections_in_window.append(forward_distance)

                curr_time = time.time()
                if curr_time - last_check_time >= 1.0:
                    if any(dist < 0 for dist in detections_in_window):
                        print("No april tag detected")
                    else:
                        avg_pred = np.mean(detections_in_window)
                        avg_pred_scipy = stats.trim_mean(detections_in_window, proportiontocut=0.2)
                        print("AVG_PRED:", avg_pred)
                        print("trimmed mean:", avg_pred_scipy)

                        try:
                            payload = {
                                "device_id": "imu01",
                                "distance": float(avg_pred_scipy),
                                "timestamp": time.time(),
                            }
                            resp = requests.post("http://localhost:5000/distance", json=payload, timeout=0.5)
                            if resp.ok:
                                print("Sent distance:", avg_pred_scipy)
                            else:
                                print("Failed to send:", resp.status_code, resp.text)
                        except Exception as e:
                            print("Error sending distance:", e)

                    detections_in_window.clear()
                    last_check_time = curr_time

            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()
    stream.close()
    print("Done.")

if __name__ == "__main__":
    run_cnn_model()
