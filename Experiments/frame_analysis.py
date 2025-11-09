import cv2
import requests
import numpy as np
from tensorflow.keras.models import load_model
from distance.estimate_dist import calc_dist
import time 

url = "http://192.168.4.1:81/stream"
stream = requests.get(url, stream=True)
model = load_model("apriltag_regressor_finetuned.keras")

OG_W, OG_H = 320, 240
IMG_W, IMG_H = 64, 64

bytes_data = b''

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1 and a < b:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is not None:
            cv2.imshow('ESP32 Stream', img)

            # --- Convert to grayscale and preprocess ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (IMG_W, IMG_H))
            gray_resized = gray_resized.astype(np.float32) / 255.0
            img_input = np.expand_dims(gray_resized, (0, -1))  # (1, 64, 64, 1)

            # --- Predict corners ---
            corner_pred = model.predict(img_input, verbose=0)[0]

            # --- Denormalise to original resolution ---
            pred_corners_px = corner_pred.copy()
            pred_corners_px[0::2] *= OG_W
            pred_corners_px[1::2] *= OG_H
            corners = pred_corners_px.reshape(4, 2)

            print("Predicted corners (px):", corners)

            forward_distance = calc_dist(corners)
            print("Estimated distance:", forward_distance)
            #time.sleep(0.1)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
stream.close()
print("Done.")
