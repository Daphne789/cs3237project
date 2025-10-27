import cv2
import requests
import numpy as np

url = "http://192.168.4.1:81/stream"
stream = requests.get(url, stream=True)

bytes_data = b''
frame_count = 0
MAX_FRAMES = 5

for chunk in stream.iter_content(chunk_size=1024):
    print("chunk", chunk)
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end
    if a != -1 and b != -1:
        print("Parsing...")
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is not None:
            cv2.imshow('ESP32 Stream', img)
            cv2.imwrite(f"captured_frames/frame_{frame_count+1}.jpg", img)
            print(f"Saved frame_{frame_count+1}.jpg")
            frame_count += 1

        if cv2.waitKey(1) == 27 or frame_count >= MAX_FRAMES:
            break

cv2.destroyAllWindows()
print("Done.")
