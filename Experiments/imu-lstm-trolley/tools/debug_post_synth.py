# filepath: src/tools/debug_post_synth.py
import requests, numpy as np, json, sys
url = "http://10.75.215.177:5000/predict?log=1"
# build synthetic window: 244 frames, baseline gravity on Y (straight) then create tilt on X (left)
def make_window(tilt='none'):
    N=244
    frames=[]
    for i in range(N):
        if tilt=='left':
            # strong accel_x negative tilt + small gyro
            frames.append([0.0, 0.0, 0.0, -6.0, 9.0, 0.0])
        elif tilt=='right':
            frames.append([0.0, 0.0, 0.0, 6.0, 9.0, 0.0])
        else:
            frames.append([0.0, 0.0, 0.0, 0.0, 9.8, 0.0])
    return frames

for t in ['none','left','right']:
    print("POST tilt=",t)
    r = requests.post(url, json={'frames': make_window(t)}, timeout=5)
    print(r.status_code, r.text)