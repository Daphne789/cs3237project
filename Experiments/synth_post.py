import requests, json, numpy as np, time
from pathlib import Path

URL = "http://127.0.0.1:5000/predict?log=1&session=test"
WINDOW = 244
frames = []
for i in range(WINDOW):
    gx = 0.0; gy = 0.0
    if 60 <= i < 80:
        gz = 3.0   # rad/s pulse, tune amplitude/time
    else:
        gz = 0.0
    ax,ay,az = 0.0,9.8,0.0
    frames.append([gx,gy,gz,ax,ay,az])

payload = {"frames": frames}

# save payload locally for diagnostics (diagnose_live_vs_train.py expects live_window.json)
out_path = Path(__file__).resolve().parent / "live_window.json"
with open(out_path, "w") as f:
    json.dump(payload, f)
print(f"Saved synthetic payload to {out_path}")

try:
    r = requests.post(URL, json=payload, timeout=5)
    print("status", r.status_code)
    try:
        print("resp", r.json())
    except Exception:
        print("resp text:", r.text)
except Exception as e:
    print("POST failed:", e)