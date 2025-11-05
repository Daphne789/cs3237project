import time, csv, requests, argparse, json
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--server", default="http://127.0.0.1:5000/predict")
parser.add_argument("--server-log", action="store_true", help="ask server to also append predictions_log.csv")
parser.add_argument("--out", default="replay_predictions.csv")
parser.add_argument("--step", type=int, default=5, help="step between windows (frames)")
parser.add_argument("--window", type=int, default=50, help="frames per POST")
parser.add_argument("--delay", type=float, default=0.02, help="delay between posts (s)")
args = parser.parse_args()
try:
    df = pd.read_csv(args.csv)
except Exception:
    print("pd.read_csv failed, retrying with engine='python' and skipping bad lines")
    df = pd.read_csv(args.csv, engine="python", on_bad_lines="skip")

features = ["gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z"]

# build server URL and ensure log=0 by default
server_url = args.server
if "?" not in server_url:
    server_url = server_url + ("?log=1" if args.server_log else "?log=0")

# prepare output (create dir, write header now)
outp = Path(args.out)
outp.parent.mkdir(parents=True, exist_ok=True)
f = outp.open("w", newline="")
w = csv.writer(f)
w.writerow(["action_id","start","len","request_time","response"])
f.flush()

sent = 0
for aid, g in df.groupby("action_id"):
    seq = g.sort_values("timestamp")[features].values.astype(float)
    for start in range(0, max(1, len(seq)-args.window+1), args.step):
        window = seq[start:start+args.window]
        payload = {"frames": window.tolist()}
        try:
            r = requests.post(server_url, json=payload, timeout=2.0)
            j = r.json()
        except Exception as e:
            j = {"error": str(e)}
        w.writerow([int(aid), start, len(window), time.time(), json.dumps(j)])
        sent += 1
        if sent % 20 == 0:
            f.flush()
        if args.delay > 0:
            time.sleep(args.delay)

f.flush()
f.close()
print("wrote", outp.resolve())