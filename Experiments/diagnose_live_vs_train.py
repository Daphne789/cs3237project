import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("/home/daphnesw/cs3237/cs3237project/Experiments/imu-lstm-trolley/models_artifacts")
CSV = Path("/home/daphnesw/cs3237/cs3237project/Experiments/imu_data_collect/imu_data/all-combined_imu_data_randomized.csv")
LIVE_JSON = Path("/home/daphnesw/cs3237/cs3237project/Experiments/live_window.json")
SCALER_PATH = MODEL_DIR / "scaler.joblib"

def load_first_action_of_label(df, label):
    g = df[df['action_label']==label].groupby('action_id')
    first_id = next(iter(g.groups))
    rows = g.get_group(first_id)
    feats = rows[['gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z']].values
    return feats

def load_live_frames(path):
    j = json.load(open(path))
    frames = np.array(j.get("frames", []), dtype=np.float32)
    return frames

def add_features(frames):
    # mimic repo _add_features: append accel_mag, gyro_mag, delta_mag per row
    ax = frames[:,3]; ay = frames[:,4]; az = frames[:,5]
    gx = frames[:,0]; gy = frames[:,1]; gz = frames[:,2]
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)
    a_stack = np.vstack([ax,ay,az]).T
    delta = np.vstack([np.zeros(3), np.diff(a_stack, axis=0)])
    delta_mag = np.linalg.norm(delta, axis=1)
    feat = np.column_stack([gx,gy,gz,ax,ay,az, accel_mag, gyro_mag, delta_mag])
    return feat

def main():
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(CSV)
    left_feat = load_first_action_of_label(df, "left")
    left_feat9 = add_features(left_feat)

    if not LIVE_JSON.exists():
        print(f"Error: live window not found: {LIVE_JSON}")
        print("Create/save live_window.json (see synth_post.py) and rerun.")
        return

    live_frames = load_live_frames(LIVE_JSON)
    if live_frames.shape[0] == 0:
        print("Error: live_window.json contains no frames")
        return
    live_feat9 = add_features(live_frames)

    left_mean_raw = left_feat9.mean(axis=0)
    live_mean_raw = live_feat9.mean(axis=0)

    left_scaled = scaler.transform(left_feat9)
    live_scaled = scaler.transform(live_feat9)

    print("raw means (first6) left:", np.round(left_mean_raw[:6],4))
    print("raw means (first6) live:", np.round(live_mean_raw[:6],4))
    print("L2 raw mean distance:", np.linalg.norm(left_mean_raw - live_mean_raw))
    print("L2 scaled mean distance:", np.linalg.norm(left_scaled.mean(axis=0) - live_scaled.mean(axis=0)))
    print("Scaled left mean (first9):", np.round(left_scaled.mean(axis=0)[:9],4))
    print("Scaled live mean (first9):", np.round(live_scaled.mean(axis=0)[:9],4))

if __name__ == "__main__":
    main()