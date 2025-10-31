# filepath: Experiments/imu-lstm-trolley/tools/post_example_sequence.py
# usage: python post_example_sequence.py --host 10.75.215.177:5000 --action left
import argparse, json
import pandas as pd
import requests
from pathlib import Path

CSV = Path("Experiments/imu-lstm-trolley/data/raw/all-combined_imu_data_randomized.csv")

def build_sequence(df, action_label, seq_len=244):
    g = df[df['action_label']==action_label]
    # pick first full action group
    # group by action_id
    grp = g.groupby('action_id')
    for aid, gdf in grp:
        if len(gdf) >= seq_len:
            arr = gdf[['gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z']].values
            return arr[:seq_len].tolist()
    raise RuntimeError("no sequence of required length")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="10.75.215.177:5000")
    p.add_argument("--action", default="left")
    args = p.parse_args()
    df = pd.read_csv(CSV)
    seq = build_sequence(df, args.action, seq_len=244)
    url = f"http://{args.host}/predict?log=1"
    r = requests.post(url, json={"frames": seq}, timeout=10)
    print("status", r.status_code, r.text)