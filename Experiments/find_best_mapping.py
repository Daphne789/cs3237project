import joblib, json, numpy as np, pandas as pd
from itertools import permutations, product
from pathlib import Path

MODEL_DIR = Path("/home/daphnesw/cs3237/cs3237project/Experiments/imu-lstm-trolley/models_artifacts")
CSV = Path("/home/daphnesw/cs3237/cs3237project/Experiments/imu_data_collect/imu_data/all-combined_imu_data_randomized.csv")
LIVE_JSON = Path("/home/daphnesw/cs3237/cs3237project/Experiments/live_window.json")
SCALER_PATH = MODEL_DIR / "scaler.joblib"

def add_features(frames):
    ax = frames[:,3]; ay = frames[:,4]; az = frames[:,5]
    gx = frames[:,0]; gy = frames[:,1]; gz = frames[:,2]
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)
    a_stack = np.vstack([ax,ay,az]).T
    delta = np.vstack([np.zeros(3), np.diff(a_stack, axis=0)])
    delta_mag = np.linalg.norm(delta, axis=1)
    return np.column_stack([gx,gy,gz,ax,ay,az, accel_mag, gyro_mag, delta_mag])

scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(CSV)
# pick first left action
g = df[df['action_label']=='left'].groupby('action_id')
first_id = next(iter(g.groups))
train_frames = g.get_group(first_id)[['gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z']].values

if not LIVE_JSON.exists():
    print("live_window.json not found. Save a real live window and rerun.")
    raise SystemExit

live = np.array(json.load(open(LIVE_JSON))['frames'], dtype=np.float32)

# utility to apply mapping: permute indices and apply signs
def map_frames(frames, gyro_perm, accel_perm, gyro_signs, accel_signs):
    g = frames[:, :3].copy()
    a = frames[:, 3:6].copy()
    g_map = np.column_stack([g[:,i] * s for i,s in zip(gyro_perm, gyro_signs)])
    a_map = np.column_stack([a[:,i] * s for i,s in zip(accel_perm, accel_signs)])
    return np.hstack([g_map, a_map])

best = []
for gperm in permutations(range(3)):
    for aperm in permutations(range(3)):
        for gsign in product([1,-1], repeat=3):
            for asign in product([1,-1], repeat=3):
                mapped_live = map_frames(live, gperm, aperm, gsign, asign)
                lf = add_features(train_frames)
                lv = add_features(mapped_live)
                try:
                    lf_s = scaler.transform(lf)
                    lv_s = scaler.transform(lv)
                except Exception:
                    continue
                dist = np.linalg.norm(lf_s.mean(axis=0) - lv_s.mean(axis=0))
                best.append((dist, gperm, aperm, gsign, asign))
# sort and print top 10
best.sort(key=lambda x: x[0])
for dist,gp,ap,gs,as_ in best[:10]:
    print(f"dist={dist:.4f} gperm={gp} aperm={ap} gsign={gs} asign={as_}")