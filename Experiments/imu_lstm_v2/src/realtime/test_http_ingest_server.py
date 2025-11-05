import os
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient

from Experiments.imu_lstm_v2.src.realtime.http_ingest_server import app, state

class DummyScaler:
    def transform(self, X):
        return X

class DummyLE:
    def __init__(self, classes):
        self.classes_ = np.array(classes)
    def inverse_transform(self, idx_arr):
        return self.classes_[np.asarray(idx_arr, dtype=int)]

class DummyModel(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = list(classes)
        self.idx = {c:i for i,c in enumerate(self.classes)}
    def forward(self, x):
        with torch.no_grad():
            B, T, C = x.shape
            out = torch.zeros((B, len(self.classes)), dtype=torch.float32)
            for b in range(B):
                xb = x[b].cpu().numpy()
                g = xb[:, :3].mean(axis=0)
                a = xb[:, 3:6].mean(axis=0)
                acc_mag = float(np.linalg.norm(a))
                if acc_mag > 13.0:
                    cls = "jump"
                elif g[0] > 0.05:
                    cls = "right"
                elif g[0] < -0.05:
                    cls = "left"
                else:
                    cls = "straight"
                out[b, self.idx[cls]] = 5.0
            return out

def setup_dummy_state(tmpdir: Path, window=3, threshold=0.2, debounce_k=1, min_acc_for_jump=12.0):
    state["windows"].clear(); state["recent"].clear(); state["rot"].clear()
    state["hist"].clear(); state["latest"].clear(); state["current_cmd"].clear()
    state["log_rows"].clear()
    classes = ["left","right","straight","jump"]
    state["le"] = DummyLE(classes)
    state["scaler"] = DummyScaler()
    state["device"] = torch.device("cpu")
    state["cfg"] = {"hidden":64,"layers":2,"dropout":0.2,"bidirectional":False}
    state["eff_window"] = window
    state["threshold"] = threshold
    state["debounce_k"] = debounce_k
    # make debounce/threshold behavior deterministic for tests
    state["debounce_map"] = {"default": debounce_k, "straight": debounce_k}
    state["threshold_map"] = {"default": threshold, "straight": threshold}
    state["min_acc_for_jump"] = min_acc_for_jump
    state["model"] = DummyModel(classes)
    state["g_ref"] = np.array([0.0, 9.81, 0.0], dtype=float)
    state["artifacts"] = str(tmpdir)
    (Path(tmpdir) / "calib").mkdir(parents=True, exist_ok=True)
    state["log_dir"] = str(Path(tmpdir) / "logs"); Path(state["log_dir"]).mkdir(parents=True, exist_ok=True)

def make_samples(gx=0.1, gy=0.0, gz=0.0, ax=0.0, ay=14.0, az=0.0, n=3):
    return [{"ts": 1000+i, "gx": gx, "gy": gy, "gz": gz, "ax": ax, "ay": ay, "az": az} for i in range(n)]

def test_ingest_predicts_and_logs(tmp_path):
    setup_dummy_state(tmp_path, window=3, threshold=0.2, debounce_k=1, min_acc_for_jump=20.0)
    client = TestClient(app)
    payload = {"device_id":"imu01", "samples": make_samples(gx=0.2, ay=9.5)}
    r = client.post("/ingest", json=payload); assert r.status_code == 200
    last = client.get("/latest", params={"device_id":"imu01"}).json()
    assert last["label"] == "right" and last["command"] == "RIGHT"
    assert len(state["log_rows"]) >= 1
    j = client.post("/flush_log").json()
    assert j["ok"] and j["path"] and os.path.exists(j["path"])

def test_jump_gating(tmp_path):
    setup_dummy_state(tmp_path, window=3, threshold=0.2, debounce_k=1, min_acc_for_jump=12.0)
    client = TestClient(app)
    r = client.post("/ingest", json={"device_id":"imu01", "samples": make_samples(gx=0.2, ay=10.0)})
    assert r.status_code == 200 and r.json()["last"]["label"] == "right"
    r2 = client.post("/ingest", json={"device_id":"imu01", "samples": make_samples(gx=0.2, ay=14.5)})
    assert r2.status_code == 200 and r2.json()["last"]["label"] == "jump"

def test_calibrate_and_swap_updates_rotation(tmp_path):
    setup_dummy_state(tmp_path, window=3, threshold=0.2, debounce_k=1)
    client = TestClient(app)
    client.post("/ingest", json={"device_id":"imu01", "samples": make_samples(gx=0.0, ay=9.81)})
    cr = client.post("/calibrate_gravity", params={"device_id":"imu01"}); assert cr.status_code == 200
    sr = client.post("/swap_left_right", params={"device_id":"imu01"}); assert sr.status_code == 200

def test_stats_and_config_and_recent(tmp_path):
    setup_dummy_state(tmp_path, window=3, threshold=0.1, debounce_k=1)
    client = TestClient(app)
    client.post("/ingest", json={"device_id":"imu01", "samples": make_samples(gx=-0.2, ay=9.7)})
    assert client.get("/stats", params={"device_id":"imu01","scaled":0}).json()["ok"]
    assert client.get("/stats", params={"device_id":"imu01","scaled":1}).json()["ok"]
    assert "effective_window" in client.get("/config").json()
    assert client.get("/recent", params={"device_id":"imu01","n":5}).json()["ok"]