import argparse, sys, json, os, time, atexit, csv
from pathlib import Path
from typing import Dict, Any
import numpy as np, torch
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from collections import deque
from datetime import datetime

# Make "src" importable
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Experiments.imu_lstm_v2.src.utils.io import load_joblib, load_json  # absolute import (namespace pkg)
from Experiments.imu_lstm_v2.src.realtime.sliding_window import SlidingWindow
from Experiments.imu_lstm_v2.src.models.lstm_v2 import LSTMIMU

app = FastAPI()

# map for gyro indices in our 6D vector [gx,gy,gz,ax,ay,az]
YAW_IDX = {"gx": 0, "gy": 1, "gz": 2}

state = {
    "scaler": None,
    "le": None,
    "cfg": None,
    "model": None,
    "device": None,
    "eff_window": None,
    "artifacts": None,
    "threshold": 0.6,
    "per_class_threshold": {},  # optional per-class thresholds
    "debounce_k": 2,            # require K consecutive predictions to switch command
    "min_acc_for_jump": 12.0,   # m/s^2; gating to suppress spurious JUMP
    # new: class-specific debounce/threshold and quick-straight settings
    "debounce_map": {"default": 3, "straight": 2},
    "threshold_map": {"default": 0.6, "straight": 0.55, "left": 0.7, "right": 0.7},
    "quiet_gyro": 0.06,   # rad/s; mean gyro magnitude considered "quiet"
    "quiet_ms": 250,      # ms required quiet time to accept STRAIGHT fast
    "last_quiet": {},     # device_id -> last timestamp (time.time()) when gyro was quiet
    # fast-turn settings
    "turn_fast_thr": 0.90,   # if non-straight prob >= this, switch immediately
    "turn_min_gyro": 0.10,   # require some rotation to avoid noise-trigger
    # --- replace old yaw-assist snap with probabilistic prior + EMA ---
    "ema_probs": {},         # device_id -> np.ndarray of class probs
    "ema_alpha": 0.6,        # 0..1, higher = less smoothing (more responsive)
    "yaw_assist": True,      # enable yaw prior shaping
    "yaw_axis": "gx",
    "yaw_sign": 1,           # +1 rightwards yaw makes RIGHT more likely
    "yaw_min": 0.12,         # min |yaw| to start adding prior (rad/s)
    "yaw_window": 12,        # samples to average for yaw
    "yaw_prior_k": 2.0,      # logit bias scale
    "yaw_prior_min": 0.08,   # start ramp
    "yaw_prior_max": 0.35,   # saturate ramp
    # --- jump-specific helpers ---
    "probs_s": {},                # device_id -> latest smoothed probs
    "jump_hold_ms": 140,          # hold JUMP for a short time
    "jump_lock_until": {},        # device_id -> unix ts until which we keep JUMP
    "windows": {},              # device_id -> SlidingWindow
    "latest": {},               # device_id -> dict(label, confidence, probs, timestamp)
    "recent": {},               # device_id -> deque of raw mapped samples
    "rot": {},                  # device_id -> 3x3 rotation matrix (np.ndarray)
    "g_ref": None,              # target gravity vector in training frame (np.ndarray, shape (3,))
    "hist": {},                 # device_id -> deque of recent labels for debounce
    "current_cmd": {},          # device_id -> current debounced command
    "classes": [],              # ordered list of class names from label encoder
    "log_rows": [],             # buffered log rows
    "log_dir": None,            # where to write csv on exit
}

# Default mapping (identity). You can override via artifacts/axis_map.json
# idx: permutation for [gx,gy,gz,ax,ay,az], sign: per-channel sign flips
AXIS_MAP = {"idx": [0, 1, 2, 3, 4, 5], "sign": [1, 1, 1, 1, 1, 1]}

def load_axis_map(artifacts: Path):
    path = artifacts / "axis_map.json"
    if path.exists():
        try:
            cfg = json.load(open(path))
            idx = cfg.get("idx", AXIS_MAP["idx"])
            sign = cfg.get("sign", AXIS_MAP["sign"])
            if len(idx) == 6 and len(sign) == 6:
                return {"idx": [int(i) for i in idx], "sign": [int(s) for s in sign]}
        except Exception:
            pass
    return AXIS_MAP

def apply_map(parts):
    # parts: [gx,gy,gz,ax,ay,az]
    out = [0.0] * 6
    for i in range(6):
        out[i] = AXIS_MAP["sign"][i] * parts[AXIS_MAP["idx"][i]]
    return out

def load_gravity_ref(artifacts: Path):
    path = artifacts / "gravity_ref.json"
    if path.exists():
        try:
            j = json.load(open(path))
            g = np.array(j.get("g_ref", [0, 9.81, 0]), dtype=float)
            if g.shape == (3,):
                return g
        except Exception:
            pass
    # default: assume most gravity on +Y in training
    return np.array([0.0, 9.81, 0.0], dtype=float)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def rot_from_vectors(a, b):
    # returns R such that R @ a = b (both 3D)
    a_u, b_u = normalize(a), normalize(b)
    v = np.cross(a_u, b_u)
    c = np.dot(a_u, b_u)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)  # parallel or antiparallel
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=float)
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R

def apply_rotation(parts, device_id: str):
    R = state["rot"].get(device_id, None)
    if R is None:
        return parts
    g = np.array(parts[0:3], dtype=float)  # gyro
    a = np.array(parts[3:6], dtype=float)  # accel
    g_r = (R @ g).tolist()
    a_r = (R @ a).tolist()
    return g_r + a_r

# Save/load rotation to artifacts/rot_<device_id>.json
def _rot_path(device_id: str) -> Path:
    d = Path(state["artifacts"]) / "calib"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"rot_{device_id}.json"

def save_rot(device_id: str, R):
    p = _rot_path(device_id)
    with open(p, "w") as f:
        json.dump({"R": (R.tolist() if hasattr(R, "tolist") else R)}, f)

def load_rot(device_id: str):
    p = _rot_path(device_id)
    if p.exists():
        try:
            j = json.load(open(p))
            return np.array(j["R"], dtype=float)
        except Exception:
            pass
    return None

def rot_axis_angle(u, theta):
    u = np.asarray(u, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)
    ux = np.array([[0, -u[2], u[1]],[u[2], 0, -u[0]],[-u[1], u[0], 0]], dtype=float)
    I = np.eye(3)
    return I * np.cos(theta) + np.sin(theta) * ux + (1 - np.cos(theta)) * np.outer(u, u)

def init_model(artifacts: Path, threshold: float, window_override: int = 0):
    # load axis map into global for apply_map()
    global AXIS_MAP
    AXIS_MAP = load_axis_map(artifacts)
    scaler = load_joblib(artifacts / "scaler.joblib")
    le = load_joblib(artifacts / "label_encoder.joblib")
    cfg = load_json(artifacts / "config.json")["config"]
    eff_window = window_override if window_override > 0 else cfg.get("effective_window", cfg.get("window", 150))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMIMU(input_size=6, hidden_size=cfg["hidden"], num_layers=cfg["layers"],
                    dropout=cfg["dropout"], num_classes=len(le.classes_), bidirectional=cfg["bidirectional"]).to(device)
    model.load_state_dict(torch.load(artifacts / "lstm_v2_best.pth", map_location=device))
    model.eval()
    state.update(dict(scaler=scaler, le=le, cfg=cfg, model=model, device=device,
                      eff_window=eff_window, threshold=threshold, artifacts=str(artifacts)))
    state["g_ref"] = load_gravity_ref(artifacts)
    # logging dir
    log_dir = Path(artifacts) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state["log_dir"] = str(log_dir)
    print("[INFO] Axis map idx=", AXIS_MAP["idx"], "sign=", AXIS_MAP["sign"])
    print("[INFO] g_ref =", state["g_ref"].tolist())

def ensure_window(device_id: str):
    if device_id not in state["windows"]:
        state["windows"][device_id] = SlidingWindow(state["eff_window"], n_feat=6)
    if device_id not in state["recent"]:
        state["recent"][device_id] = deque(maxlen=300)
    if device_id not in state["rot"]:
        R = load_rot(device_id)
        state["rot"][device_id] = (R if R is not None else np.eye(3))
    # if device_id not in state["hist"]:
    #     state["hist"][device_id] = deque(maxlen=max(1, int(state["debounce_k"])))
    # keep a slightly longer history to support variable K checks
    if device_id not in state["hist"]:
        state["hist"][device_id] = deque(maxlen=8)
    if device_id not in state["last_quiet"]:
        state["last_quiet"][device_id] = 0.0

# Quick rolling stats for debugging
def window_stats(arr):  # arr: (T,6) scaled or raw
    m = np.mean(arr, axis=0).tolist()
    s = np.std(arr, axis=0).tolist()
    return dict(mean=m, std=s)

def _gate_jump_and_decide(probs: np.ndarray, arr_raw: np.ndarray) -> (str, float, np.ndarray, int):
    """
    Apply jump gating: if JUMP is top but vertical accel (|a·ĝ|) P90 is below gate,
    suppress JUMP. Returns (label, confidence, probs, idx)
    """
    classes = list(state["le"].classes_)
    idx = int(np.argmax(probs))
    # Only when top-1 is jump
    if classes[idx].lower() == "jump":
        acc = arr_raw[:, 3:6]
        g_hat = normalize(state["g_ref"])
        a_proj_abs = np.abs(acc @ g_hat)                 # vertical |acc|
        a_p90 = float(np.percentile(a_proj_abs, 90.0))   # robust short-impulse detector
        if a_p90 < float(state["min_acc_for_jump"]):     # reuse gate as vertical P90 threshold
            probs[idx] = 0.0
            idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), probs, idx

def _apply_yaw_prior(device_id: str, probs: np.ndarray, arr_raw: np.ndarray) -> np.ndarray:
    """
    Softly bias L/R logits using recent yaw sign/magnitude. Returns renormalized probs.
    """
    classes = list(state["le"].classes_)
    try:
        li = classes.index("left"); ri = classes.index("right")
    except ValueError:
        return probs  # classes missing
    yaw_idx = YAW_IDX.get(state.get("yaw_axis", "gx"), 0)
    n = max(3, min(int(state.get("yaw_window", 12)), arr_raw.shape[0]))
    yaw = arr_raw[-n:, yaw_idx].mean() * float(state.get("yaw_sign", 1))
    # magnitude ramp 0..1 between prior_min and prior_max
    a = float(state.get("yaw_prior_min", 0.08)); b = float(state.get("yaw_prior_max", 0.35))
    mag = max(0.0, min(1.0, (abs(yaw) - a) / max(1e-6, (b - a))))
    if mag <= 0.0:
        return probs
    k = float(state.get("yaw_prior_k", 2.0))
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    if yaw > 0:     # favor RIGHT
        logits[ri] += k * mag
    elif yaw < 0:   # favor LEFT
        logits[li] += k * mag
    e = np.exp(logits - logits.max())
    return e / e.sum()

def _ema_update(device_id: str, probs: np.ndarray) -> np.ndarray:
    alpha = float(state.get("ema_alpha", 0.6))
    prev = state["ema_probs"].get(device_id, None)
    out = probs if prev is None else (alpha * probs + (1.0 - alpha) * prev)
    state["ema_probs"][device_id] = out
    # expose latest smoothed probs for jump logic
    state["probs_s"][device_id] = out
    return out

def _debounced_command(device_id: str, label: str, confidence: float, now_ts: float, arr_raw: np.ndarray) -> str:
    # Track quiet gyro
    g = arr_raw[:, :3]
    g_mag = float(np.sqrt((g**2).sum(axis=1)).mean())
    if g_mag < float(state["quiet_gyro"]):
        state["last_quiet"][device_id] = now_ts

    # Fast STRAIGHT recovery (unchanged) ...
    th_map = state["threshold_map"]; th_def = float(th_map.get("default", state["threshold"]))
    th_straight = float(th_map.get("straight", th_def))
    th_label = float(th_map.get(label.lower(), th_def))  # NEW
    if label.lower() == "straight" and confidence >= th_straight:
        if (now_ts - state["last_quiet"].get(device_id, 0.0)) * 1000.0 <= float(state["quiet_ms"]):
            state["current_cmd"][device_id] = "STRAIGHT"
            state["hist"][device_id].clear()
            return "STRAIGHT"
    
        # --- Jump onset + hold (no gyro requirement) ---
    classes = list(state["le"].classes_)
    try:
        j_idx = classes.index("jump")
    except ValueError:
        j_idx = None
    # keep jump while locked
    lock_until = state["jump_lock_until"].get(device_id, 0.0)
    if now_ts < float(lock_until):
        state["current_cmd"][device_id] = "JUMP"
        return "JUMP"

    # consider triggering jump
    if j_idx is not None:
        probs_s = state["probs_s"].get(device_id)
        p_jump = float(probs_s[j_idx]) if probs_s is not None else 0.0
        th_jump = float(th_map.get("jump", th_def))
        # vertical accel P90 gate
        acc = arr_raw[:, 3:6]
        g_hat = normalize(state["g_ref"])
        a_p90 = float(np.percentile(np.abs(acc @ g_hat), 90.0))
        # jump onset if either: strong p_jump OR (moderate p_jump + strong vertical impulse)
        if (p_jump >= float(state["turn_fast_thr"])) or (p_jump >= th_jump and a_p90 >= float(state["min_acc_for_jump"])):
            state["current_cmd"][device_id] = "JUMP"
            state["hist"][device_id].clear()
            state["jump_lock_until"][device_id] = now_ts + float(state.get("jump_hold_ms", 140)) / 1000.0
            return "JUMP"
    # --- end jump block ---

    # NEW: Fast TURN override (non-straight)
    if label.lower() != "straight":
        if confidence >= float(state["turn_fast_thr"]) and g_mag >= float(state["turn_min_gyro"]):
            state["current_cmd"][device_id] = label.upper()
            state["hist"][device_id].clear()
            return label.upper()

    # Regular debounce by class (unchanged below)
    k_map = state["debounce_map"]; k_def = int(k_map.get("default", state["debounce_k"]))
    k = int(k_map.get(label.lower(), k_def))
    th = th_def
    hist = state["hist"][device_id]
    hist.append(label)

    if confidence < th_label:
        return state["current_cmd"].get(device_id, "NONE")

    last_k = list(hist)[-k:]
    if len(last_k) == k and all(l == label for l in last_k):
        state["current_cmd"][device_id] = label.upper()
    return state["current_cmd"].get(device_id, "NONE")


def _append_log_row(device_id: str, ts: float, label: str, command: str, confidence: float, probs: Dict[str, float], raw_latest: Dict[str, float]):
    row = {
        "timestamp": ts,
        "device_id": device_id,
        "label": label,
        "command": command,
        "confidence": confidence,
    }
    # stable order for classes
    for c in state["le"].classes_:
        row[f"prob_{c}"] = float(probs.get(c, 0.0))
    if raw_latest:
        for k in ["gx","gy","gz","ax","ay","az"]:
            row[f"raw_{k}"] = float(raw_latest.get(k, 0.0))
    state["log_rows"].append(row)

def write_logs_csv(out_dir: str | None = None) -> str | None:
    """
    Write buffered prediction rows to CSV. Returns path if written.
    """
    rows = state.get("log_rows", [])
    if not rows:
        return None
    out_base = Path(out_dir or state.get("log_dir") or ".")
    out_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_base / f"predictions_log_{ts}.csv"
    # collect columns
    base_cols = ["timestamp","device_id","label","command","confidence"]
    prob_cols = [f"prob_{c}" for c in getattr(state.get("le", None), "classes_", [])]
    raw_cols = [f"raw_{k}" for k in ["gx","gy","gz","ax","ay","az"]]
    cols = base_cols + prob_cols + raw_cols
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return str(path)

@atexit.register
def _on_exit_flush_logs():
    try:
        p = write_logs_csv()
        if p:
            print(f"[INFO] Wrote prediction log to {p}")
    except Exception as e:
        print(f"[WARN] Failed to write prediction log on exit: {e}")

@app.post("/ingest")
def ingest(payload: Dict[str, Any] = Body(...)):
    try:
        device_id = str(payload.get("device_id", "imu"))
        samples = payload.get("samples", [])
        ensure_window(device_id)
        win = state["windows"][device_id]
        pred = None
        for s in samples:
            parts = [float(s["gx"]), float(s["gy"]), float(s["gz"]),
                     float(s["ax"]), float(s["ay"]), float(s["az"])]
            parts = apply_map(parts)
            parts = apply_rotation(parts, device_id)  # apply 3D rotation
            ts = int(s.get("ts", time.time() * 1000))
            latest_sample = {
                "ts": ts,
                "gx": parts[0], "gy": parts[1], "gz": parts[2],
                "ax": parts[3], "ay": parts[4], "az": parts[5],
            }
            state["recent"][device_id].append(latest_sample)
            win.push(parts)
            if win.ready():
                arr_raw = win.array()            # (T,6)
                arr = arr_raw[None, ...]
                scaler = state["scaler"]
                arr_s = scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
                xb = torch.tensor(arr_s, dtype=torch.float32).to(state["device"])
                with torch.no_grad():
                    logits = state["model"](xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                now_ts = time.time()
                
                # 1) Jump-gating first (suppress spurious jump)
                label_g, conf_g, probs_g, _ = _gate_jump_and_decide(probs.copy(), arr_raw)

                # 2) Yaw prior shaping (soft bias, no override)
                if state.get("yaw_assist", True):
                    probs_p = _apply_yaw_prior(device_id, probs_g, arr_raw)
                else:
                    probs_p = probs_g

                # 3) Temporal smoothing of probabilities (EMA)
                probs_s = _ema_update(device_id, probs_p)

                # 4) Final decision from smoothed probs
                idx = int(np.argmax(probs_s))
                classes = list(state["le"].classes_)
                label = classes[idx]
                conf = float(probs_s[idx])

                # 5) Debounce/threshold using window stats
                cmd = _debounced_command(device_id, label, conf, now_ts, arr_raw)

                mean = arr_raw.mean(axis=0).tolist()
                pred = {
                    "label": label,
                    "command": cmd or "NONE",
                    "confidence": conf,
                    "probs": {classes[i]: float(probs_s[i]) for i in range(len(classes))},
                    "raw_latest": latest_sample,
                    "raw_mean": {"gx": mean[0], "gy": mean[1], "gz": mean[2], "ax": mean[3], "ay": mean[4], "az": mean[5]},
                    "timestamp": now_ts,
                    "window": state["eff_window"]
                }

                # log row
                _append_log_row(device_id, pred["timestamp"], label, pred["command"], pred["confidence"], pred["probs"], latest_sample)

        if pred is not None:
            state["latest"][device_id] = pred
        return JSONResponse({"ok": True, "device_id": device_id, "last": state["latest"].get(device_id)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@app.post("/calibrate_gravity")
def calibrate_gravity(device_id: str = "imu01"):
    # Hold IMU in your "straight" pose for ~2s, then call this.
    ensure_window(device_id)
    win = state["windows"][device_id]
    if not win.ready():
        return {"ok": False, "error": "window not ready"}
    arr = win.array()  # (T,6)
    g_live = arr[:, 3:6].mean(axis=0)           # mean accel in live frame
    g_ref = state["g_ref"]
    R = rot_from_vectors(g_live, g_ref)
    state["rot"][device_id] = R
    save_rot(device_id, R)  # persist
    return {"ok": True, "g_live": g_live.tolist(), "g_ref": g_ref.tolist(), "R": R.tolist()}

@app.get("/debug_rot")
def debug_rot(device_id: str = "imu01"):
    R = state["rot"].get(device_id)
    return {"ok": True, "R": R.tolist() if R is not None else None, "g_ref": state["g_ref"].tolist()}

@app.get("/stats")
def stats(device_id: str = "imu01", scaled: int = 0):
    win = state["windows"].get(device_id)
    if not win or not win.ready():
        return {"ok": False, "error": "window not ready"}
    arr = win.array()  # (T,6) raw after mapping
    if scaled:
        arr = state["scaler"].transform(arr)
    st = window_stats(arr)
    return {"ok": True, "scaled": bool(scaled), "mean": st["mean"], "std": st["std"],
            "labels": ["gx","gy","gz","ax","ay","az"]}

@app.get("/latest")
def latest(device_id: str = "imu"):
    return JSONResponse(state["latest"].get(device_id, {"label":"NONE","confidence":0.0,"command":"NONE"}))

@app.get("/config")
def get_config():
    return {
        "threshold": state["threshold"],
        "effective_window": state["eff_window"],
        "classes": getattr(state["le"], "classes_", []).tolist()
        if getattr(state["le"], "classes_", None) is not None else []
    }

@app.get("/recent")
def recent(device_id: str = "imu01", n: int = 50):
    buf = list(state["recent"].get(device_id, []))
    if n > 0:
        buf = buf[-n:]
    return {"ok": True, "count": len(buf), "samples": buf}

@app.post("/swap_left_right")
def swap_left_right(device_id: str = "imu01"):
    # Rotate by 180° around training gravity to flip yaw (swap L/R) without changing gravity
    ensure_window(device_id)
    R = state["rot"].get(device_id, np.eye(3))
    Rpi = rot_axis_angle(state["g_ref"], np.pi)
    R_new = Rpi @ R
    state["rot"][device_id] = R_new
    save_rot(device_id, R_new)
    return {"ok": True, "message": "Applied 180° rotation around gravity (L/R swapped).", "R": R_new.tolist()}

@app.post("/flush_log")
def flush_log(out_dir: str | None = None):
    p = write_logs_csv(out_dir)
    if p is None:
        return {"ok": True, "written": 0, "path": None}
    # clear buffer after writing
    n = len(state.get("log_rows", []))
    state["log_rows"].clear()
    return {"ok": True, "written": n, "path": p}

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>IMU LSTM v2 Live</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 24px; }
    .row { margin: 8px 0; }
    .label { font-weight: 600; }
    .pill { display:inline-block; padding:4px 8px; border-radius:12px; background:#eee; }
    .cmd-LEFT { background:#cce5ff; }
    .cmd-RIGHT { background:#ffe5cc; }
    .cmd-STRAIGHT { background:#e5ffcc; }
    .cmd-JUMP { background:#fce4ec; }
    pre { background:#f7f7f7; padding:12px; border-radius:6px; max-width: 880px; overflow:auto; }
    table { border-collapse: collapse; }
    td, th { padding: 4px 8px; border: 1px solid #ddd; }
    button { margin-left: 8px; }
  </style>
</head>
<body>
  <h2>IMU LSTM v2 Live</h2>
  <div class="row">
    <span class="label">Device ID:</span>
    <input id="did" value="imu01" />
    <button onclick="tick()">Refresh</button>
    <button onclick="calib()">Calibrate gravity</button>
    <button onclick="swap()">Swap L/R</button>
    <button onclick="flush()">Flush CSV</button>
  </div>
  <div class="row">
    <span class="label">Command:</span>
    <span id="cmd" class="pill">NONE</span>
    <span class="label" style="margin-left:12px;">Label:</span>
    <span id="label">NONE</span>
    <span class="label" style="margin-left:12px;">Confidence:</span>
    <span id="conf">0.00</span>
  </div>
  <div class="row">
    <span class="label">Threshold:</span> <span id="th">-</span>
    <span class="label" style="margin-left:12px;">Window:</span> <span id="win">-</span>
    <span class="label" style="margin-left:12px;">Updated:</span> <span id="ts">-</span>
  </div>

  <h3>Probabilities</h3>
  <pre id="probs">-</pre>

  <h3>Raw IMU (latest sample)</h3>
  <table>
    <thead><tr><th>gx</th><th>gy</th><th>gz</th><th>ax</th><th>ay</th><th>az</th></tr></thead>
    <tbody id="raw_latest"><tr><td colspan="6">-</td></tr></tbody>
  </table>

  <h3>Raw IMU (window mean)</h3>
  <table>
    <thead><tr><th>gx</th><th>gy</th><th>gz</th><th>ax</th><th>ay</th><th>az</th></tr></thead>
    <tbody id="raw_mean"><tr><td colspan="6">-</td></tr></tbody>
  </table>

  <script>
    async function loadConfig() {
      try {
        const r = await fetch('/config');
        const j = await r.json();
        document.getElementById('th').textContent =
          (typeof j.threshold === 'number') ? j.threshold.toFixed(2) : j.threshold;
        document.getElementById('win').textContent = j.effective_window;
      } catch (e) {}
    }
    function cls(el, cmd){
      el.className = 'pill ' + (cmd ? ('cmd-' + cmd) : '');
    }
    function fmtTs(t){
      if (!t) return '-';
      const d = new Date(t*1000);
      return d.toLocaleTimeString();
    }
    function rowFor(vals){
      return '<tr>' + ['gx','gy','gz','ax','ay','az'].map(k => {
        const v = (typeof vals[k] === 'number') ? vals[k].toFixed(4) : '-';
        return '<td>'+v+'</td>';
      }).join('') + '</tr>';
    }
    async function tick(){
      const did = document.getElementById('did').value;
      const r = await fetch('/latest?device_id=' + encodeURIComponent(did));
      const j = await r.json();
      const cmd = (j.command || 'NONE').toUpperCase();
      document.getElementById('label').textContent = j.label || 'NONE';
      document.getElementById('conf').textContent = (j.confidence || 0).toFixed(3);
      document.getElementById('cmd').textContent = cmd;
      cls(document.getElementById('cmd'), cmd);
      document.getElementById('probs').textContent = JSON.stringify(j.probs || {}, null, 2);
      document.getElementById('ts').textContent = fmtTs(j.timestamp);
      if (j.raw_latest) document.getElementById('raw_latest').innerHTML = rowFor(j.raw_latest);
      if (j.raw_mean) document.getElementById('raw_mean').innerHTML = rowFor(j.raw_mean);
    }
    async function calib(){
      const did = document.getElementById('did').value;
      await fetch('/calibrate_gravity?device_id=' + encodeURIComponent(did), {method:'POST'});
      setTimeout(tick, 300);
    }
    async function swap(){
      const did = document.getElementById('did').value;
      await fetch('/swap_left_right?device_id=' + encodeURIComponent(did), {method:'POST'});
      setTimeout(tick, 300);
    }
    async function flush(){
      await fetch('/flush_log', {method:'POST'});
    }
    loadConfig();
    setInterval(tick, 500);
    tick();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http_port", type=int, default=5000)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--debounce", type=int, default=2, help="consecutive identical labels required to switch command")
    ap.add_argument("--jump_gate", type=float, default=12.0, help="suppress JUMP if vertical |acc| P90 < this (m/s^2)")
    # NEW: jump-specific tuning
    ap.add_argument("--threshold_jump", type=float, default=-1.0, help="threshold for JUMP (default: threshold)")
    ap.add_argument("--debounce_jump", type=int, default=1, help="debounce for JUMP")
    ap.add_argument("--jump_hold_ms", type=int, default=140, help="hold JUMP command for this many ms once triggered")
    ap.add_argument("--debounce_straight", type=int, default=2, help="debounce for STRAIGHT")
    ap.add_argument("--threshold_straight", type=float, default=-1.0, help="threshold for STRAIGHT (default: threshold-0.05)")
    ap.add_argument("--threshold_left", type=float, default=-1.0, help="threshold for LEFT (default: threshold)")
    ap.add_argument("--threshold_right", type=float, default=-1.0, help="threshold for RIGHT (default: threshold)")
    ap.add_argument("--quiet_gyro", type=float, default=0.06, help="mean |gyro| considered 'quiet' (rad/s)")
    ap.add_argument("--quiet_ms", type=int, default=250, help="ms of quiet needed to accept STRAIGHT immediately")
     # NEW
    ap.add_argument("--turn_fast_thr", type=float, default=0.90, help="immediate switch to non-straight if prob ≥ this")
    ap.add_argument("--turn_min_gyro", type=float, default=0.10, help="require mean |gyro| ≥ this for fast turn")

    ap.add_argument("--ema_alpha", type=float, default=0.6, help="EMA smoothing for class probabilities (0..1)")
    ap.add_argument("--yaw_prior_k", type=float, default=2.0, help="logit bias scale for yaw prior")
    ap.add_argument("--yaw_prior_min", type=float, default=0.08, help="|yaw| where prior starts (rad/s)")
    ap.add_argument("--yaw_prior_max", type=float, default=0.35, help="|yaw| where prior saturates (rad/s)")
    args = ap.parse_args()

    # apply runtime tuning
    state["debounce_k"] = max(1, int(args.debounce))
    state["min_acc_for_jump"] = float(args.jump_gate)
    state["debounce_map"]["default"] = max(1, int(args.debounce))
    state["debounce_map"]["straight"] = max(1, int(args.debounce_straight))
    state["debounce_map"]["jump"] = max(1, int(args.debounce_jump))   # NEW
    state["threshold_map"]["default"] = float(args.threshold)
    state["threshold_map"]["straight"] = (float(args.threshold_straight)
        if args.threshold_straight > 0 else max(0.0, float(args.threshold) - 0.05))
    state["threshold_map"]["left"] = (args.threshold_left if args.threshold_left > 0 else float(args.threshold))
    state["threshold_map"]["right"] = (args.threshold_right if args.threshold_right > 0 else float(args.threshold))
    state["threshold_map"]["jump"] = (args.threshold_jump if args.threshold_jump > 0 else float(args.threshold))  # NEW
    state["quiet_gyro"] = float(args.quiet_gyro)
    state["quiet_ms"] = int(args.quiet_ms)
    # NEW
    state["turn_fast_thr"] = float(args.turn_fast_thr)
    state["turn_min_gyro"] = float(args.turn_min_gyro)
    
    state["ema_alpha"] = float(args.ema_alpha)
    state["yaw_prior_k"] = float(args.yaw_prior_k)
    state["yaw_prior_min"] = float(args.yaw_prior_min)
    state["yaw_prior_max"] = float(args.yaw_prior_max)
    state["jump_hold_ms"] = int(args.jump_hold_ms)  # NEW

    init_model(Path(args.artifacts), threshold=args.threshold, window_override=args.window)
    print(
    f"[INFO] Debounce={state['debounce_map']['default']} (straight={state['debounce_map']['straight']}, jump={state['debounce_map']['jump']}) "
    f"Thr={state['threshold_map']['default']:.2f} (straight={state['threshold_map']['straight']:.2f}, "
    f"L={state['threshold_map']['left']:.2f}, R={state['threshold_map']['right']:.2f}, J={state['threshold_map']['jump']:.2f}) "
    f"JumpGate(P90)={state['min_acc_for_jump']} hold={state['jump_hold_ms']}ms "
    f"QuietGyro<{state['quiet_gyro']} for {state['quiet_ms']}ms FastTurn thr≥{state['turn_fast_thr']} gyro≥{state['turn_min_gyro']} "
    f"YawPrior assist={state['yaw_assist']} axis={state['yaw_axis']} sign={state['yaw_sign']} "
    f"win={state['yaw_window']} k={state['yaw_prior_k']:.2f} start≥{state['yaw_prior_min']:.2f} sat≥{state['yaw_prior_max']:.2f} "
    f"EMA α={state['ema_alpha']:.2f}"
    )
    uvicorn.run(app, host=args.host, port=args.http_port)