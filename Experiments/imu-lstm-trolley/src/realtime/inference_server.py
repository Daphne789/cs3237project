import os
import logging
from pathlib import Path
import re

import time
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque, Counter
import csv
import threading
from flask import jsonify

from src.models.lstm_model import EnhancedLSTM
from src.utils import (
    load_scaler_and_encoder,
    map_action_to_motor,
    accel_based_speed_scale,
    smooth_motor,
)
from src.realtime.motor_controller import send_motor_command
from src.data.loader import _add_features

# App / env
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("inference_server")
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(logging.INFO)

# MODEL_DIR = Path(
#     os.getenv(
#         "MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models_artifacts")
#     )
# )
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path.cwd() / "models_artifacts"))
PRED_LOG_PATH = os.getenv("PRED_LOG", str(MODEL_DIR / "predictions_log.csv"))
CKPT_PATH = Path(os.getenv("CKPT_PATH", MODEL_DIR / "imu_lstm_model_best.pth"))
WEIGHTS_ONLY_PATH = Path(
    os.getenv("WEIGHTS_ONLY_PATH", MODEL_DIR / "imu_lstm_model_weights_only.pth")
)

CHASSIS_IP = os.getenv("CHASSIS_IP", None)
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.25"))

device = "cuda" if torch.cuda.is_available() else "cpu"


# create CSV log file (one writer for whole process)
# _LOG_CSV_PATH = Path(os.getenv("PRED_LOG", MODEL_DIR / "predictions_log.csv"))
_LOG_CSV_PATH = Path(PRED_LOG_PATH)
_csv_lock = threading.Lock()
_LOG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
if not _LOG_CSV_PATH.exists():
    with _LOG_CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "frame_count",
                "pred_action",
                "voted_action",
                "confidence",
                "speed",
                "left",
                "right",
                "angle",
                "prob_jump",
                "prob_left",
                "prob_right",
                "prob_straight",
                "raw_seq_len",
            ]
        )


def _safe_torch_load(path, map_location):
    path = str(path)
    try:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    except Exception as first_exc:
        try:
            import numpy as _np

            candidates = [getattr(_np, "_core", None), getattr(_np, "core", None)]
            for mod in candidates:
                if not mod:
                    continue
                try:
                    scalar = getattr(mod, "multiarray").scalar
                    try:
                        torch.serialization.add_safe_globals([scalar])
                    except Exception:
                        pass
                except Exception:
                    pass
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
            except Exception as second_exc:
                raise RuntimeError(
                    f"torch.load failed: first={first_exc} second={second_exc}"
                ) from second_exc
        except Exception:
            raise RuntimeError(
                f"torch.load failed and allowlist attempt failed: {first_exc}"
            ) from first_exc


def _infer_lstm_hparams_from_state_dict(sd):
    """
    Infer input_size, hidden_size, num_layers, bidirectional from a saved state_dict `sd`.
    Returns tuple (input_size, hidden_size, num_layers, bidirectional)
    """
    # look for 'lstm.weight_ih_l0' key (or with module prefix)
    key_candidates = [k for k in sd.keys() if re.search(r"lstm\.weight_ih_l0($|\.)", k)]
    if not key_candidates:
        # also try plain 'weight_ih_l0' fallback
        key_candidates = [k for k in sd.keys() if re.search(r"weight_ih_l0($|\.)", k)]
    if not key_candidates:
        return None

    k0 = key_candidates[0]
    w = sd[k0]
    # w.shape == (4*hidden_size, input_size)
    hidden_times_4, input_size = w.shape
    hidden_size = int(hidden_times_4 // 4)

    # detect bidirectional by presence of reverse key
    has_reverse = any(
        re.search(r"lstm\.weight_ih_l0_reverse", k) for k in sd.keys()
    ) or any(re.search(r"weight_ih_l0_reverse", k) for k in sd.keys())
    # detect num_layers by scanning weight keys 'lstm.weight_ih_l{n}'
    layer_idxs = set()
    for k in sd.keys():
        m = re.search(r"lstm\.weight_ih_l(\d+)", k)
        if not m:
            m = re.search(r"weight_ih_l(\d+)", k)
        if m:
            layer_idxs.add(int(m.group(1)))
    num_layers = max(layer_idxs) + 1 if layer_idxs else 1

    return int(input_size), int(hidden_size), int(num_layers), bool(has_reverse)


def build_model_and_artifacts():
    """
    Load checkpoint, scaler and label encoder, build model and return (model, scaler, classes, seq_length).
    """
    if CKPT_PATH.exists():
        ckpt = _safe_torch_load(CKPT_PATH, map_location=device)
        classes = ckpt.get("label_classes", None)
        seq_length = int(ckpt.get("seq_length", 100))
    elif WEIGHTS_ONLY_PATH.exists():
        ckpt = _safe_torch_load(WEIGHTS_ONLY_PATH, map_location=device)
        classes = None
        seq_length = 100
    else:
        raise RuntimeError(f"No checkpoint found at {CKPT_PATH} or {WEIGHTS_ONLY_PATH}")

    # determine state_dict
    sd = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # maybe ckpt is already a state dict
        sd = ckpt
    else:
        sd = ckpt

    # infer model hparams from state_dict when possible
    inferred = _infer_lstm_hparams_from_state_dict(sd)
    if inferred is not None:
        input_size, hidden_size, num_layers, bidirectional = inferred
        LOG.info(
            "Inferred model hparams from checkpoint: input_size=%d hidden=%d layers=%d bidir=%s",
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
        )
    else:
        # fallback defaults (compatible with current training script)
        input_size, hidden_size, num_layers, bidirectional = 6, 128, 2, True
        LOG.warning("Could not infer model hparams from checkpoint; using defaults")

    scaler = None
    label_encoder = None
    try:
        scaler, label_encoder = load_scaler_and_encoder(MODEL_DIR)
        if classes is None and label_encoder is not None:
            classes = list(label_encoder.classes_)
    except Exception:
        LOG.warning("Failed to load scaler/label_encoder from %s", MODEL_DIR)

    # DEBUG: explicit stdout prints so they appear even if logging is suppressed
    print("MODEL_DIR resolved to:", MODEL_DIR)
    print("MODEL_DIR exists:", MODEL_DIR.exists())
    try:
        print("MODEL_DIR contents:", [p.name for p in MODEL_DIR.iterdir()])
    except Exception:
        print("Failed to list MODEL_DIR contents")
    print("scaler loaded:", scaler is not None)
    print("label_encoder loaded:", label_encoder is not None)
    if label_encoder is not None:
        print("label_encoder.classes_:", getattr(label_encoder, "classes_", None))

    LOG.info("Classes: %s", getattr(label_encoder, 'classes_', None))
    try:
        sc = scaler
        LOG.info("Scaler mean (first6): %s scale (first6): %s",
            np.round(sc.mean_[:6], 4).tolist() if hasattr(sc, "mean_") else None,
            np.round(sc.scale_[:6], 4).tolist() if hasattr(sc, "scale_") else None)
    except Exception:
        LOG.exception("failed printing scaler info")

    if classes is None:
        classes = ["jump", "left", "right", "straight"]
        LOG.warning("Using fallback classes: %s", classes)

    num_classes = len(classes)
    # construct model with inferred hparams so state_dict sizes match
    model = EnhancedLSTM(
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        num_classes=num_classes,
        bidirectional=bool(bidirectional),
    )
    # load weights
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        LOG.error("State dict load failed: %s", e)
        raise

    model.to(device).eval()
    return model, scaler, classes, seq_length


VOTE_WINDOW = int(os.getenv("VOTE_WINDOW", "7"))  # small odd number 3.7
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
_speed_clip_min = float(os.getenv("SPEED_MIN", "0.0"))
_speed_clip_max = float(os.getenv("SPEED_MAX", "0.4"))

# build model at import time (fail fast)
try:
    model, scaler, CLASSES, SEQ_LENGTH = build_model_and_artifacts()
    LOG.info(
        "Model loaded. classes=%s seq_length=%d device=%s", CLASSES, SEQ_LENGTH, device
    )
except Exception as e:
    LOG.exception("Failed to initialize model: %s", e)
    raise

# persistent smoothing state
_prev_motor = None
_pred_window = deque(maxlen=VOTE_WINDOW)


from collections import deque, Counter

# add near top after model built
VOTE_WINDOW = int(os.getenv("VOTE_WINDOW", "5"))  # small odd number 3..7
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
_speed_clip_min = float(os.getenv("SPEED_MIN", "0.0"))
_speed_clip_max = float(os.getenv("SPEED_MAX", "1.0"))

# persistent states
_prev_motor = None
_pred_window = deque(maxlen=VOTE_WINDOW)


@app.route("/predict", methods=["POST"])
def predict():
    global _prev_motor, _pred_window
    payload = request.get_json(force=True)
    frames = np.array(payload.get("frames", []), dtype=np.float32)

    if frames.ndim != 2 or frames.shape[1] != 6:
        return jsonify({"error": "frames must be Nx6 array"}), 400
    
    # DEBUG: log a short summary of raw frames received for quick verification
    try:
        if frames.size:
            LOG.info("recv frames shape=%s first_rows=%s mean=%s std=%s",
                     frames.shape,
                     frames[:3].tolist() if frames.shape[0] >= 1 else frames.tolist(),
                     np.round(frames.mean(axis=0), 4).tolist(),
                     np.round(frames.std(axis=0), 4).tolist())
    except Exception:
        LOG.exception("failed to log raw frames")

    raw_frames = frames.copy()
    # feature engineering as before
    try:
        frames_feat = _add_features(frames)
    except Exception:
        frames_feat = frames.copy()

    frames_scaled = frames_feat.copy()
    if scaler is not None:
        try:
            frames_scaled = scaler.transform(frames_scaled)
        except Exception:
            LOG.debug("scaler.transform failed; using unscaled features", exc_info=True)

    feat_dim = frames_scaled.shape[1]
    if len(frames_scaled) >= SEQ_LENGTH:
        seq = frames_scaled[-SEQ_LENGTH:]
    else:
        pad = np.zeros((SEQ_LENGTH - len(frames_scaled), feat_dim), dtype=np.float32)
        seq = np.vstack([frames_scaled, pad])

    x = torch.from_numpy(seq[np.newaxis]).float().to(device)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        pred_action = CLASSES[idx] if idx < len(CLASSES) else "unknown"
        pred_conf = float(probs[idx])

    # DEBUG: log label->prob mapping to verify ordering
    try:
        label_prob_map = {str(lbl): float(p) for lbl, p in zip(CLASSES, probs.tolist())}
        LOG.info("label_probs=%s", label_prob_map)
    except Exception:
        LOG.debug("failed to log label->prob map", exc_info=True)

    # push to vote window and compute mode
    _pred_window.append(pred_action)
    try:
        action = Counter(_pred_window).most_common(1)[0][0]
    except Exception:
        action = pred_action

    # safety: if confidence low, prefer STOP-like action ('jump') or scale down
    if pred_conf < CONF_THRESHOLD:
        # reduce aggressiveness: set action to jump (stop) OR keep but scale speed later
        # here we keep action but will scale speed down using pred_conf
        pass

    base_motor = map_action_to_motor(action)
    try:
        scale = accel_based_speed_scale(raw_frames)
    except Exception:
        scale = 1.0

    left_sp = float(base_motor.get("left", 0.0) * scale)
    right_sp = float(base_motor.get("right", 0.0) * scale)
    # overall speed is mean magnitude of wheels
    speed = (abs(left_sp) + abs(right_sp)) / 2.0
    # scale down speed if low confidence
    if pred_conf < CONF_THRESHOLD:
        speed *= pred_conf / CONF_THRESHOLD  # progressively reduce
    # clip to safe range
    speed = float(max(_speed_clip_min, min(_speed_clip_max, speed)))
    # Normalize per-wheel relative to requested speed keeping turn ratio
    if (abs(left_sp) + abs(right_sp)) > 1e-6:
        ratio_l = left_sp / (abs(left_sp) + abs(right_sp))
        ratio_r = right_sp / (abs(left_sp) + abs(right_sp))
        left_sp = ratio_l * speed * 2.0 if speed > 0 else 0.0
        right_sp = ratio_r * speed * 2.0 if speed > 0 else 0.0
    else:
        left_sp = right_sp = 0.0

    # clip each wheel to [-1,1] or [0,1] depending on how motor controller expects
    left_sp = float(max(-1.0, min(1.0, left_sp)))
    right_sp = float(max(-1.0, min(1.0, right_sp)))

    motor_scaled = {
        "left": left_sp,
        "right": right_sp,
        "angle": float(base_motor.get("angle", 0.0)),
        "speed": speed,
        "confidence": pred_conf,
        "voted_action": action,
    }

    motor_smoothed = smooth_motor(_prev_motor, motor_scaled, alpha=SMOOTH_ALPHA)
    _prev_motor = motor_smoothed

    LOG.info(
        "pred=%s voted=%s prob=%.3f scale=%.3f motor=%s",
        pred_action,
        action,
        pred_conf,
        scale,
        motor_smoothed,
    )

    chassis_resp = None
    if CHASSIS_IP:
        status, text = send_motor_command(
            CHASSIS_IP, {"action": action, "motor": motor_smoothed, "prob": pred_conf}
        )
        chassis_resp = {"status": status, "text": text}
        LOG.info("sent to chassis %s status=%s resp=%s", CHASSIS_IP, status, text)

    # # write CSV row
    # row = [
    #     time.time(),
    #     len(frames),           # number of frames sent in this request
    #     pred_action,
    #     action,
    #     float(pred_conf),
    #     float(motor_smoothed.get("speed", 0.0)),
    #     float(motor_smoothed.get("left", 0.0)),
    #     float(motor_smoothed.get("right", 0.0)),
    #     float(motor_smoothed.get("angle", 0.0)),
    #     float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3]),
    #     frames.shape[0]
    # ]
    # try:
    #     with _csv_lock:
    #         with _LOG_CSV_PATH.open("a", newline="") as f:
    #             csv.writer(f).writerow(row)
    # except Exception:
    #     LOG.exception("Failed to write prediction log")

    # optionally write CSV (skip if client requests log=0)
    if request.args.get("log", "1") != "0":
        row = [
            time.time(),
            len(frames),
            pred_action,
            action,
            float(pred_conf),
            float(motor_smoothed.get("speed", 0.0)),
            float(motor_smoothed.get("left", 0.0)),
            float(motor_smoothed.get("right", 0.0)),
            float(motor_smoothed.get("angle", 0.0)),
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
            float(probs[3]),
            frames.shape[0],
        ]
        try:
            with _csv_lock:
                with _LOG_CSV_PATH.open("a", newline="") as f:
                    csv.writer(f).writerow(row)
        except Exception:
            LOG.exception("Failed to write prediction log")

    return jsonify(
        {
            "action": action,
            "probabilities": probs.tolist(),
            "motor_command": motor_smoothed,
            "chassis_resp": chassis_resp,
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True}), 200

@app.route("/ping", methods=["GET"])
def ping():
    app.logger.info("PING from %s", request.remote_addr)
    return "pong", 200