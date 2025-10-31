import argparse, json, csv
from pathlib import Path
import math

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", required=True)
parser.add_argument("--out", dest="out", default=None)
args = parser.parse_args()

inp = Path(args.inp)
out = Path(args.out) if args.out else inp.with_name("replay_parsed.csv")

def try_load(s):
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    s = s.strip()
    # try JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # try replace single quotes
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        pass
    return {}

def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return None

rows = []
with inp.open("r", newline="") as f:
    r = csv.DictReader(f)
    for rec in r:
        resp_txt = rec.get("response", "")
        parsed = try_load(resp_txt)

        # If the server response was stored as {'response': '<json>'} (nested), handle that
        if isinstance(parsed, dict) and len(parsed)==1 and isinstance(list(parsed.values())[0], str):
            nested = try_load(list(parsed.values())[0])
            if nested:
                parsed = nested

        # top-level action
        pred_action = parsed.get("action") or parsed.get("pred_action") or parsed.get("pred") or None

        # motor may be in several places
        motor = parsed.get("motor_command") or parsed.get("motor") or parsed.get("motor_command_raw") or {}
        motor = try_load(motor)

        # voted_action may be inside motor or top-level
        voted_action = motor.get("voted_action") or parsed.get("voted_action") or parsed.get("voted") or pred_action

        # confidence can be top-level or inside motor
        confidence = motor.get("confidence")
        if confidence is None:
            confidence = parsed.get("confidence") or parsed.get("pred_conf") or None
        confidence = to_float(confidence)

        speed = to_float(motor.get("speed") or parsed.get("speed"))
        left = to_float(motor.get("left") or parsed.get("left"))
        right = to_float(motor.get("right") or parsed.get("right"))
        angle = to_float(motor.get("angle") or parsed.get("angle"))

        probs = parsed.get("probabilities") or parsed.get("probs") or None
        if isinstance(probs, str):
            probs = try_load(probs)
        prob_jump = prob_left = prob_right = prob_straight = None
        if isinstance(probs, (list, tuple)) and len(probs) >= 4:
            prob_jump, prob_left, prob_right, prob_straight = [to_float(x) for x in probs[:4]]
        else:
            # try individual keys
            prob_jump = to_float(parsed.get("prob_jump") or parsed.get("p_jump"))
            prob_left = to_float(parsed.get("prob_left") or parsed.get("p_left"))
            prob_right = to_float(parsed.get("prob_right") or parsed.get("p_right"))
            prob_straight = to_float(parsed.get("prob_straight") or parsed.get("p_straight"))

        rows.append({
            "action_id": rec.get("action_id"),
            "start": rec.get("start"),
            "len": rec.get("len"),
            "request_time": rec.get("request_time"),
            "pred_action": pred_action,
            "voted_action": voted_action,
            "confidence": confidence,
            "speed": speed,
            "left": left,
            "right": right,
            "angle": angle,
            "prob_jump": prob_jump,
            "prob_left": prob_left,
            "prob_right": prob_right,
            "prob_straight": prob_straight,
            "raw_response": resp_txt
        })

if not rows:
    print("no rows parsed in", inp)
else:
    with out.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out.resolve())