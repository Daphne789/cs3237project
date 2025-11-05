import time, argparse, csv
from pathlib import Path
import pandas as pd
import numpy as np

def parse_plan(plan_str: str):
    steps=[]
    for part in plan_str.split(","):
        part=part.strip()
        if not part: continue
        name, dur = part.split(":")
        steps.append((name.strip(), float(dur)))
    return steps

def countdown(sec: int, label: str):
    print(f"Get ready: {label}")
    for s in range(sec, 0, -1):
        print(f"{s}...")
        time.sleep(1)
    print("GO")

def run_prompts(plan, warmup=3):
    print("Stage A live IMU test (motors disabled).")
    print("Ensure server is running and predictions_log.csv is being written.")
    input("Attach IMU, press Enter to start...")
    countdown(warmup, "starting session")
    t0 = time.time()
    schedule = []
    t = t0
    for (action, dur) in plan:
        print(f"DO: {action} for {dur} sec")
        seg_start = t
        time.sleep(dur)
        t = time.time()
        seg_end = t
        schedule.append({"action": action, "start": seg_start, "end": seg_end, "duration": seg_end-seg_start})
    t1 = time.time()
    print("Session done.")
    return t0, t1, schedule

def eval_against_log(pred_log_path: Path, t0: float, t1: float, schedule, accept=(0.85, 0.5, 0.15)):
    df = pd.read_csv(pred_log_path)
    for c in ["timestamp","confidence","speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)].copy()
    if df.empty:
        print("No prediction rows in time window; check server/logging times.")
        return pd.DataFrame(), {}
    t = df["timestamp"]
    dur = max(1e-6, float(t.max() - t.min()))
    va = df["voted_action"].astype(str)
    pa = df["pred_action"].astype(str) if "pred_action" in df.columns else va
    flips = int((va != va.shift(1)).sum())
    flips_sec = flips / dur
    disagree = float((pa != va).mean())
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    spd = pd.to_numeric(df["speed"], errors="coerce")
    dv = spd.diff().abs()
    overall = {
        "rows": int(len(df)),
        "duration_s": round(dur,3),
        "flips": flips,
        "flips_sec": round(flips_sec,3),
        "disagree_rate": round(disagree,3),
        "conf_mean": round(float(conf.mean()),3),
        "conf_p10": round(float(conf.quantile(0.10)),3),
        "conf_p90": round(float(conf.quantile(0.90)),3),
        "low_conf_rate": round(float((conf < 0.8).mean()),3),
        "p95_abs_dspeed": round(float(dv.quantile(0.95)),4),
    }
    rows=[]
    for seg in schedule:
        m = df[(df["timestamp"] >= seg["start"]) & (df["timestamp"] < seg["end"])].copy()
        if m.empty:
            rows.append({**seg, "frames": 0, "majority": None, "match_acc": None, "conf_mean": None, "p95_abs_dspeed": None})
            continue
        maj = m["voted_action"].astype(str).mode().iloc[0]
        acc = float((m["voted_action"].astype(str) == seg["action"]).mean())
        rows.append({
            **seg,
            "frames": int(len(m)),
            "majority": maj,
            "match_acc": round(acc,3),
            "conf_mean": round(float(pd.to_numeric(m["confidence"], errors="coerce").mean()),3),
            "p95_abs_dspeed": round(float(pd.to_numeric(m["speed"], errors="coerce").diff().abs().quantile(0.95)),4)
        })
    seg_df = pd.DataFrame(rows)
    OK_CONF, OK_FLIPS, OK_DV = accept
    pass_flag = (overall["conf_mean"] >= OK_CONF) and (overall["flips_sec"] <= OK_FLIPS) and (overall["p95_abs_dspeed"] <= OK_DV)
    print("--- Stage A overall ---")
    print(overall)
    print("--- Per segment (expected -> majority, acc) ---")
    for r in rows:
        print(f"{r['action']:>9} -> {str(r['majority']):>9} | acc={r['match_acc']} frames={r['frames']}")
    print("SUMMARY:", "PASS for Stage B (bench motors)" if pass_flag else "CAUTION – tune smoothing/voting before motors")
    return seg_df, overall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="straight:8,left:6,straight:6,right:6,jump:4,straight:6",
                    help="comma list action:seconds")
    ap.add_argument("--pred-log", default=str(Path.cwd() / "models_artifacts" / "predictions_log.csv"))
    ap.add_argument("--out-prefix", default=str(Path.cwd() / "models_artifacts" / "stageA"))
    args = ap.parse_args()
    plan = parse_plan(args.plan)
    t0, t1, schedule = run_prompts(plan)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ts = int(t0)
    sched_csv = out_prefix.with_name(out_prefix.name + f"_schedule_{ts}.csv")
    with sched_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["action","start","end","duration"])
        w.writeheader(); w.writerows(schedule)
    print("Saved schedule:", sched_csv)
    seg_df, _ = eval_against_log(Path(args.pred_log), t0, t1, schedule)
    if not seg_df.empty:
        eval_csv = out_prefix.with_name(out_prefix.name + f"_eval_{ts}.csv")
        seg_df.to_csv(eval_csv, index=False)
        print("Saved eval:", eval_csv)

if __name__ == "__main__":
    main()