"""
Build a per-frame feature table from the recorded keypoint CSVs.

It reuses the feature code of the real-time program (extract_features, MotionEstimator and the
thresholds in your Phase 2 file), so the training features match what the live system computes.

Usage (run from the project folder, next to your Phase 2 file):
    python build_features.py
    python build_features.py --data data --out features.csv
    python build_features.py --module fall_detect_phase2      # name of the Phase 2 file without .py

Output: features.csv with one row per frame:
    file, person, activity, frame, t, label, valid,
    theta, hy, v, a, r, t_still, thigh, knee, lying
(empty cell = value not available, e.g. person not detected or legs not visible)
"""
import argparse
import csv
import glob
import importlib
import math
import os
import sys
from collections import defaultdict

import numpy as np

N_LM = 33
JUMP_MAX_CONSEC = 5     # after this many rejected frames in a row, accept the new position (tracker was right)
OUT_COLS = ["file", "person", "activity", "frame", "t", "label", "valid",
            "theta", "hy", "v", "a", "r", "t_still", "thigh", "knee", "lying"]


class LM:
    """Minimal landmark object (same attributes as a MediaPipe landmark)."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, v):
        self.x, self.y, self.visibility = x, y, v


def load_feature_module(name):
    sys.path.insert(0, os.getcwd())
    try:
        F = importlib.import_module(name)
    except ImportError as e:
        raise SystemExit(f"Cannot import '{name}' ({e}).\n"
                         "Put your Phase 2 file in this folder and pass its name, e.g. --module fall_detect")
    for attr in ("extract_features", "MotionEstimator", "UPRIGHT_DEG", "STILL_SPEED"):
        if not hasattr(F, attr):
            raise SystemExit(f"'{name}' has no '{attr}'. It must be the Phase 2 file, not the Phase 1 file.")
    return F


def fmt(x, nd=5):
    return "" if x is None else f"{x:.{nd}f}"


def process_file(path, F, drop_missing=True, max_jump=10.0, label_lead=0.0):
    """Return (list of output rows, metadata dict) for one recording.

    With drop_missing=True, timestamps that have no usable coordinates (no person detected, empty or
    unparsable landmark cells, or too little visibility for extract_features) are NOT written to the
    output. They are still passed through the running state (previous time, t_still reset), so the
    features of the kept frames are identical to the ones the live system computes.

    max_jump (torso lengths/s, 0 = off): causal jump rejection. If the hip moves between two accepted
    frames faster than this, the frame is treated as a tracker glitch and dropped. After JUMP_MAX_CONSEC
    rejections in a row the new position is accepted and the motion window is reset.
    label_lead (s, 0 = off): in fall files, also label this many seconds before the first label-1 frame as 1
    (compensates the reaction delay of a human pressing the key).
    """
    out = []
    motion = F.MotionEstimator()
    L_ref = None
    t_still = 0.0
    prev_t = None
    min_v = 0.0
    n_raw = n_dropped = n_jump = 0
    consec = 0
    ok_t = ok_hy = ok_hx = None
    last_t = 0.0
    person = activity = "?"
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        idx = {n: i for i, n in enumerate(header)}
        x0 = idx["x0"]
        for row in rd:
            t = float(row[idx["t"]])
            dt = (t - prev_t) if prev_t is not None else 1 / 30
            prev_t = t
            last_t = t
            if n_raw == 0:
                person, activity = row[idx["person"]], row[idx["activity"]]
            n_raw += 1
            w, h = int(row[idx["img_w"]]), int(row[idx["img_h"]])
            label = int(row[idx["label"]])
            lm = None
            if row[idx["detected"]] == "1":
                try:
                    lm = [LM(float(row[x0 + 3 * i]), float(row[x0 + 3 * i + 1]), float(row[x0 + 3 * i + 2]))
                          for i in range(N_LM)]
                except ValueError:                              # empty / broken coordinate cells
                    lm = None
            f = F.extract_features(lm, w, h, L_ref) if lm is not None else None

            base = [os.path.basename(path), row[idx["person"]], row[idx["activity"]], row[idx["frame"]],
                    f"{t:.4f}", label]
            if f is None:
                t_still = 0.0
                n_dropped += 1
                if not drop_missing:
                    out.append(base + [0] + [""] * 9)
                continue
            if max_jump > 0 and ok_t is not None and t - ok_t <= 0.5:
                jump = math.hypot(f["hy"] - ok_hy, f["hx"] - ok_hx) / max(t - ok_t, 0.02)
                if jump > max_jump:
                    if consec < JUMP_MAX_CONSEC:                # glitch: ignore this frame
                        consec += 1
                        n_jump += 1
                        t_still = 0.0
                        if not drop_missing:
                            out.append(base + [0] + [""] * 9)
                        continue
                    motion.reset()                              # persistent move: treat as a new track
            consec = 0
            ok_t, ok_hy, ok_hx = t, f["hy"], f["hx"]
            if f["theta"] < F.UPRIGHT_DEG:                      # same L_ref update rule as the live system
                L_ref = f["L_cur"] if L_ref is None else 0.95 * L_ref + 0.05 * f["L_cur"]
            v, a, speed = motion.update(t, f["hy"], f["hx"])
            t_still = t_still + dt if speed < F.STILL_SPEED else 0.0
            min_v = min(min_v, v)
            out.append(base + [1, fmt(f["theta"], 2), fmt(f["hy"], 4), fmt(v, 4), fmt(a, 3), fmt(f["r"], 4),
                               fmt(t_still, 3), fmt(f["thigh"], 2), fmt(f["knee"], 2), int(f["lying"])])
    if label_lead > 0 and activity == "fall":
        on = next((i for i, r in enumerate(out) if r[5] == 1), None)
        if on is not None:
            t_on = float(out[on][4])
            for r in out[:on]:
                if r[5] == 0 and t_on - float(r[4]) <= label_lead:
                    r[5] = 1
    meta = dict(file=os.path.basename(path), person=person, activity=activity, min_v=min_v,
                frames=sum(1 for r in out if r[6] == 1), raw_frames=n_raw, dropped=n_dropped, jumps=n_jump, duration=last_t)
    return out, meta


def summarize(all_rows, metas):
    print("\n================ DATASET SUMMARY ================")
    persons = sorted({m["person"] for m in metas})
    print(f"files: {len(metas)} | persons: {len(persons)} ({', '.join(persons)})")
    raw = sum(m["raw_frames"] for m in metas)
    kept = sum(m["frames"] for m in metas)
    miss = sum(m["dropped"] for m in metas)
    jumps = sum(m["jumps"] for m in metas)
    print(f"frames: raw {raw} | kept {kept} | no coordinates {miss} ({100.0 * miss / max(raw, 1):.0f}%) | "
          f"rejected as jumps {jumps} | recorded time: {sum(m['duration'] for m in metas) / 60:.1f} min")
    for m in metas:
        if m["jumps"] >= 3:
            print(f"  - {m['file']}: {m['jumps']} frames rejected as jumps")

    lab = defaultdict(int)
    for r in all_rows:
        lab[r[5]] += 1
    print("frames per label:", {k: lab[k] for k in sorted(lab)})

    print("\nfiles per person x activity:")
    table = defaultdict(lambda: defaultdict(int))
    for m in metas:
        table[m["person"]][m["activity"]] += 1
    acts = sorted({m["activity"] for m in metas})
    print("  person  " + "  ".join(f"{a:>8}" for a in acts))
    for p in persons:
        print(f"  {p:<7} " + "  ".join(f"{table[p][a]:>8}" for a in acts))

    # How far does the hip drop in each recording? Use this to tune V_FALL.
    fall_drop = [-m["min_v"] for m in metas if m["activity"] == "fall"]
    other_drop = [-m["min_v"] for m in metas if m["activity"] != "fall"]
    print("\nlargest downward hip speed per file (torso lengths/s), use it to tune V_FALL:")
    if fall_drop:
        print(f"  fall files   : min {min(fall_drop):.2f} | median {np.median(fall_drop):.2f} | max {max(fall_drop):.2f}")
    if other_drop:
        print(f"  other files  : min {min(other_drop):.2f} | median {np.median(other_drop):.2f} | max {max(other_drop):.2f}")
    if other_drop:
        top = sorted(((-m["min_v"], m["file"]) for m in metas if m["activity"] != "fall"), reverse=True)[:5]
        print("  fastest non-fall files (check these for tracker glitches):")
        for d, name in top:
            print(f"    {d:5.2f}  {name}")
    if fall_drop and other_drop:
        lo, hi = max(other_drop), min(fall_drop)
        if hi > lo:
            print(f"  -> separable: any V_FALL between {lo:.2f} and {hi:.2f} works on this data "
                  f"(try about {(lo + hi) / 2:.2f})")
        else:
            print(f"  -> overlapping (other files reach {lo:.2f}, slowest fall is {hi:.2f}): "
                  "speed alone cannot separate them, rely on the lying + still conditions too")

    print("\nwarnings:")
    warned = False
    if len(persons) < 3:
        print("  - fewer than 3 persons: a person-wise train/test split will be very weak")
        warned = True
    n_fall = sum(1 for m in metas if m["activity"] == "fall")
    if n_fall < 15:
        print(f"  - only {n_fall} fall recordings (aim for 15 or more)")
        warned = True
    for m in metas:
        if m["activity"] == "fall" and not any(r[0] == m["file"] and r[5] == 1 for r in all_rows):
            print(f"  - {m['file']}: no frame labelled 1 (fall), check the labels")
            warned = True
    low = [m["file"] for m in metas if m["frames"] + m["jumps"] < 0.8 * m["raw_frames"]]
    for f_ in low:
        print(f"  - {f_}: person detected in less than 80% of frames")
        warned = True
    if not warned:
        print("  none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with the recorded CSV files")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--module", default="fall_detect", help="Phase 2 file name without .py")
    ap.add_argument("--max-jump", type=float, default=10.0,
                    help="reject frames where the hip moves faster than this (torso lengths/s); 0 = off")
    ap.add_argument("--label-lead", type=float, default=0.0,
                    help="seconds before the first label-1 frame to also label as 1 in fall files (default 0 = off)")
    ap.add_argument("--keep-missing", action="store_true",
                    help="keep timestamps without coordinates as empty rows (valid = 0) instead of dropping them")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in '{args.data}'. Record some data first with record_data.py")
    F = load_feature_module(args.module)

    all_rows, metas = [], []
    for p in files:
        rows, meta = process_file(p, F, drop_missing=not args.keep_missing,
                                max_jump=args.max_jump, label_lead=args.label_lead)
        if not rows:
            print("skipped file with no usable frames:", p)
            continue
        all_rows += rows
        metas.append(meta)
        print(f"processed {meta['file']}: kept {meta['frames']} of {meta['raw_frames']} frames")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(OUT_COLS)
        w.writerows(all_rows)
    print(f"\nSaved {args.out} ({len(all_rows)} rows)")
    summarize(all_rows, metas)


if __name__ == "__main__":
    main()