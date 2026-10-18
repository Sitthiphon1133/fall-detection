"""
Data recorder: MediaPipe keypoints -> CSV (one row per frame) with live labels.

Usage (keep this file in the same folder as fall_detect_phase2.py):
    python record_data.py --person P01 --activity walk
    python record_data.py --person P01 --activity fall --save-video

Keys in the video window:
    SPACE = start (after a 3 s countdown) / stop recording
    0 = label "normal" | 1 = label "fall" | 2 = label "lying_still" (abnormal) | q = quit

Labels are per frame and stay until you press another label key.
Each recording starts with label 0.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Reuse the model path, camera index and mirror setting from the main program,
# so the recorded data matches what the real-time system sees.
from fall_detect import CAMERA_INDEX, MIRROR, MODEL_PATH, ensure_model

ACTIVITIES = ["walk", "sit_fast", "bend", "squat", "sofa", "fall"]
LABEL_NAMES = {0: "normal", 1: "fall", 2: "lying_still"}
COUNTDOWN_SEC = 3.0
N_LANDMARKS = 33


def csv_header():
    cols = ["frame", "t", "person", "activity", "label", "detected", "img_w", "img_h"]
    for i in range(N_LANDMARKS):
        cols += [f"x{i}", f"y{i}", f"v{i}"]
    return cols


def make_row(idx, t, person, activity, label, lm, w, h):
    """One CSV row. If no person was detected, the landmark columns are left empty."""
    row = [idx, f"{t:.4f}", person, activity, label, 0 if lm is None else 1, w, h]
    if lm is None:
        row += [""] * (N_LANDMARKS * 3)
    else:
        for p in lm:
            row += [f"{p.x:.5f}", f"{p.y:.5f}", f"{(p.visibility or 0.0):.3f}"]
    return row


class Recorder:
    def __init__(self, out_dir, person, activity, save_video):
        self.out_dir, self.person, self.activity, self.save_video = out_dir, person, activity, save_video
        self.file = self.writer = self.video = None
        self.path = None

    @property
    def active(self):
        return self.file is not None

    def start(self, frame_shape, fps=30.0):
        h, w = frame_shape[:2]
        base = f"{self.person}_{self.activity}_{datetime.now():%Y%m%d_%H%M%S}"
        self.path = os.path.join(self.out_dir, base + ".csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(csv_header())
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video = cv2.VideoWriter(os.path.join(self.out_dir, base + ".mp4"), fourcc, fps, (w, h))
        self.idx = 0
        self.detected = 0
        self.t0 = time.time()
        self.label_counts = Counter()
        print(f"Recording -> {self.path}")

    def write(self, now, label, lm, frame):
        h, w = frame.shape[:2]
        self.writer.writerow(make_row(self.idx, now - self.t0, self.person, self.activity, label, lm, w, h))
        if self.video is not None:
            self.video.write(frame)          # raw frame, without overlay
        self.idx += 1
        self.detected += 0 if lm is None else 1
        self.label_counts[label] += 1

    def elapsed(self):
        return time.time() - self.t0

    def stop(self):
        if not self.active:
            return
        self.file.close()
        if self.video is not None:
            self.video.release()
        dur = time.time() - self.t0
        rate = 100.0 * self.detected / max(self.idx, 1)
        print(f"Saved {self.path}")
        print(f"  frames: {self.idx} | duration: {dur:.1f} s | person detected in {rate:.0f}% of frames")
        for k in sorted(self.label_counts):
            print(f"  label {k} ({LABEL_NAMES[k]}): {self.label_counts[k]} frames")
        if rate < 80:
            print("  WARNING: low detection rate. Check lighting / camera position and consider retaking.")
        if self.activity == "fall" and self.label_counts[1] == 0:
            print("  WARNING: no frames labelled 1 (fall). Did you press the 1 key at the moment of the fall?")
        self.file = self.writer = self.video = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="performer ID, e.g. P01 (no real names)")
    ap.add_argument("--activity", required=True, choices=ACTIVITIES)
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    ap.add_argument("--save-video", action="store_true", help="also save the raw video (delete it after extracting keypoints)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Cannot open the camera. Try changing CAMERA_INDEX in fall_detect_phase2.py")

    if args.activity == "fall":
        print("SAFETY: falls must be performed on a mattress / thick cushions only.")
    print(f"Person {args.person} | activity {args.activity} | SPACE = start/stop, 0/1/2 = label, q = quit")

    rec = Recorder(args.out, args.person, args.activity, args.save_video)
    state = "idle"                 # idle -> countdown -> rec
    countdown_end = 0.0
    label = 0
    start = time.time()
    last_ts_ms = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = max(int((now - start) * 1000), last_ts_ms + 1)   # timestamps must be strictly increasing
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)
            lm = result.pose_landmarks[0] if result.pose_landmarks else None

            if state == "countdown" and now >= countdown_end:
                rec.start(frame.shape)
                label = 0
                state = "rec"
            if state == "rec":
                rec.write(now, label, lm, frame)

            # ------- overlay -------
            if lm is not None:
                for p in lm:
                    if (p.visibility or 0.0) >= 0.5:
                        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
            if state == "idle":
                status, color = "IDLE (SPACE to start)", (200, 200, 200)
            elif state == "countdown":
                status, color = f"GET READY {max(countdown_end - now, 0):.1f}", (0, 200, 255)
            else:
                status, color = f"REC {rec.elapsed():5.1f}s", (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"{args.person} / {args.activity}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"label {label}: {LABEL_NAMES[label]}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "person: " + ("yes" if lm is not None else "NO"), (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if lm is not None else (0, 0, 255), 2)
            cv2.imshow("Data recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if state == "idle":
                    state, countdown_end = "countdown", time.time() + COUNTDOWN_SEC
                elif state == "countdown":
                    state = "idle"
                else:
                    rec.stop()
                    state = "idle"
            elif key in (ord("0"), ord("1"), ord("2")):
                label = int(chr(key))
    finally:
        rec.stop()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
