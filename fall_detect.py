"""
Fall Detection - Phase 2
Camera -> MediaPipe Pose -> Normalize (3x3 matrix: T -> S -> R)
      -> features (trunk angle, hip height, v/a from linear regression, r, still time, legs)
      -> state machine (NORMAL -> SUSPECT -> LYING -> ALERT) -> Telegram + "I'm OK" button

Hotkeys in the video window:  q = quit | o = simulate pressing "I'm OK" | c = recalibrate torso length (stand upright first)
Test the math without opening the camera:  python fall_detect_phase2.py --selftest
"""
import json
import math
import os
import ssl
import sys
import threading
import time
import urllib.request
from collections import deque

import certifi
import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ===================== Settings (all values are examples, tune them with real data) =====================
CAMERA_INDEX = 0
MIRROR = True                # flip the image left-right
CAMERA_ROLL_DEG = 0.0        # compensate camera tilt (degrees); if the camera is tilted, try small +/- values

BOT_TOKEN = ""               # leave empty = no Telegram, only print/show on screen
CHAT_ID = ""

MIN_VISIBILITY = 0.5         # ignore keypoints with lower confidence than this
WINDOW_SEC = 0.4             # time window for regression (longer = smoother but slower to react)
MIN_SAMPLES = 4              # minimum number of samples before fitting a line

ANGLE_LYING = 60.0           # degrees: trunk tilt above this counts as a lying/fallen posture
THIGH_LYING_MIN = 45.0       # degrees: if legs are visible, thighs must tilt beyond this to count as lying (not just bending)
V_FALL = 1.4                 # downward hip speed (torso lengths/second) that counts as a "fast drop"
SUSPECT_TIMEOUT = 3.0        # seconds: after a fast drop, a lying posture must follow within this time, otherwise it is not a fall
STILL_SPEED = 0.25           # torso lengths/second: slower than this counts as "still"
STILL_SECONDS = 3.0          # T: stay still in a lying posture this long before alerting (can lower to 3 for testing)
RECOVER_SEC = 1.5            # normal posture for this long = the person has got up
UPRIGHT_DEG = 25.0           # trunk more upright than this -> allowed to update the reference torso length L_ref
# ========================================================================================================

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

LS, RS, LH, RH = 11, 12, 23, 24      # left/right shoulder, left/right hip
LK, RK, LA, RA = 25, 26, 27, 28      # left/right knee, left/right ankle

NORMAL, SUSPECT, LYING, ALERT = "NORMAL", "SUSPECT", "LYING", "ALERT"
STATE_COLOR = {NORMAL: (0, 200, 0), SUSPECT: (0, 200, 255), LYING: (0, 140, 255), ALERT: (0, 0, 255)}


# ---------------------------------------------------------------- Math ----------------------
def gauss_solve(A, b):
    """Solve A x = b with Gaussian elimination (with partial pivoting)."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        p = i + int(np.argmax(np.abs(M[i:, i])))
        if abs(M[p, i]) < 1e-12:
            raise ValueError("singular matrix")
        M[[i, p]] = M[[p, i]]
        for j in range(i + 1, n):
            M[j] -= (M[j, i] / M[i, i]) * M[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
    return x


def fit_line(ts, ys):
    """Least squares: y = a + b*t, solving the normal equation (X^T X) theta = X^T y. Returns (a, b)."""
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    X = np.column_stack([np.ones(len(ts)), ts - ts[0]])   # shift time to start at 0 for numerical stability
    a, b = gauss_solve(X.T @ X, X.T @ ys)
    return a, b


def make_transform(origin, L, alpha_deg):
    """3x3 homogeneous matrix:  M = R . S . T  (apply T first, then S, then R)."""
    ox, oy = origin
    al = math.radians(alpha_deg)
    T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], dtype=float)
    S = np.array([[1 / L, 0, 0], [0, 1 / L, 0], [0, 0, 1]], dtype=float)
    R = np.array([[math.cos(al), -math.sin(al), 0], [math.sin(al), math.cos(al), 0], [0, 0, 1]], dtype=float)
    return R @ S @ T


def angle_between(u, v):
    """Angle (degrees) between two vectors using the dot product."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def selftest():
    print("== Compare with the worked example in section 8 of the document ==")
    a, b = fit_line([0, 1, 2, 3, 4], [0.55, 0.57, 0.62, 0.69, 0.78])
    print(f"Regression: a = {a:.3f} (doc 0.526), b = {b:.3f} (doc 0.058)")
    M = make_transform((0.50, 0.55), 0.2508, 0.0)
    p = M @ np.array([0.52, 0.30, 1.0])
    print(f"Normalized shoulder (standing): ({p[0]:.3f}, {p[1]:.3f})  (doc ≈ (0.080, -0.997))")
    e = np.array([0.0, -1.0])
    print(f"Angle, standing: {angle_between(np.array([0.02, -0.25]), e):.1f}°  (doc ≈ 4.6°)")
    print(f"Angle, lying: {angle_between(np.array([-0.25, -0.02]), e):.1f}°  (doc ≈ 85.4°)")
    M1 = make_transform((0.5, 0.55), 0.25, 30.0)
    T = np.array([[1, 0, -0.5], [0, 1, -0.55], [0, 0, 1]])
    S = np.diag([1 / 0.25, 1 / 0.25, 1.0])
    al = math.radians(30)
    R = np.array([[math.cos(al), -math.sin(al), 0], [math.sin(al), math.cos(al), 0], [0, 0, 1]])
    print("R·S·T == make_transform:", np.allclose(M1, R @ S @ T), "| T·S·R equal?:", np.allclose(M1, T @ S @ R))


# ---------------------------------------------------------------- Model / Telegram ----------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading the MediaPipe Pose model (first run only)...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, "wb") as f:
        f.write(r.read())
    print("Download complete")


ok_event = threading.Event()          # set when the user presses "I'm OK"
stop_event = threading.Event()


def _api(method):
    return f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"


def tg_enabled():
    return bool(BOT_TOKEN and CHAT_ID)


def send_text(text, with_button=False):
    if not tg_enabled():
        print("[Telegram not configured]", text)
        return
    payload = {"chat_id": CHAT_ID, "text": text}
    if with_button:
        payload["reply_markup"] = {"inline_keyboard": [[{"text": "I'm OK", "callback_data": "ok"}]]}

    def _send():
        try:
            r = requests.post(_api("sendMessage"), json=payload, timeout=10)
            print("Telegram status:", r.status_code)
        except Exception as e:
            print("Telegram send failed:", e)

    threading.Thread(target=_send, daemon=True).start()


def poll_loop():
    """Poll Telegram for button presses / messages (long polling) in a separate thread."""
    offset = None
    try:
        res = requests.get(_api("getUpdates"), params={"offset": -1, "timeout": 0}, timeout=10).json()
        items = res.get("result", [])
        if items:
            offset = items[-1]["update_id"] + 1      # skip old pending messages
    except Exception as e:
        print("Failed to start polling:", e)

    while not stop_event.is_set():
        try:
            params = {"timeout": 20}
            if offset is not None:
                params["offset"] = offset
            res = requests.get(_api("getUpdates"), params=params, timeout=30).json()
            for upd in res.get("result", []):
                offset = upd["update_id"] + 1
                cb = upd.get("callback_query")
                if cb and str(cb.get("message", {}).get("chat", {}).get("id")) == str(CHAT_ID):
                    if cb.get("data") == "ok":
                        ok_event.set()
                    requests.post(_api("answerCallbackQuery"), json={"callback_query_id": cb["id"]}, timeout=10)
                msg = upd.get("message")
                if msg and str(msg.get("chat", {}).get("id")) == str(CHAT_ID):
                    text = (msg.get("text") or "").strip().lower()
                    if text in ("ok", "okay", "i'm ok", "im ok"):
                        ok_event.set()
        except Exception:
            time.sleep(2)


# ---------------------------------------------------------------- Features ------------------
def extract_features(lm, w, h, L_ref):
    """Extract the features computable from a single frame. Returns a dict, or None if the main keypoints are unclear."""
    vis = np.array([(p.visibility or 0.0) for p in lm])
    if vis[[LS, RS, LH, RH]].min() < MIN_VISIBILITY:
        return None

    P = np.array([[p.x * w, p.y * h, 1.0] for p in lm]).T          # 3 x 33 matrix of homogeneous pixel coordinates
    s_px = (P[:2, LS] + P[:2, RS]) / 2                              # shoulder midpoint
    h_px = (P[:2, LH] + P[:2, RH]) / 2                              # hip midpoint
    L_cur = float(np.linalg.norm(s_px - h_px))                      # current torso length in pixels
    if L_cur < 1e-6:
        return None
    L = L_ref if L_ref else L_cur

    # 1) Normalize the pose: hip at the origin, unit = torso length
    Pn = make_transform(h_px, L, CAMERA_ROLL_DEG) @ P
    s_n = (Pn[:2, LS] + Pn[:2, RS]) / 2                              # hip is at (0,0), so trunk vector t' = s_n
    theta = angle_between(s_n, np.array([0.0, -1.0]))                 # trunk angle from vertical (e = up)
    if theta is None:
        return None

    # 2) Hip height: same matrix but with origin = bottom-center of the image (not the hip, otherwise hy would always be 0)
    hf = make_transform((w / 2, h), L, CAMERA_ROLL_DEG) @ np.array([h_px[0], h_px[1], 1.0])
    hy, hx = float(-hf[1]), float(hf[0])

    # 3) Aspect ratio (width/height) of the whole-body keypoint box
    ok = vis >= MIN_VISIBILITY
    r = None
    if ok.sum() >= 6:
        xs, ys = Pn[0, ok], Pn[1, ok]
        r = float((xs.max() - xs.min()) / max(ys.max() - ys.min(), 1e-6))

    # 4) Legs (if clearly visible): thigh angle from vertical and knee angle
    thighs, knees, leg_lines = [], [], []
    for hip_i, knee_i, ank_i in ((LH, LK, LA), (RH, RK, RA)):
        if vis[[hip_i, knee_i, ank_i]].min() >= MIN_VISIBILITY:
            hip_n, knee_n, ank_n = Pn[:2, hip_i], Pn[:2, knee_i], Pn[:2, ank_i]
            th = angle_between(knee_n - hip_n, np.array([0.0, 1.0]))   # hip->knee vs. downward vertical
            kn = angle_between(hip_n - knee_n, ank_n - knee_n)         # angle at the knee (straight leg ≈ 180°)
            if th is not None and kn is not None:
                thighs.append(th)
                knees.append(kn)
            leg_lines.append((P[:2, hip_i], P[:2, knee_i], P[:2, ank_i]))
    thigh = float(np.mean(thighs)) if thighs else None
    knee = float(np.mean(knees)) if knees else None

    lying = theta > ANGLE_LYING and (thigh is None or thigh > THIGH_LYING_MIN)
    return dict(theta=theta, hy=hy, hx=hx, r=r, thigh=thigh, knee=knee, lying=lying,
                L_cur=L_cur, s_px=s_px, h_px=h_px, leg_lines=leg_lines)


class MotionEstimator:
    """Keep a time window of hip positions and estimate v and a with linear regression."""

    def __init__(self):
        self.samples = deque()   # (t, hy, hx)
        self.vhist = deque()     # (t, v)

    def reset(self):
        self.samples.clear()
        self.vhist.clear()

    def update(self, t, hy, hx):
        if self.samples and t - self.samples[-1][0] > 0.5:   # data was missing for a while, drop stale samples
            self.reset()
        self.samples.append((t, hy, hx))
        while t - self.samples[0][0] > WINDOW_SEC:
            self.samples.popleft()
        if len(self.samples) < MIN_SAMPLES:
            return 0.0, 0.0, 99.0
        ts, ys, xs = zip(*self.samples)
        _, v = fit_line(ts, ys)          # vertical speed (positive = up, negative = dropping)
        _, vx = fit_line(ts, xs)         # horizontal speed
        self.vhist.append((t, v))
        while t - self.vhist[0][0] > WINDOW_SEC:
            self.vhist.popleft()
        a = 0.0
        if len(self.vhist) >= MIN_SAMPLES:
            vt, vv = zip(*self.vhist)
            _, a = fit_line(vt, vv)      # acceleration = slope of v over time
        return v, a, math.hypot(vx, v)


# ---------------------------------------------------------------- State machine -------------
class FallStateMachine:
    def __init__(self):
        self.state = NORMAL
        self.t_suspect = 0.0
        self.t_not_lying = None
        self.t_still = 0.0
        self.disarmed = False      # after pressing "I'm OK", the person must get up before detection re-arms

    def _go(self, state):
        self.state = state
        self.t_not_lying = None
        if state == NORMAL:
            self.t_still = 0.0

    def cancel(self):
        if self.state != NORMAL:
            self._go(NORMAL)
            self.disarmed = True
            return True
        return False

    def update(self, f, v, speed, now, dt):
        """f = None when no person is seen / main keypoints are unclear. Returns 'ALERT' / 'RECOVERED' / None."""
        lost = f is None
        lying = (not lost) and f["lying"]

        # Still-time counter (if detection is lost while in the lying state, count it as still,
        # because a person who has fallen is often not detected by the pose model)
        if lost:
            self.t_still = self.t_still + dt if self.state == LYING else 0.0
        elif speed < STILL_SPEED:
            self.t_still += dt
        else:
            self.t_still = 0.0

        if self.state == NORMAL:
            if lost:
                return None
            if self.disarmed:
                if not lying:
                    self.disarmed = False
                return None
            if v <= -V_FALL:                        # fast drop
                self._go(SUSPECT)
                self.t_suspect = now
            return None

        if self.state == SUSPECT:
            if lying:
                self._go(LYING)
                self.t_still = 0.0
            elif now - self.t_suspect > SUSPECT_TIMEOUT:
                self._go(NORMAL)                    # bending / sitting down fast, not a fall
            return None

        # LYING or ALERT: check whether the person has got up
        if lying or lost:
            self.t_not_lying = None
        elif self.t_not_lying is None:
            self.t_not_lying = now
        elif now - self.t_not_lying >= RECOVER_SEC:
            was_alert = self.state == ALERT
            self._go(NORMAL)
            return "RECOVERED" if was_alert else None

        if self.state == LYING and self.t_still >= STILL_SECONDS:
            self._go(ALERT)
            return "ALERT"
        return None


# ---------------------------------------------------------------- Main ----------------------
def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def main():
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_python.BaseOptions.Delegate.CPU,     # avoids the GPU (Metal) crash on Mac
        ),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Cannot open the camera. Try changing CAMERA_INDEX")

    if tg_enabled():
        threading.Thread(target=poll_loop, daemon=True).start()
    else:
        print("BOT_TOKEN/CHAT_ID not set: results are shown on screen only; press 'o' instead of the \"I'm OK\" button")

    motion = MotionEstimator()
    sm = FallStateMachine()
    L_ref = None
    start = time.time()
    last_t = start
    last_ts_ms = -1
    print("Started | q = quit, o = I'm OK (test), c = recalibrate")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = max(now - last_t, 1e-3)
        last_t = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int((now - start) * 1000), last_ts_ms + 1)      # timestamps must be strictly increasing
        last_ts_ms = ts_ms
        result = landmarker.detect_for_video(mp_image, ts_ms)

        f = None
        if result.pose_landmarks:
            f = extract_features(result.pose_landmarks[0], w, h, L_ref)
        note = "" if f else ("NO PERSON" if not result.pose_landmarks else "LOW VISIBILITY")

        v = a = 0.0
        speed = 99.0
        if f:
            if f["theta"] < UPRIGHT_DEG:                    # update L_ref only while standing upright
                L_ref = f["L_cur"] if L_ref is None else 0.95 * L_ref + 0.05 * f["L_cur"]
            v, a, speed = motion.update(now, f["hy"], f["hx"])

        # "I'm OK" button (Telegram or the 'o' key)
        if ok_event.is_set():
            ok_event.clear()
            if sm.cancel():
                print("User pressed I'm OK -> alert cancelled")
                send_text("Acknowledged. Alert cancelled.")

        event = sm.update(f, v, speed, now, dt)
        if event == "ALERT":
            print("ALERT: fast drop followed by lying still")
            send_text(f"Alert: a possible fall was detected and the person has been lying still for more than {STILL_SECONDS:.0f} seconds. "
                      "Press the button below if they are OK.", with_button=True)
        elif event == "RECOVERED":
            print("User is back in a normal posture (standing/sitting)")
            send_text("The person is back in a normal posture.")

        # ------- Draw overlay -------
        if f:
            cv2.line(frame, tuple(f["h_px"].astype(int)), tuple(f["s_px"].astype(int)), (0, 255, 0), 3)
            for hip_p, knee_p, ank_p in f["leg_lines"]:
                cv2.line(frame, tuple(hip_p.astype(int)), tuple(knee_p.astype(int)), (255, 200, 0), 2)
                cv2.line(frame, tuple(knee_p.astype(int)), tuple(ank_p.astype(int)), (255, 200, 0), 2)
            cv2.circle(frame, tuple(f["s_px"].astype(int)), 6, (255, 0, 0), -1)
            cv2.circle(frame, tuple(f["h_px"].astype(int)), 6, (0, 0, 255), -1)

        color = STATE_COLOR[sm.state]
        title = sm.state + (f"  ({note})" if note else "") + ("  [disarmed]" if sm.disarmed else "")
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if f:
            lines = [
                f"angle {f['theta']:5.1f}  hip_h {f['hy']:4.2f}  r {fmt(f['r'], '4.2f')}",
                f"v {v:+5.2f}  a {a:+6.1f}  still {sm.t_still:3.1f}s",
                f"thigh {fmt(f['thigh'], '5.1f')}  knee {fmt(f['knee'], '5.1f')}  lying {int(f['lying'])}",
            ]
            for i, t in enumerate(lines):
                cv2.putText(frame, t, (10, 60 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Fall Detection - Phase 2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            ok_event.set()
        elif key == ord("c"):
            L_ref = None
            print("Torso length recalibrated")

    stop_event.set()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()