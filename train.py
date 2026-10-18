"""
Train and compare three ways of deciding "a fall is starting":
    A. Rule-based        (hip drop speed v <= -v0)
    B. kNN               (NumPy, standardised features, Euclidean distance, k = 5)
    C. Logistic Regression (NumPy, gradient descent, class-weighted)

Design (same idea as section 9 of the project document):
  - The classifier decides the FALL ONSET from single-frame features
    x = [theta, hy, v, a, r, t_still]   (positive = frames with label 1, negative = label 0).
  - Frames with label 2 (lying still after the fall) are NOT used at frame level, because one frame of
    "lying still after a fall" looks exactly like lying on a sofa. That part is decided by the state machine
    (lying posture + still for T seconds), exactly like the real-time program.
  - End-to-end: each classifier replaces the "v <= -V_FALL" trigger inside the Phase 2 state machine,
    and we count alerts.

Evaluation: leave-one-person-out cross-validation (train on the other persons, test on the held-out person).

Usage (run from the project folder, next to build_features.py and your Phase 2 file):
    python build_features.py          # makes features.csv first
    python train_compare.py
    python train_compare.py --module fall_detect --k 5 --knn-threshold 0.4
"""
import argparse
import csv
import math
import sys

import numpy as np

from build_features import load_feature_module

FEATS = ["theta", "hy", "v", "a", "r", "t_still"]
V_IDX = FEATS.index("v")
STILL_IDX = FEATS.index("t_still")


# ---------------------------------------------------------------- data ----------------------
def load_table(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def col(name):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])

    return dict(
        file=np.array([r["file"] for r in rows]),
        person=np.array([r["person"] for r in rows]),
        activity=np.array([r["activity"] for r in rows]),
        t=col("t"),
        label=col("label").astype(int),
        valid=col("valid").astype(int),
        lying=col("lying"),
        X=np.column_stack([col(n) for n in FEATS]),
    )


def prf(y, p):
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


# ---------------------------------------------------------------- models --------------------
class Scaler:
    """Fill missing values with the train median, then z = (x - mu) / sigma."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu = Xi.mean(0)
        self.sd = Xi.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def transform(self, X):
        return (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd


class RuleTrigger:
    """Trigger when the hip drops faster than v0 (v0 is tuned on the training data by F1)."""

    def __init__(self, v0=None):
        self.v0 = v0
        self.name = "Rule-based (v0 tuned)" if v0 is None else f"Rule-based (V_FALL={v0})"

    def fit(self, X, y):
        if self.v0 is not None:
            return self
        best = (-1.0, 1.5)
        for v0 in np.arange(0.5, 6.01, 0.25):
            f1 = prf(y, X[:, V_IDX] <= -v0)[2]
            if f1 > best[0]:
                best = (f1, float(v0))
        self.v0 = best[1]
        return self

    def predict(self, X):
        return X[:, V_IDX] <= -self.v0


class KNN:
    name = "kNN"

    def __init__(self, k=5, thr=0.4):
        self.k, self.thr = k, thr
        self.name = f"kNN (k={k}, vote>={thr})"

    def fit(self, X, y):
        self.sc = Scaler().fit(X)
        self.Z = self.sc.transform(X)
        self.y = np.asarray(y, dtype=float)
        self.zz = (self.Z ** 2).sum(1)
        return self

    def predict(self, X):
        Z = self.sc.transform(X)
        out = np.zeros(len(Z), dtype=bool)
        k = min(self.k, len(self.Z) - 1)
        for s in range(0, len(Z), 500):
            q = Z[s:s + 500]
            d = (q ** 2).sum(1)[:, None] + self.zz[None, :] - 2 * q @ self.Z.T   # squared Euclidean distance
            idx = np.argpartition(d, k, axis=1)[:, :k]
            out[s:s + 500] = self.y[idx].mean(1) >= self.thr
        return out


class LogReg:
    def __init__(self, thr=0.5, lr=0.1, iters=1500, l2=1e-3):
        self.thr, self.lr, self.iters, self.l2 = thr, lr, iters, l2
        self.name = f"Logistic Regression (p>={thr})"

    def fit(self, X, y):
        self.sc = Scaler().fit(X)
        Z = self.sc.transform(X)
        y = np.asarray(y, dtype=float)
        npos = y.sum()
        sw = np.where(y == 1, (len(y) - npos) / max(npos, 1.0), 1.0)    # balance the rare positive class
        sw = sw / sw.mean()
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w + self.b, -30, 30)))
            g = (p - y) * sw
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w)     # gradient descent step
            self.b -= self.lr * g.mean()
        return self

    def predict(self, X):
        z = self.sc.transform(X) @ self.w + self.b
        return z >= math.log(self.thr / (1 - self.thr))


# ---------------------------------------------------------------- end-to-end ----------------
def run_pipeline(T, idxs, trigger, F):
    """Run the Phase 2 state machine over one recording; the classifier supplies the 'fast drop' trigger."""
    sm = F.FallStateMachine()
    alerts = []
    prev_t = None
    for k, i in enumerate(idxs):
        t = T["t"][i]
        dt = (t - prev_t) if prev_t is not None else 1 / 30
        prev_t = t
        if T["valid"][i] != 1:
            f, v, speed = None, 0.0, 99.0
        else:
            f = {"lying": bool(T["lying"][i] == 1)}
            v = -1e3 if trigger[k] else 0.0                      # forces the NORMAL -> SUSPECT transition
            speed = 0.0 if T["X"][i, STILL_IDX] > 0 else 99.0    # still / moving
        if sm.update(f, v, speed, t, dt) == "ALERT":
            alerts.append(t)
    return alerts


def diagnose_miss(T, info, trig, F):
    """Explain why a fall recording produced no alert after the fall started."""
    idx, onset = info["idx"], info["onset"]
    t = T["t"][idx]
    post = t >= onset
    valid = T["valid"][idx] == 1
    n_post = max(int(post.sum()), 1)
    valid_frac = float((post & valid).sum()) / n_post
    fired = bool((trig & (t >= onset - 1.0) & (t <= onset + F.SUSPECT_TIMEOUT)).any()) \
        if hasattr(F, "SUSPECT_TIMEOUT") else bool(trig[post].any())
    ok = valid & (T["lying"][idx] == 1) & (T["X"][idx, STILL_IDX] > 0) & post
    longest = run = 0.0
    prev = None
    for k in range(len(idx)):
        if ok[k]:
            run += (t[k] - prev) if prev is not None else 0.0
            longest = max(longest, run)
        else:
            run = 0.0
        prev = t[k]
    lying_any = bool((valid & (T["lying"][idx] == 1) & post).any())
    if valid_frac < 0.6:
        why = f"person detected in only {100 * valid_frac:.0f}% of frames after the fall"
    elif not fired:
        why = "no fast-drop trigger near the fall onset (v never reached the threshold)"
    elif not lying_any:
        why = "never classified as lying posture after the fall"
    elif longest < F.STILL_SECONDS:
        why = f"lying+still only {longest:.1f}s (< T = {F.STILL_SECONDS:.0f}s)"
    else:
        why = "trigger and posture fine, check the state machine timing (SUSPECT timeout)"
    return why


def event_eval(T, files, triggers, F):
    tp = n_fall = fp_pre = fp_extra = fp_normal = 0
    delays, normal_sec = [], 0.0
    missed, false_files = [], []
    for name, info in files.items():
        alerts = run_pipeline(T, info["idx"], triggers[name], F)
        if info["activity"] == "fall":
            if info["onset"] is None:
                continue
            n_fall += 1
            pre = [a for a in alerts if a < info["onset"]]
            post = [a for a in alerts if a >= info["onset"]]
            fp_pre += len(pre)
            if post:
                tp += 1
                delays.append(post[0] - info["onset"])
                fp_extra += len(post) - 1
            else:
                missed.append((name, info["person"], diagnose_miss(T, info, triggers[name], F)))
            if pre or len(post) > 1:
                false_files.append((name, f"fall file: {len(pre)} alert(s) before the fall, {max(len(post) - 1, 0)} extra after"))
        else:
            fp_normal += len(alerts)
            normal_sec += info["duration"]
            if alerts:
                false_files.append((name, f"{info['activity']}: {len(alerts)} alert(s) at t = "
                                          + ", ".join(f"{a:.1f}s" for a in alerts[:5])))
    fp_total = fp_normal + fp_pre + fp_extra
    return dict(tp=tp, n_fall=n_fall, fp_normal=fp_normal, fp_other=fp_pre + fp_extra,
                per_hour=fp_normal / (normal_sec / 3600) if normal_sec > 0 else float("nan"),
                recall=tp / n_fall if n_fall else 0.0,
                precision=tp / (tp + fp_total) if tp + fp_total else 0.0,
                delay=float(np.mean(delays)) if delays else float("nan"),
                missed=missed, false_files=false_files)


# ---------------------------------------------------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--module", default="fall_detect", help="Phase 2 file name without .py")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--knn-threshold", type=float, default=0.4, help="fraction of positive neighbours needed")
    ap.add_argument("--lr-threshold", type=float, default=0.5, help="probability threshold for Logistic Regression")
    ap.add_argument("--out", default="results.txt")
    args = ap.parse_args()

    F = load_feature_module(args.module)
    for attr in ("FallStateMachine", "V_FALL", "STILL_SECONDS"):
        if not hasattr(F, attr):
            raise SystemExit(f"'{args.module}' has no '{attr}': use the Phase 2 file")
    T = load_table(args.features)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    persons = sorted(set(T["person"]))
    if len(persons) < 2:
        raise SystemExit("Need at least 2 persons for a person-wise split")
    if not (T["label"] == 1).any():
        raise SystemExit("No frames with label 1 in features.csv. Label the fall files first "
                         "(record with keys 1/2, or run relabel_falls.py) and run build_features.py again.")

    # per-file info
    files = {}
    for name in dict.fromkeys(T["file"]):
        idx = np.where(T["file"] == name)[0]
        lab = T["label"][idx]
        onset_rows = np.where(lab >= 1)[0]
        files[name] = dict(idx=idx, person=T["person"][idx[0]], activity=T["activity"][idx[0]],
                           duration=float(T["t"][idx[-1]] - T["t"][idx[0]]),
                           onset=float(T["t"][idx[onset_rows[0]]]) if len(onset_rows) else None)
        if files[name]["activity"] == "fall" and files[name]["onset"] is None:
            out(f"WARNING: {name} is a fall file without label 1/2, it is ignored")
        if files[name]["activity"] == "fall" and files[name]["onset"] is not None:
            l2 = np.where(lab == 2)[0]
            lying_sec = T["t"][idx[-1]] - T["t"][idx[l2[0]]] if len(l2) else 0.0
            if lying_sec < F.STILL_SECONDS:
                out(f"NOTE: {name}: lying part is only {lying_sec:.1f}s (< T = {F.STILL_SECONDS:.0f}s), "
                    "the state machine cannot raise an alert in this recording")

    frame_ok = (T["valid"] == 1) & np.isin(T["label"], (0, 1))
    y_all = (T["label"] == 1)
    models_spec = [
        ("rule_default", lambda: RuleTrigger(v0=F.V_FALL)),
        ("rule_tuned", lambda: RuleTrigger()),
        ("knn", lambda: KNN(args.k, args.knn_threshold)),
        ("logreg", lambda: LogReg(args.lr_threshold)),
    ]
    names, frame_true, frame_pred, triggers = {}, [], defaultdict_list(models_spec), {}
    for key, make in models_spec:
        triggers[key] = {}
        names[key] = make().name

    for p in persons:
        tr = frame_ok & (T["person"] != p)
        te_rows = np.where(T["person"] == p)[0]
        if y_all[tr].sum() == 0:
            out(f"fold {p}: no positive training frames, skipped")
            continue
        out(f"fold: test on {p} | train frames {int(tr.sum())} (positives {int(y_all[tr].sum())})")
        Xtr, ytr = T["X"][tr], y_all[tr]
        te_eval = np.intersect1d(te_rows, np.where(frame_ok)[0])
        frame_true.append(y_all[te_eval])
        for key, make in models_spec:
            m = make().fit(Xtr, ytr)
            names[key] = m.name
            if key == "rule_tuned":
                out(f"   tuned v0 = {m.v0:.2f}")
            # frame-level predictions
            frame_pred[key].append(m.predict(T["X"][te_eval]))
            # per-frame triggers for the end-to-end run (only valid rows can trigger)
            for name, info in files.items():
                if info["person"] != p:
                    continue
                idx = info["idx"]
                trig = np.zeros(len(idx), dtype=bool)
                ok = T["valid"][idx] == 1
                if ok.any():
                    trig[ok] = m.predict(T["X"][idx[ok]])
                triggers[key][name] = trig

    yt = np.concatenate(frame_true)
    out("\n=== 1) Frame level: is this frame part of the falling moment? (label 1 vs label 0, label 2 excluded) ===")
    out(f"{'method':<34}{'precision':>10}{'recall':>9}{'F1':>7}{'TP':>7}{'FP':>7}{'FN':>7}")
    for key, _ in models_spec:
        pr = np.concatenate(frame_pred[key])
        pp, rr, ff, tp, fp, fn = prf(yt, pr)
        out(f"{names[key]:<34}{pp:>10.3f}{rr:>9.3f}{ff:>7.3f}{tp:>7d}{fp:>7d}{fn:>7d}")

    out("\n=== 2) End to end: trigger + Phase 2 state machine (lying posture, still for T) -> alert ===")
    out(f"T = {F.STILL_SECONDS:.0f}s. A fall recording counts as detected if an alert appears after the fall started.")
    out(f"{'method':<34}{'detected':>10}{'recall':>8}{'prec.':>7}{'false alarms':>14}{'per hour':>10}{'delay(s)':>10}")
    results = {}
    for key, _ in models_spec:
        r = event_eval(T, files, triggers[key], F)
        results[key] = r
        fa = f"{r['fp_normal']}+{r['fp_other']}"
        out(f"{names[key]:<34}{str(r['tp']) + '/' + str(r['n_fall']):>10}{r['recall']:>8.2f}{r['precision']:>7.2f}"
            f"{fa:>14}{r['per_hour']:>10.1f}{r['delay']:>10.1f}")
    out("false alarms = alerts in non-fall recordings + (alerts in fall recordings before the fall / extra alerts)")

    out("\n=== 3) Which recordings went wrong? ===")
    base_missed = {m[0] for m in results["rule_default"]["missed"]}
    out(f"Missed falls, {names['rule_default']}:")
    for name, person, why in results["rule_default"]["missed"] or [("-", "", "none")]:
        out(f"   {name} ({person}): {why}")
    for key, _ in models_spec[1:]:
        ms = {m[0] for m in results[key]["missed"]}
        if ms != base_missed:
            out(f"Missed falls differ for {names[key]}:")
            for name, person, why in results[key]["missed"]:
                out(f"   {name} ({person}): {why}")
    for key, _ in models_spec:
        ff = results[key]["false_files"]
        if ff:
            out(f"False alarms, {names[key]}:")
            for name, what in ff:
                out(f"   {name}: {what}")

    # Interpretable weights of Logistic Regression trained on all data
    tr_all = frame_ok
    if y_all[tr_all].sum() > 0:
        lr_all = LogReg(args.lr_threshold).fit(T["X"][tr_all], y_all[tr_all])
        out("\nLogistic Regression weights (standardised features, trained on all data):")
        for n, w in sorted(zip(FEATS, lr_all.w), key=lambda z: -abs(z[1])):
            out(f"   {n:<8}{w:+.3f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved {args.out}")


def defaultdict_list(spec):
    return {key: [] for key, _ in spec}


if __name__ == "__main__":
    main()