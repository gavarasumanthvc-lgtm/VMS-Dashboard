"""
Video frame extraction.

Fix vs. the original implementation: the old code ran a full video decode
pass to find "still moments" (detect_still_moments) and then threw that
result away, doing a *second* full decode pass with plain even-spaced
timestamps. Here, stillness detection and frame extraction happen in the
SAME pass per step: we sample candidate frames once, score them by motion,
and reuse the already-decoded candidates directly as the extracted frames.
"""

import base64
import cv2
import numpy as np

from config import (
    CANDIDATES_PER_STEP,
    FRAME_JPEG_QUALITY,
    FRAME_TARGET_WIDTH,
)


def _encode_frame(frame):
    h, w = frame.shape[:2]
    target_w = FRAME_TARGET_WIDTH
    target_h = int(h * target_w / w)
    resized = cv2.resize(frame, (target_w, target_h))
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


def select_still_frames(cap, start_t, end_t, n_per_step, log=None,
                         n_candidates=CANDIDATES_PER_STEP):
    """
    Sample n_candidates timestamps evenly across [start_t, end_t], decode
    each once, score each candidate by how little the frame changed versus
    its neighbor (a proxy for "operator holding the item still for the
    camera"), then keep the n_per_step stillest candidates — enforcing a
    minimum time gap between selections so we don't return near-duplicates.

    Returns a list of base64-encoded JPEG frames, in chronological order.
    """
    n_candidates = max(n_candidates, n_per_step)
    if n_candidates > 1:
        times = [start_t + (end_t - start_t) * i / (n_candidates - 1) for i in range(n_candidates)]
    else:
        times = [start_t]

    candidates = []  # list of (time, frame_bgr, motion_score)
    prev_gray = None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            if log:
                log(f"  frame read failed at {t:.1f}s", "WARN")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(gray, prev_gray))) if prev_gray is not None else 0.0
        candidates.append((t, frame, motion))
        prev_gray = gray

    if not candidates:
        return []

    # Pick the stillest candidates, enforcing minimum spacing so selections
    # spread across the window instead of clustering on one quiet moment.
    min_gap = (end_t - start_t) / (n_per_step * 2) if n_per_step > 0 else 0
    ranked = sorted(candidates, key=lambda c: c[2])  # lowest motion first

    chosen = []
    for t, frame, motion in ranked:
        if len(chosen) >= n_per_step:
            break
        if any(abs(t - ct) < min_gap for ct, _, _ in chosen):
            continue
        chosen.append((t, frame, motion))

    # Backfill if spacing constraint left us short (e.g. very short window)
    if len(chosen) < n_per_step:
        for t, frame, motion in ranked:
            if len(chosen) >= n_per_step:
                break
            if any(t == ct for ct, _, _ in chosen):
                continue
            chosen.append((t, frame, motion))

    chosen.sort(key=lambda c: c[0])  # chronological order for the model

    frames_b64 = []
    for t, frame, motion in chosen:
        b64 = _encode_frame(frame)
        if b64:
            frames_b64.append(b64)
            if log:
                log(f"  frame at {t:.1f}s (motion={motion:.2f}, {len(b64)//1024}KB)")
    return frames_b64


def extract_step_frames(video_path, step_segments, n_per_step, log=None):
    """
    Open the video once and extract frames for every SOP step in one pass.

    step_segments: dict of {step_name: (start_pct, end_pct)}
    Returns: dict of {step_name: [b64_frame, ...]}
    """
    if log:
        log(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total / fps if fps else 0

    if log:
        log(f"Video: {total} frames, {fps:.1f} FPS, {duration:.1f}s")

    if total <= 0 or duration <= 0:
        if log:
            log("ERROR: Cannot read video", "ERROR")
        cap.release()
        return {}

    step_frames = {}
    for step, (start_pct, end_pct) in step_segments.items():
        start_t = duration * start_pct
        end_t = duration * end_pct
        if log:
            log(f"--- Extracting: {step} ({start_t:.1f}s-{end_t:.1f}s) ---")
        step_frames[step] = select_still_frames(cap, start_t, end_t, n_per_step, log=log)

    cap.release()
    if log:
        log("Extracted frames — " + ", ".join(f"{k}:{len(v)}" for k, v in step_frames.items()))
    return step_frames
