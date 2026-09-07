import streamlit as st
import cv2
import base64
import json
import tempfile
import os
import pandas as pd
import requests
import numpy as np
from datetime import datetime

st.set_page_config(page_title="VMS Inspection Dashboard", layout="wide", page_icon="📦")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #f0f0f0; }
.header-box {
    background: #1a1d27; padding: 16px 20px; border-radius: 8px;
    margin-bottom: 20px; border-left: 4px solid #3b82f6;
}
.verdict-accepted { background:#1a4731; color:#4ade80; padding:4px 14px; border-radius:5px; font-weight:700; font-size:13px; }
.verdict-review   { background:#3d2c00; color:#fbbf24; padding:4px 14px; border-radius:5px; font-weight:700; font-size:13px; }
.verdict-rejected { background:#3d0a0a; color:#f87171; padding:4px 14px; border-radius:5px; font-weight:700; font-size:13px; }
.big-score { font-size:38px; font-weight:700; }
.step-card {
    background:#12151f; border:0.5px solid #2a2d3a;
    border-radius:8px; padding:12px 14px; margin-bottom:10px;
}
.step-title { font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:6px; }
.step-body  { font-size:13px; color:#94a3b8; line-height:1.6; }
.bar-bg   { background:#2a2d3a; border-radius:4px; height:7px; overflow:hidden; margin:6px 0 10px; }
.bar-fill { height:7px; border-radius:4px; }
.barcode-pill {
    display:inline-block; background:#1e2d40; border:1px solid #2a4a6b;
    border-radius:6px; padding:4px 12px; font-size:13px;
    color:#60a5fa; font-family:monospace; margin-top:6px;
}
.log-box {
    background:#0a0d14; border:1px solid #2a2d3a; border-radius:8px;
    padding:12px; font-family:monospace; font-size:12px;
    color:#4ade80; max-height:300px; overflow-y:auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h3 style="margin:0;color:#fff;font-weight:600;">📦 VMS Inspection Dashboard</h3>
</div>
""", unsafe_allow_html=True)

# ── Auth ──────────────────────────────────────────────────────────────────────
hf_token = st.secrets.get("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN missing — add it in Streamlit Cloud → Settings → Secrets")
    st.stop()

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "google/gemma-4-31B-it:together"
HEADERS    = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    video_type = st.selectbox("Process Stream Type", ["Return", "Forward"])
    uploaded_files = st.file_uploader(
        "Upload Inspection Videos",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )
    st.divider()
    num_frames = st.slider("Frames per step", 2, 5, 3,
                           help="More frames per SOP step = more accurate")
    show_logs = st.toggle("Show debug logs", value=True)
    st.divider()
    st.markdown("**💡 Slow upload?**")
    st.caption("• Phone: use **Video Compress** app")
    st.caption("• PC: use **HandBrake** — RF 28")
    st.caption("• Record at **720p** not 1080p")

# ── Logging ───────────────────────────────────────────────────────────────────
log_lines = []

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {level}: {msg}"
    log_lines.append(line)
    print(line)

def show_log_box(placeholder):
    if show_logs:
        placeholder.markdown(
            "<div class='log-box'>" + "<br>".join(log_lines[-40:]) + "</div>",
            unsafe_allow_html=True
        )

# ── Improvement 1: Motion-based key frame detection ───────────────────────────
def detect_still_moments(video_path, duration, n_segments=4):
    """
    Find frames where operator holds something still (inspection moments).
    Returns timestamps for each SOP segment.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    segment_duration = duration / n_segments
    best_frames = {}

    for seg in range(n_segments):
        seg_start = seg * segment_duration
        seg_end = (seg + 1) * segment_duration
        # Sample frames in this segment
        times = [seg_start + (seg_end - seg_start) * i / 5 for i in range(5)]
        min_motion = float('inf')
        best_time = times[len(times) // 2]
        prev_gray = None

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(np.mean(diff))
                if motion < min_motion:
                    min_motion = motion
                    best_time = t
            prev_gray = gray

        best_frames[seg] = best_time
        log(f"Segment {seg+1} best still moment at {best_time:.1f}s (motion={min_motion:.2f})")

    cap.release()
    return best_frames

# ── Improvement 2: Smart frame extraction ─────────────────────────────────────
def extract_smart_frames(video_path, n_per_step=3):
    """
    Extract high-res frames at key moments for each SOP step.
    Returns dict: {step_name: [b64_frames]}
    """
    log(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total / fps
    log(f"Video: {total} frames, {fps:.1f} FPS, {duration:.1f}s")

    if total <= 0:
        log("ERROR: Cannot read video", "ERROR")
        cap.release()
        return {}

    # SOP steps happen at different parts of the video
    # Label is shown FIRST, unboxing NEXT, tags AFTER, product LAST
    step_segments = {
        "label":   (0.0,  0.25),   # first 25% — label shown before opening
        "unboxing":(0.20, 0.55),   # 20-55% — opening package
        "tags":    (0.45, 0.80),   # 45-80% — tags shown after opening
        "product": (0.65, 1.00),   # last 35% — product displayed
    }

    # Find still moments for each segment
    still_moments = detect_still_moments(video_path, duration, n_segments=4)

    step_frames = {}
    for step, (start_pct, end_pct) in step_segments.items():
        start_t = duration * start_pct
        end_t   = duration * end_pct
        times   = [start_t + (end_t - start_t) * i / (n_per_step - 1)
                   for i in range(n_per_step)] if n_per_step > 1 else [start_t]

        frames = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                # Improvement 4: Higher resolution — 960x720
                h, w = frame.shape[:2]
                target_w = 960
                target_h = int(h * target_w / w)
                frame = cv2.resize(frame, (target_w, target_h))
                _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                b64 = base64.b64encode(buf).decode('utf-8')
                frames.append(b64)
                log(f"  {step} frame at {t:.1f}s ({len(b64)//1024}KB)")
            else:
                log(f"  {step} frame failed at {t:.1f}s", "WARN")

        step_frames[step] = frames

    cap.release()
    log(f"Extracted frames — " + ", ".join(f"{k}:{len(v)}" for k,v in step_frames.items()))
    return step_frames

# ── Improvement 3: Targeted prompts per SOP step ──────────────────────────────
STEP_PROMPTS = {
    "label": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze the shipping label and barcode visibility.

Look carefully for:
- Outer polybag or box with a shipping/AWB label
- Ekart, Delhivery, Bluedart, or other carrier labels
- Any barcode, QR code, or tracking number
- Whether the label is held clearly toward the camera
- Whether text is readable and not blurred

READ AND REPORT:
1. Exact tracking/AWB number if visible (e.g. MYER1069106708)
2. Carrier name if visible
3. Whether label is clear and unobstructed

SCORING CRITERIA (30 points):
- Label clearly visible and held toward camera: GOOD
- Barcode/tracking number readable: GOOD  
- Label blurred, obscured, or not shown: BAD
- No label visible at all: FAIL

Give your assessment in 3-4 specific sentences. Start with what you actually see.""",

    "unboxing": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze the packaging integrity and unboxing procedure.

Look carefully for:
- Is the outer polybag or box seal being inspected before opening?
- Is the operator cutting or unsealing the package on camera?
- Is the entire unboxing happening within the camera frame?
- Any signs of pre-cutting, tampering, or damage to the seal?
- Is the operator rotating the package to show all sides?

SCORING CRITERIA (20 points):
- Package opened fully on camera: GOOD
- Seal inspected before opening: GOOD
- Package rotated to show all sides: GOOD
- Pre-cut or tampered seal: BAD
- Unboxing happening off-camera: FAIL

Give your assessment in 3-4 specific sentences. Start with what you actually see.""",

    "tags": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze brand tag and price tag verification.

Look carefully for:
- Brand fabric labels (neck label, waistband label)
- Paper hangtags with price, size, or brand name
- Plastic fasteners/tags still attached
- Tags being held close to the camera lens
- Whether tag text (brand, price, size) is readable

SCORING CRITERIA (25 points):
- Brand tag clearly shown close to camera: GOOD
- Price/size tag visible and readable: GOOD
- Tag text legible: GOOD
- Tags not shown or too far from camera: BAD
- Tags missing or removed: FAIL

READ AND REPORT any brand name, price, or size you can see.
Give your assessment in 3-4 specific sentences. Start with what you actually see.""",

    "product": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze product display and condition assessment.

Look carefully for:
- Is the item fully unfolded and displayed?
- What type of item is it (shirt, pants, dress, etc)?
- Is the front side shown clearly?
- Is the back side also shown?
- Any visible defects, stains, tears, or damage?
- Signs of wear or incorrect item return?

SCORING CRITERIA (25 points):
- Item fully unfolded: GOOD
- Both front and back shown: GOOD
- Condition clearly visible: GOOD
- Item only partially shown: BAD
- Item kept folded or off-camera: FAIL

Give your assessment in 3-4 specific sentences describing exactly what you see."""
}

# ── API call ──────────────────────────────────────────────────────────────────
def query_vlm_multi(frames_b64, prompt, step_name):
    """Send multiple frames for one step — model sees all frames together."""
    log(f"Calling API for: {step_name} ({len(frames_b64)} frames)")
    log(f"Model: {HF_MODEL}")

    try:
        # Build content with all frames + prompt
        content = []
        for i, b64 in enumerate(frames_b64):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": HF_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 400,
            "temperature": 0.1
        }

        log(f"Sending request ({len(frames_b64)} images)...")
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=120)
        log(f"Response status: {response.status_code}")

        if response.status_code == 503:
            import time
            log("Model loading — waiting 20s...", "WARN")
            time.sleep(20)
            response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=120)
            log(f"Retry status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            log(f"Raw response: {str(result)[:300]}")
            if isinstance(result, dict) and "choices" in result:
                text = result["choices"][0]["message"]["content"]
                log(f"Answer: {text[:150]}")
                return text
            return str(result)
        else:
            error = response.text[:300]
            log(f"API ERROR {response.status_code}: {error}", "ERROR")
            return f"API error {response.status_code}: {error}"

    except requests.exceptions.Timeout:
        log("Timeout after 120s", "ERROR")
        return "Error: Request timed out"
    except Exception as e:
        log(f"Exception: {str(e)}", "ERROR")
        return f"Error: {str(e)}"

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_response(text, weight):
    """Score based on positive/negative language in AI response."""
    text_lower = text.lower()
    
    positive = ["visible", "can see", "shows", "displays", "readable", "present",
                "detected", "found", "held", "opened", "unfolded", "clearly",
                "label", "barcode", "tag", "brand", "tracking", "good", "compliant"]
    negative = ["not visible", "cannot see", "no label", "not present", "unclear",
                "cannot read", "not shown", "error", "missing", "absent", "fail",
                "no tag", "not found", "obscured", "blurred", "off camera"]

    pos = sum(2 if phrase in text_lower else 0 for phrase in positive)
    neg = sum(2 if phrase in text_lower else 0 for phrase in negative)

    if "error" in text_lower and "api" in text_lower:
        return int(weight * 0.20)
    if pos > neg * 1.5:  return int(weight * 0.90)
    if pos > neg:        return int(weight * 0.65)
    if pos == neg:       return int(weight * 0.50)
    return int(weight * 0.20)

# ── Main audit ────────────────────────────────────────────────────────────────
def run_audit(video_bytes, stream_type, n_per_step, log_placeholder):
    log(f"Starting audit — {stream_type} — {len(video_bytes)/1024/1024:.1f}MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    log(f"Saved to: {tmp_path}")
    show_log_box(log_placeholder)

    try:
        # Extract smart frames
        step_frames = extract_smart_frames(tmp_path, n_per_step=n_per_step)
        os.remove(tmp_path)
        show_log_box(log_placeholder)

        if not step_frames:
            return {"score": 0, "verdict": "REJECTED", "tracking_number": "None",
                    "label_check": "Could not read video.",
                    "unboxing_check": "N/A", "brand_tag_check": "N/A", "product_check": "N/A"}

        # Analyze each step with targeted prompt + multiple frames
        results = {}
        step_map = [
            ("label",    "label_check"),
            ("unboxing", "unboxing_check"),
            ("tags",     "brand_tag_check"),
            ("product",  "product_check"),
        ]

        for step_key, result_key in step_map:
            log(f"--- Analyzing: {step_key} ---")
            frames = step_frames.get(step_key, [])
            if frames:
                text = query_vlm_multi(frames, STEP_PROMPTS[step_key], step_key)
            else:
                text = "No frames available for this step."
            results[result_key] = text
            show_log_box(log_placeholder)

        # Extract tracking number from label result
        import re
        tracking = "Not visible"
        label_text = results.get("label_check", "")
        # Look for alphanumeric strings 8+ chars (tracking numbers)
        candidates = re.findall(r'\b[A-Z0-9]{8,}\b', label_text.upper())
        # Filter out common false positives
        false_positives = {"TRACKING", "SHIPPING", "BARCODE", "VISIBLE", "CAMERA",
                          "LABEL", "CARRIER", "EKART", "CLEARLY", "SHOWING"}
        real_numbers = [c for c in candidates if c not in false_positives]
        if real_numbers:
            tracking = real_numbers[0]
            log(f"Tracking number: {tracking}")

        # Score each step
        s1 = score_response(results["label_check"],     30)
        s2 = score_response(results["unboxing_check"],  20)
        s3 = score_response(results["brand_tag_check"], 25)
        s4 = score_response(results["product_check"],   25)
        total = s1 + s2 + s3 + s4

        log(f"Scores — Label:{s1}/30 Unboxing:{s2}/20 Tags:{s3}/25 Product:{s4}/25 = {total}/100")
        verdict = "ACCEPTED" if total >= 85 else "REVIEW REQUIRED" if total >= 50 else "REJECTED"
        log(f"Verdict: {verdict}")

        return {
            "score": total, "verdict": verdict, "tracking_number": tracking,
            **results
        }

    except Exception as e:
        log(f"FATAL: {str(e)}", "ERROR")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"score": 0, "verdict": "ERROR", "tracking_number": "Error",
                "label_check": f"Error: {e}",
                "unboxing_check": "N/A", "brand_tag_check": "N/A", "product_check": "N/A"}

def score_color(s):
    return "#22c55e" if s >= 85 else "#f59e0b" if s >= 50 else "#ef4444"

def verdict_badge(s):
    if s >= 85: return '<span class="verdict-accepted">✅ ACCEPTED</span>'
    if s >= 50: return '<span class="verdict-review">⚠️ REVIEW REQUIRED</span>'
    return '<span class="verdict-rejected">❌ REJECTED</span>'

STEP_WEIGHTS = [
    ("Shipping Label & Barcode",    "label_check",     30),
    ("Packaging & Unboxing",        "unboxing_check",  20),
    ("Brand Tag & Price Tag",       "brand_tag_check", 25),
    ("Product Display & Condition", "product_check",   25),
]

# ── UI ────────────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.info("Upload one or more inspection videos from the sidebar to begin.")
    st.stop()

st.subheader(f"Batch Inspection — {len(uploaded_files)} video(s) · Stream: {video_type}")
csv_rows = []

for file in uploaded_files:
    st.divider()
    log_lines.clear()

    vid_col, result_col = st.columns([1, 1], gap="large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_bytes = file.read()
        tmp.write(video_bytes)
        play_path = tmp.name

    with vid_col:
        st.markdown(f"**{file.name}**")
        st.video(play_path)
        if show_logs:
            st.markdown("**Debug Log**")
        log_placeholder = st.empty()

    with result_col:
        with st.spinner(f"Analyzing {file.name}..."):
            log(f"Processing: {file.name} ({len(video_bytes)/1024/1024:.1f} MB)")
            show_log_box(log_placeholder)
            audit = run_audit(video_bytes, video_type, num_frames, log_placeholder)

        os.remove(play_path)
        show_log_box(log_placeholder)

        score    = audit.get("score", 0)
        verdict  = audit.get("verdict", "REJECTED")
        tracking = audit.get("tracking_number", "Not visible")
        color    = score_color(score)

        st.markdown(
            f'{verdict_badge(score)}&nbsp;&nbsp;'
            f'<span class="big-score" style="color:{color}">{score}</span>'
            f'<span style="font-size:16px;color:#475569">/100</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="bar-bg"><div class="bar-fill" style="width:{score}%;background:{color}"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="margin-bottom:10px">🔍 <b style="color:#94a3b8">Tracking / AWB:</b> '
            f'<span class="barcode-pill">{tracking}</span></div>',
            unsafe_allow_html=True,
        )

        for label, key, weight in STEP_WEIGHTS:
            text = audit.get(key, "N/A")
            st.markdown(f"""
            <div class="step-card">
                <div class="step-title">{label}
                    <span style="float:right;color:#475569;font-weight:400">{weight} pts</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{score}%;background:{color}"></div>
                </div>
                <div class="step-body">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        csv_rows.append({
            "Timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "File":              file.name,
            "Stream Type":       video_type,
            "Verdict":           verdict,
            "Score":             score,
            "Tracking / AWB":    tracking,
            "Label & Barcode":   audit.get("label_check", ""),
            "Packaging":         audit.get("unboxing_check", ""),
            "Brand Tags":        audit.get("brand_tag_check", ""),
            "Product Condition": audit.get("product_check", ""),
        })

if csv_rows:
    st.divider()
    st.subheader("Batch Summary")
    df = pd.DataFrame(csv_rows)
    st.dataframe(
        df[["File", "Verdict", "Score", "Tracking / AWB", "Stream Type", "Timestamp"]],
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Download Full Audit Report (CSV)",
        data=df.to_csv(index=False),
        file_name=f"VMS_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
