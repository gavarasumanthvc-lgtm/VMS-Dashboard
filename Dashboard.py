import streamlit as st
import cv2
import base64
import json
import tempfile
import os
import pandas as pd
import requests
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
HEADERS = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

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
    num_frames = st.slider("Frames to sample per video", 4, 10, 6,
                           help="More frames = more accurate but slower")
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
    print(line)  # also shows in Streamlit Cloud logs

def show_log_box(placeholder):
    if show_logs:
        placeholder.markdown(
            "<div class='log-box'>" +
            "<br>".join(log_lines[-30:]) +
            "</div>",
            unsafe_allow_html=True
        )

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_frames(video_path, n=6):
    log(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"Video info: {total} frames, {fps:.1f} FPS")

    if total <= 0:
        log("ERROR: Could not read video frames", "ERROR")
        cap.release()
        return []

    interval = max(1, total // n)
    frames_b64 = []
    for i in range(n):
        pos = min(i * interval, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64 = base64.b64encode(buf).decode('utf-8')
            frames_b64.append(b64)
            log(f"Frame {i+1}/{n} extracted at position {pos} ({len(b64)} bytes)")
        else:
            log(f"Frame {i+1}/{n} failed to read", "WARN")

    cap.release()
    log(f"Total frames extracted: {len(frames_b64)}")
    return frames_b64

def query_vlm(image_b64, prompt, step_name):
    log(f"Calling HF API for: {step_name}")
    log(f"Model: {HF_MODEL} via SambaNova")
    log(f"Image size: {len(image_b64)} bytes")

    try:
        payload = {
            "model": HF_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 256,
            "temperature": 0.1
        }

        log(f"Sending POST request to HuggingFace...")
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=90)
        log(f"Response status: {response.status_code}")

        if response.status_code == 503:
            log("Model loading (503) — waiting 20s and retrying...", "WARN")
            import time
            time.sleep(20)
            response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=90)
            log(f"Retry response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            log(f"Raw response type: {type(result).__name__}")
            log(f"Raw response: {str(result)[:200]}")

            # OpenAI-compatible chat completions format
            if isinstance(result, dict) and "choices" in result:
                text = result["choices"][0]["message"]["content"]
                log(f"Extracted text: {text[:100]}")
                return text
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
                if prompt in text:
                    text = text.replace(prompt, "").strip()
                log(f"Extracted text: {text[:100]}")
                return text
            if isinstance(result, dict):
                text = result.get("generated_text", str(result))
                log(f"Dict response text: {text[:100]}")
                return text
            return str(result)

        else:
            error_text = response.text[:300]
            log(f"API ERROR {response.status_code}: {error_text}", "ERROR")
            return f"API error {response.status_code}: {error_text}"

    except requests.exceptions.Timeout:
        log("Request timed out after 90 seconds", "ERROR")
        return "Error: Request timed out"
    except Exception as e:
        log(f"Exception: {str(e)}", "ERROR")
        return f"Error: {str(e)}"

def analyze_frame_for_step(image_b64, step_name, step_prompt):
    full_prompt = f"""You are a warehouse inspection auditor checking SOP compliance.
Look at this video frame and answer specifically about: {step_name}
{step_prompt}
Be specific about what you see. Keep answer to 2-3 sentences."""
    return query_vlm(image_b64, full_prompt, step_name)

def run_audit(video_bytes, stream_type, n_frames, log_placeholder):
    log(f"Starting audit — stream type: {stream_type}")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    log(f"Video saved to temp: {tmp_path} ({len(video_bytes)} bytes)")

    try:
        frames = extract_frames(tmp_path, n=n_frames)
        os.remove(tmp_path)
        show_log_box(log_placeholder)

        if not frames:
            log("No frames extracted — aborting", "ERROR")
            return {
                "score": 0, "verdict": "REJECTED", "tracking_number": "None",
                "label_check": "Could not read video file.",
                "unboxing_check": "No footage.",
                "brand_tag_check": "No tags.",
                "product_check": "Product not visible."
            }

        mid = len(frames) // 2
        frame_early = frames[0]
        frame_mid   = frames[mid]
        frame_late  = frames[-1]
        frame_tag   = frames[min(mid + 1, len(frames) - 1)]

        log("--- Step 1: Shipping Label & Barcode ---")
        label_result = analyze_frame_for_step(
            frame_early,
            "Shipping Label & Barcode Visibility",
            "Is there a shipping label, AWB label, or barcode visible? "
            "Can you read any tracking number, barcode digits, or carrier name? "
            "Is the label clear and unobstructed?"
        )
        show_log_box(log_placeholder)

        log("--- Step 2: Packaging & Unboxing ---")
        unboxing_result = analyze_frame_for_step(
            frame_mid,
            "Packaging Integrity & Unboxing",
            "Is the operator opening a package or polybag on camera? "
            "Does the package appear sealed or tampered? "
            "Is the unboxing happening fully within the camera frame?"
        )
        show_log_box(log_placeholder)

        log("--- Step 3: Brand Tag & Price Tag ---")
        tag_result = analyze_frame_for_step(
            frame_tag,
            "Brand Tag & Price Tag Verification",
            "Are there any brand labels, price tags, hangtags, or size tags visible? "
            "What brand name, price, or size information can you read? "
            "Are the tags held close to the camera?"
        )
        show_log_box(log_placeholder)

        log("--- Step 4: Product Display & Condition ---")
        product_result = analyze_frame_for_step(
            frame_late,
            "Product Display & Condition",
            "Is a product/item fully unfolded and displayed? "
            "What type of item is it and what is its condition? "
            "Are both front and back sides being shown?"
        )
        show_log_box(log_placeholder)

        # Extract tracking number
        import re
        tracking = "Not visible"
        numbers = re.findall(r'[A-Z0-9]{8,}', label_result.upper())
        if numbers:
            tracking = numbers[0]
            log(f"Tracking number found: {tracking}")
        else:
            log("No tracking number found in label result")

        # Scoring
        def score_response(text, weight):
            text_lower = text.lower()
            negative = ["not visible", "cannot see", "no label", "not present",
                       "unclear", "cannot read", "not shown", "error", "n/a"]
            positive = ["visible", "can see", "shows", "displays", "readable",
                       "present", "detected", "found", "held", "opened"]
            neg = sum(1 for w in negative if w in text_lower)
            pos = sum(1 for w in positive if w in text_lower)
            if pos > neg:   return int(weight * 0.85)
            elif pos == neg: return int(weight * 0.55)
            else:            return int(weight * 0.20)

        s1 = score_response(label_result, 30)
        s2 = score_response(unboxing_result, 20)
        s3 = score_response(tag_result, 25)
        s4 = score_response(product_result, 25)
        total = s1 + s2 + s3 + s4

        log(f"Scores — S1:{s1} S2:{s2} S3:{s3} S4:{s4} Total:{total}")
        verdict = "ACCEPTED" if total >= 85 else "REVIEW REQUIRED" if total >= 50 else "REJECTED"
        log(f"Verdict: {verdict}")

        return {
            "score": total, "verdict": verdict, "tracking_number": tracking,
            "label_check": label_result, "unboxing_check": unboxing_result,
            "brand_tag_check": tag_result, "product_check": product_result,
        }

    except Exception as e:
        log(f"FATAL ERROR: {str(e)}", "ERROR")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {
            "score": 0, "verdict": "ERROR", "tracking_number": "Error",
            "label_check": f"Error: {e}",
            "unboxing_check": "N/A", "brand_tag_check": "N/A", "product_check": "N/A"
        }

def score_color(score):
    if score >= 85: return "#22c55e"
    if score >= 50: return "#f59e0b"
    return "#ef4444"

def verdict_badge(score):
    if score >= 85: return '<span class="verdict-accepted">✅ ACCEPTED</span>'
    if score >= 50: return '<span class="verdict-review">⚠️ REVIEW REQUIRED</span>'
    return '<span class="verdict-rejected">❌ REJECTED</span>'

STEP_WEIGHTS = [
    ("Shipping Label & Barcode",    "label_check",     30),
    ("Packaging & Unboxing",        "unboxing_check",  20),
    ("Brand Tag & Price Tag",       "brand_tag_check", 25),
    ("Product Display & Condition", "product_check",   25),
]

# ── Main ──────────────────────────────────────────────────────────────────────
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
        else:
            log_placeholder = st.empty()

    with result_col:
        with st.spinner(f"Analyzing {file.name}..."):
            log(f"Processing file: {file.name} ({len(video_bytes)/1024/1024:.1f} MB)")
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
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download Full Audit Report (CSV)",
        data=df.to_csv(index=False),
        file_name=f"VMS_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
