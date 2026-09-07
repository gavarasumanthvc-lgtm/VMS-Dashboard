import streamlit as st
import cv2
import base64
import json
import tempfile
import os
import pandas as pd
from datetime import datetime
from groq import Groq

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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h3 style="margin:0;color:#fff;font-weight:600;">📦 VMS Inspection Dashboard</h3>
    <p style="margin:4px 0 0;color:#94a3b8;font-size:13px;">
        AI-powered Return & Packing SOP Verification — Groq Vision
    </p>
</div>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY missing — add it in Streamlit Cloud → Settings → Secrets")
    st.stop()

client = Groq(api_key=api_key)

with st.sidebar:
    st.header("Configuration")
    video_type = st.selectbox("Process Stream Type", ["Return", "Forward"])
    uploaded_files = st.file_uploader(
        "Upload Inspection Videos",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )
    st.divider()
    num_frames = st.slider("Frames to sample per video", 5, 15, 10,
                           help="More frames = more accurate but slower")
    st.caption("Powered by Groq · LLaMA 3.2 Vision")

def extract_frames(video_path, n=10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    interval = max(1, total // n)
    frames_b64 = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * interval, total - 1))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (960, 720))
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frames_b64.append(base64.b64encode(buf).decode('utf-8'))
    cap.release()
    return frames_b64

def run_audit(video_bytes, stream_type, n_frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = extract_frames(tmp_path, n=n_frames)
        os.remove(tmp_path)

        if not frames:
            return {"score": 0, "verdict": "REJECTED", "tracking_number": "None",
                    "label_check": "Corrupted or unreadable video.",
                    "unboxing_check": "No footage recorded.",
                    "brand_tag_check": "No tags detected.",
                    "product_check": "Product not visible."}

        prompt = f"""
You are a VMS Return & Packing Audit Engine inspecting a logistics stream ({stream_type}).
Analyze these {n_frames} sequential video frames and generate a precise compliance report.

Evaluate against 4 mandatory SOP criteria:
1. Shipping Label & Barcode Visibility (30 pts): Is the outer shipping label or AWB clearly held toward the camera?
   IMPORTANT: Read and extract the exact barcode number, AWB number, or tracking number visible on any label.
   If you can read any alphanumeric string from a barcode or label, include it exactly as seen.
2. Packaging Integrity & Unboxing (20 pts): Is the outer seal inspected for tampering and opened fully on camera?
3. Brand Tag & Price Tag Verification (25 pts): Are brand labels, hangtags, price tags shown close to lens?
   State exact brand name, size, price, or tag condition visible.
4. Product Inspection & Condition (25 pts): Is the item fully unfolded and displayed on both sides?
   Describe fabric condition, defects, colour, buttons, or sleeves visible.

Respond ONLY with strict JSON — no extra text:
{{
    "score": <integer 0-100>,
    "verdict": "<ACCEPTED | REVIEW REQUIRED | REJECTED>",
    "tracking_number": "<exact barcode/AWB/tracking string read from label, or 'Not visible' if unreadable>",
    "label_check": "<specific observations about label and barcode visibility>",
    "unboxing_check": "<specific observations about package seal and unboxing>",
    "brand_tag_check": "<specific observations about brand tags, price tags, size tags>",
    "product_check": "<specific observations about product display and condition>"
}}
"""
        payload = [{"type": "text", "text": prompt}]
        for b64 in frames:
            payload.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        resp = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": payload}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"score": 0, "verdict": "ERROR", "tracking_number": "Error",
                "label_check": f"Error: {e}",
                "unboxing_check": "N/A", "brand_tag_check": "N/A", "product_check": "N/A"}

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

if not uploaded_files:
    st.info("Upload one or more inspection videos from the sidebar to begin.")
    st.stop()

st.subheader(f"Batch Inspection — {len(uploaded_files)} video(s) · Stream: {video_type}")

csv_rows = []

for file in uploaded_files:
    st.divider()

    vid_col, result_col = st.columns([1, 1], gap="large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_bytes = file.read()
        tmp.write(video_bytes)
        play_path = tmp.name

    with vid_col:
        st.markdown(f"**{file.name}**")
        st.video(play_path)

    with result_col:
        with st.spinner(f"Analyzing {file.name} with Groq Vision AI..."):
            audit = run_audit(video_bytes, video_type, num_frames)

        os.remove(play_path)

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

        # Tracking number read by AI
        st.markdown(
            f'<div style="margin-bottom:10px">🔍 <b style="color:#94a3b8">Tracking / AWB:</b> '
            f'<span class="barcode-pill">{tracking}</span></div>',
            unsafe_allow_html=True,
        )

        for label, key, weight in STEP_WEIGHTS:
            text = audit.get(key, "N/A")
            st.markdown(f"""
            <div class="step-card">
                <div class="step-title">{label} <span style="float:right;color:#475569;font-weight:400">{weight} pts</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{score}%;background:{color}"></div></div>
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
