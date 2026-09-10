import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import streamlit as st

from config import SOP_SEGMENTS, STEP_ORDER, STEP_PROMPTS, STEP_WEIGHTS
from scoring import extract_tracking_number, score_response, verdict_for_score
from video_utils import extract_step_frames
from vlm_client import query_vlm_multi

st.set_page_config(page_title="VMS Inspection Dashboard", layout="wide", page_icon="📦")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #f0f0f0; }
.header-box {
    background: #1a1d27; padding: 16px 20px; border-radius: 8px;
    margin-bottom: 20px; border-left: 4px solid #3b82f6;
}
.verdict-accepted { background:#1a4731; color:#4ade80; padding:4px 14px; border-radius:5px; font-weight:700; font-size:13px; }
.verdict-review { background:#3d2c00; color:#fbbf24; padding:4px 14px; border-radius:5px; font-weight:700; font-size:13px; }
.verdict-rejected { background:#3d0a0a; color:#f87171; padding:4px 14px; border-radius:5px; font-weight:700; font-size:13px; }
.big-score { font-size:38px; font-weight:700; }
.step-card {
    background:#12151f; border:0.5px solid #2a2d3a;
    border-radius:8px; padding:12px 14px; margin-bottom:10px;
}
.step-title { font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:6px; }
.step-body { font-size:13px; color:#94a3b8; line-height:1.6; }
.bar-bg { background:#2a2d3a; border-radius:4px; height:7px; overflow:hidden; margin:6px 0 10px; }
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

HEADERS = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    video_type = st.selectbox("Process Stream Type", list(SOP_SEGMENTS.keys()))
    uploaded_files = st.file_uploader(
        "Upload Inspection Videos",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )
    st.divider()
    num_frames = st.slider("Frames per step", 2, 5, 2,
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
            unsafe_allow_html=True,
        )


# ── Main audit ────────────────────────────────────────────────────────────────
def run_audit(video_path, stream_type, n_per_step, log_placeholder):
    """
    Runs the full audit against an already-written-to-disk video file.
    (The caller owns the temp file's lifecycle — this function does not
    create or delete it, so the same file can be reused for both playback
    and processing instead of being written to disk twice.)
    """
    log(f"Starting audit — {stream_type}")
    show_log_box(log_placeholder)

    try:
        step_segments = SOP_SEGMENTS[stream_type]
        step_frames = extract_step_frames(video_path, step_segments, n_per_step, log=log)
        show_log_box(log_placeholder)

        if not step_frames:
            return {
                "score": 0, "verdict": "REJECTED", "tracking_number": "None",
                "label_check": "Could not read video.",
                "unboxing_check": "N/A", "brand_tag_check": "N/A", "product_check": "N/A",
            }

        step_map = [
            ("label", "label_check"),
            ("unboxing", "unboxing_check"),
            ("tags", "brand_tag_check"),
            ("product", "product_check"),
        ]

        def analyze_step(step_key, result_key):
            log(f"--- Analyzing: {step_key} ---")
            frames = step_frames.get(step_key, [])
            if not frames:
                return result_key, "No frames available for this step."
            return result_key, query_vlm_multi(frames, STEP_PROMPTS[step_key], step_key, HEADERS, log=log)

        results = {}
        with ThreadPoolExecutor(max_workers=len(step_map)) as executor:
            futures = [executor.submit(analyze_step, step_key, result_key) for step_key, result_key in step_map]
            for future in as_completed(futures):
                result_key, text = future.result()
                results[result_key] = text
                show_log_box(log_placeholder)

        tracking = extract_tracking_number(results.get("label_check", ""))
        log(f"Tracking number: {tracking}")

        s1 = score_response(results["label_check"], 30)
        s2 = score_response(results["unboxing_check"], 20)
        s3 = score_response(results["brand_tag_check"], 25)
        s4 = score_response(results["product_check"], 25)
        total = s1 + s2 + s3 + s4
        log(f"Scores — Label:{s1}/30 Unboxing:{s2}/20 Tags:{s3}/25 Product:{s4}/25 = {total}/100")

        verdict = verdict_for_score(total)
        log(f"Verdict: {verdict}")

        return {"score": total, "verdict": verdict, "tracking_number": tracking, **results}

    except Exception as e:
        log(f"FATAL: {str(e)}", "ERROR")
        return {
            "score": 0, "verdict": "ERROR", "tracking_number": "Error",
            "label_check": f"Error: {e}",
            "unboxing_check": "N/A", "brand_tag_check": "N/A", "product_check": "N/A",
        }


def score_color(s):
    return "#22c55e" if s >= 85 else "#f59e0b" if s >= 50 else "#ef4444"


def verdict_badge(s):
    if s >= 85:
        return '<span class="verdict-accepted">✅ ACCEPTED</span>'
    if s >= 50:
        return '<span class="verdict-review">⚠️ REVIEW REQUIRED</span>'
    return '<span class="verdict-rejected">❌ REJECTED</span>'


# ── UI ────────────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.info("Upload one or more inspection videos from the sidebar to begin.")
    st.stop()

if not st.button("Analyze uploaded videos", type="primary", use_container_width=True):
    st.info("Your videos are ready. Start the batch analysis when you are ready.")
    st.stop()

st.subheader(f"Batch Inspection — {len(uploaded_files)} video(s) · Stream: {video_type}")

csv_rows = []

for file in uploaded_files:
    st.divider()
    log_lines.clear()
    vid_col, result_col = st.columns([1, 1], gap="large")

    extension = os.path.splitext(file.name)[1].lower() or ".mp4"
    video_bytes = file.getvalue()

    # Single temp-file write, reused for both the <video> preview and the
    # OpenCV processing pass (previously written to disk twice).
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    with vid_col:
        st.markdown(f"**{file.name}**")
        st.video(tmp_path)
        if show_logs:
            st.markdown("**Debug Log**")
            log_placeholder = st.empty()
        else:
            log_placeholder = st.empty()

    with result_col:
        with st.spinner(f"Analyzing {file.name}..."):
            log(f"Processing: {file.name} ({len(video_bytes)/1024/1024:.1f} MB)")
            show_log_box(log_placeholder)
            audit = run_audit(tmp_path, video_type, num_frames, log_placeholder)
            show_log_box(log_placeholder)

        score = audit.get("score", 0)
        verdict = audit.get("verdict", "REJECTED")
        tracking = audit.get("tracking_number", "Not visible")
        color = score_color(score)

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

    # Video file is only needed for the duration of this file's preview + processing.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    csv_rows.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "File": file.name,
        "Stream Type": video_type,
        "Verdict": verdict,
        "Score": score,
        "Tracking / AWB": tracking,
        "Label & Barcode": audit.get("label_check", ""),
        "Packaging": audit.get("unboxing_check", ""),
        "Brand Tags": audit.get("brand_tag_check", ""),
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
