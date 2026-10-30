"""CurbSense Lite — parking-bay analysis for photos and video clips."""
from __future__ import annotations

import hashlib
import html
import io
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


HERE = Path(__file__).resolve().parent
WEIGHT_CANDIDATES = (
    HERE / "bay_detector.pt",
    HERE / "runs" / "bay_detector_run" / "weights" / "best.pt",
)
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_VIDEO_BYTES = 150 * 1024 * 1024
MAX_OUTPUT_SIDE = 1280

# OpenCV uses BGR channel order.
FREE_BGR = (66, 190, 150)       # teal
OCCUPIED_BGR = (62, 135, 245)   # orange
UNKNOWN_BGR = (190, 112, 198)   # violet


st.set_page_config(
    page_title="CurbSense Lite | Smart Parking",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --cs-bg:#0b1120; --cs-panel:#121c2e; --cs-panel-2:#17243a; --cs-line:#25344c;
            --cs-text:#eef4ff; --cs-muted:#96a8c1; --cs-teal:#42d6ae; --cs-orange:#ff9b58; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #172947 0%, #0b1120 44%, #080d18 100%); color:var(--cs-text); }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .block-container { max-width:1500px; padding-top:1.7rem; padding-bottom:3rem; }
    section[data-testid="stSidebar"] { background:rgba(12,20,34,.96); border-right:1px solid var(--cs-line); }
    section[data-testid="stSidebar"] .block-container { padding-top:2rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.025em; color:var(--cs-text); }
    p, label, span { color:inherit; }
    .hero { border:1px solid #2a3c57; border-radius:22px; padding:25px 30px; margin:0 0 1.35rem 0;
            background:linear-gradient(115deg,rgba(31,58,86,.92),rgba(17,29,48,.93) 56%,rgba(18,39,53,.93));
            box-shadow:0 18px 55px rgba(0,0,0,.18); position:relative; overflow:hidden; }
    .hero:after { content:' '; position:absolute; width:250px; height:250px; border-radius:50%; right:6%; top:-130px;
                  background:rgba(66,214,174,.09); filter:blur(4px); }
    .eyebrow { color:#75e4c7; font-size:.72rem; letter-spacing:.17em; font-weight:700; text-transform:uppercase; }
    .hero-title { font:800 clamp(2rem,4vw,3rem) 'Manrope',sans-serif; line-height:1.1; margin:.4rem 0; color:#f4f8ff; }
    .hero-title span { color:#6ce0bd; }
    .hero-copy { color:#b5c4d9; max-width:720px; margin:.45rem 0 .9rem 0; font-size:1rem; }
    .hero-tags { display:flex; flex-wrap:wrap; gap:8px; }
    .hero-tag { border:1px solid #38516d; border-radius:999px; padding:5px 10px; font-size:.72rem; color:#d3e1f2; background:rgba(255,255,255,.035); }
    .section-label { color:#9eb1ca; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; margin:1rem 0 .45rem 0; }
    .kpi { border:1px solid #263751; border-radius:16px; padding:16px 17px; min-height:112px;
           background:linear-gradient(145deg,rgba(24,39,61,.98),rgba(17,28,45,.98)); }
    .kpi-label { color:#9fb0c8; font-size:.77rem; font-weight:600; letter-spacing:.04em; }
    .kpi-value { color:#f4f7ff; font:800 1.85rem 'Manrope',sans-serif; margin:.28rem 0 .12rem 0; }
    .kpi-note { color:#7f94b0; font-size:.72rem; }
    .status-pill { display:inline-block; border-radius:999px; padding:6px 11px; font-size:.76rem; font-weight:700; border:1px solid transparent; }
    .pill-good { color:#72e4c2; background:rgba(66,214,174,.10); border-color:rgba(66,214,174,.28); }
    .pill-warn { color:#ffc18e; background:rgba(255,155,88,.10); border-color:rgba(255,155,88,.28); }
    .pill-full { color:#ff9c9c; background:rgba(255,95,95,.10); border-color:rgba(255,95,95,.25); }
    .insight { border:1px solid #2b405b; border-left:4px solid #42d6ae; background:rgba(20,34,54,.88); padding:14px 16px; border-radius:12px; color:#dce8f8; }
    .muted { color:#96a8c1; font-size:.87rem; }
    div[data-testid="stMetric"] { border:1px solid #263751; border-radius:15px; padding:13px 15px; background:rgba(20,33,53,.88); }
    div[data-testid="stMetricLabel"] { color:#9fb0c8; }
    div[data-testid="stMetricValue"] { color:#f4f7ff; }
    div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#20b891,#36cda8); color:#071a19; border:0; border-radius:10px; font-weight:800; }
    div.stButton > button { border-radius:10px; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #263751; }
    .stTabs [data-baseweb="tab"] { background:rgba(21,34,54,.75); border-radius:10px 10px 0 0; padding:10px 16px; }
    .stTabs [aria-selected="true"] { background:rgba(66,214,174,.12) !important; }
    div[data-testid="stFileUploader"] { border:1px dashed #3a5270; border-radius:14px; background:rgba(17,29,47,.55); padding:8px; }
    .small-note { color:#8195b1; font-size:.76rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_weights() -> Path | None:
    return next((path for path in WEIGHT_CANDIDATES if path.is_file()), None)


@st.cache_resource(show_spinner="Loading the CurbSense detection model…")
def load_model(weights_path: str, modified_ns: int):
    # Import lazily so the dashboard can show a clear error if dependencies are missing.
    from ultralytics import YOLO

    return YOLO(weights_path)


def get_class_name(names: Any, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, names.get(str(class_id), f"class-{class_id}")))
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return f"class-{class_id}"


def classify_bay(class_name: str) -> str:
    label = class_name.strip().lower().replace("_", "-")
    if any(word in label for word in ("free", "available", "vacant", "unoccupied")):
        return "Free"
    if any(word in label for word in ("occupied", "taken", "busy")):
        return "Occupied"
    return "Unknown"


def predict_bays(model, image_bgr: np.ndarray, confidence: float, iou: float, image_size: int) -> list[dict]:
    result = model.predict(
        source=image_bgr,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        verbose=False,
    )[0]
    rows: list[dict] = []
    if result.boxes is None:
        return rows

    names = getattr(result, "names", getattr(model, "names", {}))
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence_score = float(box.conf.item())
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        class_name = get_class_name(names, class_id)
        rows.append(
            {
                "Bay": len(rows) + 1,
                "Class": class_name,
                "Status": classify_bay(class_name),
                "Confidence": confidence_score,
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
            }
        )
    return rows


def annotate_bgr(image_bgr: np.ndarray, rows: list[dict]) -> np.ndarray:
    base = image_bgr.copy()
    overlay = image_bgr.copy()
    height, width = base.shape[:2]

    for row in rows:
        x1 = max(0, min(width - 1, int(row["X1"])))
        y1 = max(0, min(height - 1, int(row["Y1"])))
        x2 = max(0, min(width - 1, int(row["X2"])))
        y2 = max(0, min(height - 1, int(row["Y2"])))
        colour = FREE_BGR if row["Status"] == "Free" else OCCUPIED_BGR if row["Status"] == "Occupied" else UNKNOWN_BGR
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, thickness=-1)

    annotated = cv2.addWeighted(overlay, 0.16, base, 0.84, 0)
    for row in rows:
        x1 = max(0, min(width - 1, int(row["X1"])))
        y1 = max(0, min(height - 1, int(row["Y1"])))
        x2 = max(0, min(width - 1, int(row["X2"])))
        y2 = max(0, min(height - 1, int(row["Y2"])))
        colour = FREE_BGR if row["Status"] == "Free" else OCCUPIED_BGR if row["Status"] == "Occupied" else UNKNOWN_BGR
        label = f"{row['Status']}  {row['Confidence']:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness=2, lineType=cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.68, width / 1800))
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 2)
        label_top = max(0, y1 - text_h - baseline - 8)
        cv2.rectangle(annotated, (x1, label_top), (min(width - 1, x1 + text_w + 12), y1), colour, -1)
        cv2.putText(annotated, label, (x1 + 6, max(text_h + 1, y1 - 6)), font, scale, (8, 18, 28), 2, cv2.LINE_AA)
    return annotated


def summarize(rows: list[dict]) -> dict:
    free = sum(row["Status"] == "Free" for row in rows)
    occupied = sum(row["Status"] == "Occupied" for row in rows)
    unknown = sum(row["Status"] == "Unknown" for row in rows)
    known = free + occupied
    occupancy = (occupied / known * 100) if known else None
    mean_conf = float(np.mean([row["Confidence"] for row in rows])) if rows else None
    if known == 0:
        rating = "Unclassified"
    elif occupancy < 40:
        rating = "Light"
    elif occupancy <= 75:
        rating = "Steady"
    else:
        rating = "Heavy"
    return {
        "total": len(rows), "free": free, "occupied": occupied, "unknown": unknown,
        "known": known, "occupancy": occupancy, "rating": rating, "mean_confidence": mean_conf,
    }


def recommendation(summary: dict) -> str:
    if summary["known"] == 0:
        if summary["total"]:
            return "Bays were detected, but the model's class labels could not be mapped to free or occupied. Check the model class names before relying on the occupancy estimate."
        return "No parking bays were detected. Try a clearer image, a different camera angle, or a slightly lower confidence threshold."
    free = summary["free"]
    if free == 0:
        return "The detected parking area appears full. Drivers should check another parking area."
    noun = "space" if free == 1 else "spaces"
    if summary["rating"] == "Heavy":
        return f"Only {free} {noun} appear available. The lot is heavily occupied; consider checking another area."
    if summary["rating"] == "Steady":
        return f"{free} {noun} appear available. There is room, but the lot is filling up."
    return f"{free} {noun} appear available. The detected area has relatively light occupancy."


def render_kpis(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        with column:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
                f'<div class="kpi-value">{html.escape(value)}</div>'
                f'<div class="kpi-note">{html.escape(note)}</div></div>',
                unsafe_allow_html=True,
            )


def render_status(summary: dict) -> None:
    rating = summary["rating"]
    css_class = "pill-good" if rating == "Light" else "pill-warn" if rating == "Steady" else "pill-full"
    if rating == "Unclassified":
        css_class = "pill-warn"
    st.markdown(
        f'<span class="status-pill {css_class}">LOT STATUS · {html.escape(rating.upper())}</span>',
        unsafe_allow_html=True,
    )


def rows_to_csv(rows: list[dict]) -> bytes:
    frame = pd.DataFrame(rows)
    if "Confidence" in frame.columns:
        frame["Confidence"] = frame["Confidence"].map(lambda value: round(float(value), 4))
    return frame.to_csv(index=False).encode("utf-8-sig")


def add_history(source_type: str, filename: str, summary: dict) -> None:
    history = st.session_state.setdefault("history", [])
    history.insert(
        0,
        {
            "Analysed at": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
            "Type": source_type,
            "File": filename,
            "Bays detected": summary["total"],
            "Available": summary["free"],
            "Occupied": summary["occupied"],
            "Occupancy (%)": round(summary["occupancy"], 1) if summary["occupancy"] is not None else "—",
            "Status": summary["rating"],
        },
    )
    del history[50:]


def resize_for_video(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    longest = max(height, width)
    if longest > MAX_OUTPUT_SIDE:
        scale = MAX_OUTPUT_SIDE / longest
        width, height = int(width * scale), int(height * scale)
    # MP4 encoders are more reliable with even frame dimensions.
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    if (width, height) != (frame_bgr.shape[1], frame_bgr.shape[0]):
        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return frame_bgr


def create_video_writer(path: str, fps: float, size: tuple[int, int]):
    # H.264 is preferred for browser playback; MPEG-4 Part 2 is the portable fallback.
    for codec in ("avc1", "mp4v"):
        if os.path.exists(path):
            os.remove(path)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("This environment could not create an MP4 video. Try a shorter MP4/AVI source clip.")


def process_video(
    video_bytes: bytes,
    suffix: str,
    model,
    confidence: float,
    iou: float,
    image_size: int,
    requested_step: int,
    max_samples: int,
    show_preview: bool,
) -> tuple[str, pd.DataFrame, dict, int, float, str]:
    input_path = None
    output_path = None
    capture = None
    writer = None
    progress = st.progress(0, text="Preparing video…")
    preview = st.empty() if show_preview else None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as source_file:
            source_file.write(video_bytes)
            input_path = source_file.name

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened. Try exporting it as MP4 (H.264) and uploading again.")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = max(1, int(requested_step))
        if total_frames > 0:
            frame_step = max(frame_step, math.ceil(total_frames / max(1, max_samples)))
        # Reduce output FPS in proportion to the sampling interval so the exported clip
        # stays close to the source duration without writing duplicate full-rate frames.
        output_fps = min(source_fps, max(1.0, source_fps / frame_step))
        hold_count = max(1, int(round(frame_step * output_fps / source_fps)))

        file_descriptor, output_path = tempfile.mkstemp(prefix="curbsense_annotated_", suffix=".mp4")
        os.close(file_descriptor)
        os.remove(output_path)

        frame_index = 0
        analysed = 0
        records: list[dict] = []
        codec_used = ""
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if frame_index % frame_step == 0 and analysed < max_samples:
                frame_bgr = resize_for_video(frame_bgr)
                rows = predict_bays(model, frame_bgr, confidence, iou, image_size)
                annotated = annotate_bgr(frame_bgr, rows)
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer, codec_used = create_video_writer(output_path, output_fps, (width, height))

                # Hold the analysed frame for the sampled interval at the reduced output FPS.
                repeats = hold_count
                if total_frames > 0:
                    remaining_frames = max(1, total_frames - frame_index)
                    repeats = max(1, int(round(min(frame_step, remaining_frames) * output_fps / source_fps)))
                for _ in range(repeats):
                    writer.write(annotated)

                stats = summarize(rows)
                records.append(
                    {
                        "Time (seconds)": round(frame_index / source_fps, 2),
                        "Frame": frame_index,
                        "Bays detected": stats["total"],
                        "Available": stats["free"],
                        "Occupied": stats["occupied"],
                        "Unclassified": stats["unknown"],
                        "Occupancy (%)": round(stats["occupancy"], 2) if stats["occupancy"] is not None else np.nan,
                        "Mean confidence (%)": round(stats["mean_confidence"] * 100, 1) if stats["mean_confidence"] is not None else np.nan,
                    }
                )
                analysed += 1

                if preview is not None:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Analysed frame {frame_index + 1}", use_container_width=True)
                if total_frames > 0:
                    progress.progress(min(1.0, frame_index / total_frames), text=f"Analysing sampled frame {analysed}…")
                else:
                    progress.progress(min(0.98, analysed / max_samples), text=f"Analysing sampled frame {analysed}…")

            frame_index += 1
            if total_frames <= 0 and analysed >= max_samples:
                break

        if capture is not None:
            capture.release()
            capture = None
        if writer is not None:
            writer.release()
            writer = None
        progress.progress(1.0, text="Video analysis complete")

        if not records or not output_path or not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("No frames could be analysed. Please try a different video file.")

        frame_df = pd.DataFrame(records)
        valid_occupancy = frame_df["Occupancy (%)"].dropna()
        valid_free = frame_df["Available"].dropna()
        valid_occupied = frame_df["Occupied"].dropna()
        summary = {
            "frames_analysed": len(frame_df),
            "duration_seconds": (total_frames / source_fps) if total_frames > 0 else (frame_index / source_fps),
            "mean_occupancy": float(valid_occupancy.mean()) if not valid_occupancy.empty else None,
            "peak_occupancy": float(valid_occupancy.max()) if not valid_occupancy.empty else None,
            "mean_free": float(valid_free.mean()) if not valid_free.empty else 0.0,
            "mean_occupied": float(valid_occupied.mean()) if not valid_occupied.empty else 0.0,
        }
        return output_path, frame_df, summary, frame_step, source_fps, codec_used

    except Exception:
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Vision intelligence · Parking operations</div>
          <div class="hero-title">CURBSENSE <span>LITE</span></div>
          <div class="hero-copy">A clearer view of parking availability. Analyse a still image or sample a video clip to understand how occupancy changes across a parking area.</div>
          <div class="hero-tags"><span class="hero-tag">YOLOv8 detection</span><span class="hero-tag">Image + video analysis</span><span class="hero-tag">Downloadable reports</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    weights_path = find_weights()
    if weights_path is None:
        st.error("The trained model weights were not found. Place `bay_detector.pt` beside `app.py`, or include `runs/bay_detector_run/weights/best.pt`.")
        st.stop()

    with st.sidebar:
        st.markdown("### Analysis controls")
        confidence = st.slider("Detection confidence", 0.10, 0.90, 0.40, 0.05, help="Higher values are stricter and may ignore weaker detections.")
        iou = st.slider("Overlap / NMS threshold", 0.20, 0.90, 0.70, 0.05, help="Controls how overlapping detections are filtered.")
        image_size = st.select_slider("Model image size", options=[480, 640, 800, 960], value=640)
        st.markdown("---")
        st.markdown("**Detection key**")
        st.markdown("🟩 Free bay  \n🟧 Occupied bay  \n🟪 Unclassified")
        st.markdown('<div class="small-note">The app reports detections from the uploaded media. It does not connect to live cameras or a live parking database.</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption(f"Model file · `{weights_path.name}`")

    image_tab, video_tab, history_tab, info_tab = st.tabs(
        ["📷  Image analysis", "🎞️  Video analysis", "📊  Session history", "ⓘ  How it works"]
    )

    with image_tab:
        st.markdown("## Parking snapshot")
        st.markdown('<div class="muted">Upload a clear parking-lot image to detect and classify visible bays.</div>', unsafe_allow_html=True)
        image_file = st.file_uploader("Choose a parking-lot image", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
        if image_file is not None:
            image_bytes = image_file.getvalue()
            image_signature = hashlib.sha256(image_bytes).hexdigest()
            if len(image_bytes) > MAX_IMAGE_BYTES:
                st.error("This image is larger than 30 MB. Resize or compress it, then upload again.")
            else:
                left, right = st.columns([1, 1])
                with left:
                    st.markdown('<div class="section-label">Selected image</div>', unsafe_allow_html=True)
                    try:
                        source_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
                        st.image(source_image, caption=f"{image_file.name} · {source_image.width} × {source_image.height}", use_container_width=True)
                    except (UnidentifiedImageError, OSError):
                        source_image = None
                        st.error("The selected file is not a readable image.")

                run_image = st.button("Analyse image", type="primary", disabled=source_image is None, use_container_width=True, key="run_image")
                if run_image and source_image is not None:
                    try:
                        with st.spinner("Detecting parking bays…"):
                            model = load_model(str(weights_path), weights_path.stat().st_mtime_ns)
                            image_rgb = np.asarray(source_image)
                            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                            detections = predict_bays(model, image_bgr, confidence, iou, image_size)
                            annotated_bgr = annotate_bgr(image_bgr, detections)
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                            output = io.BytesIO()
                            Image.fromarray(annotated_rgb).save(output, format="PNG")
                            stats = summarize(detections)
                            st.session_state["image_result"] = {
                                "signature": image_signature,
                                "annotated_png": output.getvalue(),
                                "rows": detections,
                                "summary": stats,
                                "filename": image_file.name,
                            }
                            add_history("Image", image_file.name, stats)
                    except Exception as exc:
                        st.error(f"Image analysis failed: {exc}")

                result = st.session_state.get("image_result")
                if result and result.get("signature") == image_signature:
                    stats = result["summary"]
                    with right:
                        st.markdown('<div class="section-label">AI-annotated result</div>', unsafe_allow_html=True)
                        st.image(result["annotated_png"], caption="Detected bays · teal = free · orange = occupied", use_container_width=True)
                    st.markdown("---")
                    render_status(stats)
                    render_kpis(
                        [
                            ("Bays detected", str(stats["total"]), "All model detections"),
                            ("Available", str(stats["free"]), "Classified as free"),
                            ("Occupied", str(stats["occupied"]), "Classified as taken"),
                            ("Occupancy", f"{stats['occupancy']:.1f}%" if stats["occupancy"] is not None else "—", "Of classifiable bays"),
                        ]
                    )
                    st.markdown('<div class="section-label">Parking insight</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="insight">{html.escape(recommendation(stats))}</div>', unsafe_allow_html=True)
                    if stats["unknown"]:
                        st.warning(f"{stats['unknown']} detected bay(s) have labels the app could not classify. They are excluded from the occupancy percentage.")
                    if stats["mean_confidence"] is not None:
                        st.caption(f"Average detection confidence: {stats['mean_confidence']:.1%}")

                    chart_col, table_col = st.columns([1, 1.35])
                    with chart_col:
                        st.markdown("#### Bay distribution")
                        distribution = pd.DataFrame(
                            {"Detected bays": [stats["free"], stats["occupied"], stats["unknown"]]},
                            index=["Free", "Occupied", "Unclassified"],
                        )
                        st.bar_chart(distribution)
                    with table_col:
                        st.markdown("#### Detection details")
                        table = pd.DataFrame(result["rows"])
                        if not table.empty:
                            table["Confidence"] = (table["Confidence"] * 100).round(1).astype(str) + "%"
                            st.dataframe(table, use_container_width=True, hide_index=True)
                        else:
                            st.info("No parking bays were detected in this image.")
                    download_col1, download_col2 = st.columns(2)
                    with download_col1:
                        st.download_button("Download annotated image", result["annotated_png"], file_name="curbsense_annotated.png", mime="image/png", use_container_width=True)
                    with download_col2:
                        st.download_button("Download detection CSV", rows_to_csv(result["rows"]), file_name="curbsense_image_report.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("Upload a JPG, PNG or WEBP image to begin.")

    with video_tab:
        st.markdown("## Parking video review")
        st.markdown('<div class="muted">Upload a clip to sample frames, annotate detections and review occupancy changes over time.</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Choose a parking-lot video", type=["mp4", "mov", "avi", "mkv", "m4v"], key="video_upload")
        if video_file is not None:
            video_bytes = video_file.getvalue()
            video_signature = hashlib.sha256(video_bytes).hexdigest()
            if len(video_bytes) > MAX_VIDEO_BYTES:
                st.error("This video is larger than 150 MB. Trim or compress the clip, then upload again.")
            else:
                st.caption(f"Selected: {video_file.name} · {len(video_bytes) / (1024 * 1024):.1f} MB")
                control1, control2, control3 = st.columns([1, 1, 1])
                with control1:
                    frame_interval = st.select_slider("Analyse every N frames", options=[1, 2, 3, 5, 8, 10, 15], value=5, help="The app analyses one frame at each interval and holds its annotation between samples.")
                with control2:
                    max_frames = st.select_slider("Maximum analysed frames", options=[60, 120, 180, 240, 360, 480], value=180)
                with control3:
                    preview_enabled = st.checkbox("Show processing preview", value=False)

                if st.button("Analyse video", type="primary", use_container_width=True, key="run_video"):
                    try:
                        with st.spinner("Processing video frames…"):
                            model = load_model(str(weights_path), weights_path.stat().st_mtime_ns)
                            suffix = Path(video_file.name).suffix.lower() or ".mp4"
                            output_path, frame_df, video_summary, actual_step, source_fps, codec = process_video(
                                video_bytes,
                                suffix,
                                model,
                                confidence,
                                iou,
                                image_size,
                                frame_interval,
                                max_frames,
                                preview_enabled,
                            )
                            st.session_state["video_result"] = {
                                "signature": video_signature,
                                "output_path": output_path,
                                "frame_df": frame_df,
                                "summary": video_summary,
                                "filename": video_file.name,
                                "frame_step": actual_step,
                                "source_fps": source_fps,
                                "codec": codec,
                            }
                            video_history_summary = {
                                "total": int(round(video_summary["mean_free"] + video_summary["mean_occupied"])),
                                "free": int(round(video_summary["mean_free"])),
                                "occupied": int(round(video_summary["mean_occupied"])),
                                "unknown": 0,
                                "occupancy": video_summary["mean_occupancy"],
                                "rating": "Unclassified" if video_summary["mean_occupancy"] is None else "Light" if video_summary["mean_occupancy"] < 40 else "Steady" if video_summary["mean_occupancy"] <= 75 else "Heavy",
                            }
                            add_history("Video", video_file.name, video_history_summary)
                    except Exception as exc:
                        st.error(f"Video analysis failed: {exc}")

                result = st.session_state.get("video_result")
                if result and result.get("signature") == video_signature:
                    summary = result["summary"]
                    st.markdown("---")
                    st.markdown("### Annotated video")
                    if os.path.isfile(result["output_path"]):
                        st.video(result["output_path"])
                        with open(result["output_path"], "rb") as video_handle:
                            st.download_button("Download annotated video", video_handle.read(), file_name="curbsense_annotated.mp4", mime="video/mp4", use_container_width=True)
                    else:
                        st.warning("The temporary processed video is no longer available. Analyse the source clip again to regenerate it.")

                    avg_occ = f"{summary['mean_occupancy']:.1f}%" if summary["mean_occupancy"] is not None else "—"
                    peak_occ = f"{summary['peak_occupancy']:.1f}%" if summary["peak_occupancy"] is not None else "—"
                    render_kpis(
                        [
                            ("Average occupancy", avg_occ, "Across analysed frames"),
                            ("Peak occupancy", peak_occ, "Highest sampled frame"),
                            ("Avg. available", f"{summary['mean_free']:.1f}", "Bays per analysed frame"),
                            ("Frames analysed", str(summary["frames_analysed"]), f"Every {result['frame_step']} source frames"),
                        ]
                    )
                    st.caption(f"Source duration: {summary['duration_seconds']:.1f}s · Source frame rate: {result['source_fps']:.1f} FPS · Video encoder: {result['codec']}")
                    st.info("Video detections are sampled snapshots, not tracked parking-bay identities. The exported clip uses a reduced frame rate to keep its duration close to the source. The chart shows the model's estimate at each sampled time.")
                    trend = result["frame_df"].dropna(subset=["Occupancy (%)"])
                    chart_col, report_col = st.columns([1.25, 1])
                    with chart_col:
                        st.markdown("#### Occupancy over time")
                        if not trend.empty:
                            st.line_chart(trend.set_index("Time (seconds)")["Occupancy (%)"])
                        else:
                            st.info("No classifiable bays were found in the sampled frames.")
                    with report_col:
                        st.markdown("#### Frame-by-frame summary")
                        st.dataframe(result["frame_df"], use_container_width=True, hide_index=True)
                        st.download_button("Download video analysis CSV", result["frame_df"].to_csv(index=False).encode("utf-8-sig"), file_name="curbsense_video_report.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("Upload an MP4, MOV, AVI, MKV or M4V clip to begin. Short clips work best for a quick review.")

    with history_tab:
        st.markdown("## Recent analyses")
        history = st.session_state.get("history", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
            if st.button("Clear session history", key="clear_history"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.info("Analyses from this browser session will appear here. History is cleared when the session resets.")

    with info_tab:
        st.markdown("## How CurbSense works")
        st.markdown(
            """
            1. **Upload media.** Choose a parking-lot image or a short video clip.
            2. **Run detection.** The included YOLOv8 model identifies visible parking bays and their trained classes.
            3. **Review the result.** Free bays are shown in teal, occupied bays in orange, and any unrecognised class in violet.
            4. **Export.** Download the annotated image/video and a CSV report for further review.

            **Important limitations**
            - Results depend on the camera angle, image quality, lighting and the model's training data.
            - The occupancy percentage is calculated from detected bays whose classes map to free or occupied; unclassified detections are excluded.
            - Video mode samples frames and exports the annotated samples at a reduced frame rate. It is not live CCTV, object tracking, or a live parking reservation system.
            - The app analyses only the media uploaded by the user. It does not verify whether a space remains available after the image or clip was captured.
            """
        )
        st.markdown("---")
        st.caption("CurbSense Lite · YOLOv8 parking-bay detection · Built with Streamlit and OpenCV")


if __name__ == "__main__":
    main()
