import sys, os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from src.roi import detect_two_stage_roi, crop_roi
from src.image_preprocessing import adaptive_preprocess_pipeline
from src.strip_measurement import extract_local_strips, compute_strip_profiles
from src.wire_detection import (
    detect_orientation,
    detect_wire_positions,
    validate_wire_count
)
from src.wire_tracking import track_wires_across_strips, compute_robust_wire_positions
from src.confidence import calculate_confidence
from src.gap_measurement import evaluate_gaps
from src.calibration import compute_pixels_per_mm
from src.visualization import draw_wire_annotations, plot_1d_projection_profile
from src.synthetic import generate_synthetic_wire_belt

# Streamlit version compatibility helpers
def st_image_compat(image, caption=None, width="stretch"):
    try:
        st.image(image, caption=caption, width=width)
    except Exception:
        st.image(image, caption=caption, use_container_width=True)

def st_dataframe_compat(df, width="stretch"):
    try:
        st.dataframe(df, width=width)
    except Exception:
        st.dataframe(df, use_container_width=True)

# Page configuration
st.set_page_config(
    page_title="Wire Gap Detection System — CV Prototype",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for industrial theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .status-normal {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-abnormal {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-warning {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .calibration-banner {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">REAL-TIME-CAPABLE WIRE GAP DETECTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computer Vision Inspection Prototype — 48 Parallel Steel Wires (0.9 mm Diameter)</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Control Panel")

# Processing Mode Selection
processing_mode = st.sidebar.radio(
    "Processing Mode",
    ["Mode A: Wire Detection Verification", "Mode B: Measurement & Gap Analysis"],
    help="Mode A verifies 48-wire identification; Mode B performs pitch/gap measurement and abnormality evaluation."
)

debug_mode = st.sidebar.checkbox("🐞 DEBUG Mode", value=False, help="Displays raw candidates, rejected peaks, multi-strip positions, and tracking support scores")

# Input source selection
input_source_type = st.sidebar.selectbox(
    "Input Image Source",
    ["Synthetic Demo Suite", "Sample Image (1-7)", "Upload Custom Image", "Live Camera / Stream"]
)

input_image = None
image_title = "Input Image"

if input_source_type == "Synthetic Demo Suite":
    scenario = st.sidebar.selectbox(
        "Synthetic Test Scenario",
        [
            "Scenario 1: Normal Spacing (48 Wires)",
            "Scenario 2: Artificial Abnormal Gap (1 Displaced Wire)",
            "Scenario 3: Missing Wire (47 Wires)",
            "Scenario 4: Noisy Illumination & Sensor Noise",
            "Scenario 5: Curved Wires (Sinusoidal Curvature)",
            "Scenario 6: Displaced Local Wire Portion",
            "Scenario 7: Reflection Artifact Glare Highlight",
            "Scenario 8: Partial Wire Obscuration / Visibility"
        ]
    )
    sc_key_map = {
        "Scenario 1: Normal Spacing (48 Wires)": "normal",
        "Scenario 2: Artificial Abnormal Gap (1 Displaced Wire)": "abnormal_gap",
        "Scenario 3: Missing Wire (47 Wires)": "missing_wire",
        "Scenario 4: Noisy Illumination & Sensor Noise": "noisy_illumination",
        "Scenario 5: Curved Wires (Sinusoidal Curvature)": "curved_wires",
        "Scenario 6: Displaced Local Wire Portion": "displaced_local_wire",
        "Scenario 7: Reflection Artifact Glare Highlight": "reflection_artifact",
        "Scenario 8: Partial Wire Obscuration / Visibility": "partial_obscuration"
    }
    sc_key = sc_key_map[scenario]
    input_image, _ = generate_synthetic_wire_belt(scenario=sc_key, num_wires=48)
    image_title = f"Synthetic: {scenario}"

elif input_source_type == "Sample Image (1-7)":
    sample_list = [f"image{i}.jpeg" for i in range(1, 8)]
    if os.path.exists(os.path.join(root_dir, 'sample_images')):
        syn_files = [os.path.join('sample_images', f) for f in os.listdir(os.path.join(root_dir, 'sample_images')) if f.endswith('.jpeg')]
        sample_list.extend(syn_files)
        
    sample_idx = st.sidebar.selectbox("Select Sample Image", sample_list)
    sample_path = os.path.join(root_dir, sample_idx)
    if os.path.exists(sample_path):
        input_image = cv2.imread(sample_path)
        image_title = f"Sample: {sample_idx}"
    else:
        st.error(f"Sample image {sample_idx} not found.")

elif input_source_type == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)
        image_title = f"Uploaded: {uploaded_file.name}"

elif input_source_type == "Live Camera / Stream":
    camera_idx = st.sidebar.number_input("Camera Index", value=0, step=1)
    if st.sidebar.button("Capture Frame from Camera"):
        cap = cv2.VideoCapture(camera_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            input_image = frame
            image_title = "Live Camera Frame"
        else:
            st.sidebar.error("Could not capture frame from camera.")

# Fallback default image
if input_image is None:
    input_image, _ = generate_synthetic_wire_belt(scenario='normal', num_wires=48)
    image_title = "Synthetic: Scenario 1: Normal Spacing (48 Wires)"

# --- CALIBRATION SETTINGS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Calibration Settings")

enable_calibration = st.sidebar.checkbox("Enable Pixel-to-mm Calibration", value=False)
pixels_per_mm = None

if enable_calibration:
    calib_mode = st.sidebar.radio("Calibration Method", ["Direct px/mm Entry", "Known Physical Reference"])
    if calib_mode == "Direct px/mm Entry":
        pixels_per_mm = st.sidebar.number_input("Pixels per mm (px/mm)", value=20.0, min_value=1.0, max_value=200.0, step=0.5)
    else:
        known_mm = st.sidebar.number_input("Known Distance (mm)", value=10.0, min_value=0.1, step=0.5)
        measured_px = st.sidebar.number_input("Measured Image Distance (px)", value=200.0, min_value=1.0, step=5.0)
        pixels_per_mm = compute_pixels_per_mm(measured_px, known_mm)
        st.sidebar.info(f"Calculated Scale: **{pixels_per_mm:.2f} px/mm**")
else:
    st.sidebar.info("⚠️ **Calibration required** — Measurements displayed in pixels (px)")

# --- THRESHOLD SETTINGS ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Threshold Configuration")

use_demo_threshold = st.sidebar.checkbox("Use Demo Threshold", value=True, help="Demo threshold — not an engineering specification")

if pixels_per_mm is not None:
    max_gap_input = st.sidebar.number_input(
        "Max Allowed Gap (mm)",
        value=2.0 if use_demo_threshold else 1.5,
        min_value=0.1,
        max_value=10.0,
        step=0.1
    )
else:
    max_gap_input = st.sidebar.number_input(
        "Max Allowed Gap (px)",
        value=20.0 if use_demo_threshold else 18.0,
        min_value=1.0,
        max_value=100.0,
        step=1.0
    )

# --- INDEPENDENT PARAMETERS ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Wire Parameters")
st.sidebar.markdown("**Wire Diameter ($D$)**: 0.90 mm *(physical wire thickness)*")
expected_wires = st.sidebar.number_input("Expected Wire Count", value=48, min_value=1, max_value=100, step=1)

# --- PIPELINE CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Pipeline Fine-Tuning")
orientation_opt = st.sidebar.selectbox("Wire Orientation", ["Auto Detect", "HORIZONTAL", "VERTICAL"])
channel_mode_opt = st.sidebar.selectbox("Representation Channel", ["AUTO (Quality Score Selector)", "R-G", "RED", "BLACK_HAT", "INVERT_GRAY", "GRAY"])
channel_mode_clean = "AUTO" if "AUTO" in channel_mode_opt else channel_mode_opt

manual_prominence = st.sidebar.slider("Detection Sensitivity Prominence", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
force_deskew = st.sidebar.checkbox("Force Deskewing / Alignment", value=False)
trim_mounting_rails = st.sidebar.checkbox("Trim Mounting Rails from ROI", value=True)

# Manual Crop ROI Adjustments Fallback
enable_manual_roi = st.sidebar.checkbox("Enable Manual ROI Crop Sliders", value=False)
if enable_manual_roi and input_image is not None:
    ih, iw = input_image.shape[:2]
    crop_x = st.sidebar.slider("X Range (px)", 0, iw, (0, iw))
    crop_y = st.sidebar.slider("Y Range (px)", 0, ih, (0, ih))
    manual_bbox = (crop_x[0], crop_y[0], crop_x[1] - crop_x[0], crop_y[1] - crop_y[0])
else:
    manual_bbox = None

# --- MAIN EXECUTION ENGINE ---
start_time = time.time()

# 1. Orientation Detection FIRST
if orientation_opt == "Auto Detect":
    orientation = detect_orientation(input_image)
else:
    orientation = orientation_opt

# 2. Stage 2 Directional Two-Stage ROI Localization
if manual_bbox is not None:
    roi_box = manual_bbox
else:
    roi_box = detect_two_stage_roi(input_image, orientation=orientation, trim_rails=trim_mounting_rails)

roi = crop_roi(input_image, roi_box)

# 3. Preprocessing with matched feature polarity
prep_results = adaptive_preprocess_pipeline(
    roi,
    orientation=orientation,
    channel_mode=channel_mode_clean,
    force_deskew=force_deskew
)

processed_roi = prep_results['processed_roi']
feature_channel = prep_results['feature_channel']
selected_mode = prep_results.get('selected_mode', channel_mode_clean)
matched_polarity = prep_results['polarity']
binary_mask = prep_results['binary']
seg_quality = prep_results['segmentation_quality']
deskew_angle = prep_results['deskew_angle']

# 4. Multi-strip extraction (20% to 80% active range)
strips = extract_local_strips(processed_roi, orientation=orientation)
profiles = compute_strip_profiles(strips, orientation=orientation, channel=feature_channel)

# 5. Local strip candidate detection with MATCHED POLARITY
strip_detections = []
all_raw_candidates = []
all_rejected_candidates = []

for p_info in profiles:
    det = detect_wire_positions(p_info['profile'], detection_mode=matched_polarity, manual_prominence=manual_prominence)
    strip_detections.append(det)
    all_raw_candidates.extend(det['raw_candidates'])
    all_rejected_candidates.extend(det['rejected_candidates'])

# 6. Pitch bootstrap from High/Medium quality strips
valid_pitches = [d['estimated_pitch'] for d in strip_detections if d['quality_category'] in ['HIGH QUALITY', 'MEDIUM QUALITY']]
ref_pitch = float(np.median(valid_pitches)) if len(valid_pitches) > 0 else 15.0

# 7. Monotonic ordered cross-strip tracking
track_res = track_wires_across_strips(strip_detections, reference_pitch=ref_pitch, max_jump_alpha=config.MAX_JUMP_ALPHA)
robust_res = compute_robust_wire_positions(track_res['trajectories'])

positions = robust_res['positions']

count_validation = validate_wire_count(len(positions), expected_count=expected_wires)

# 8. FINAL pitch recalculation AFTER tracking
gap_eval = evaluate_gaps(
    positions,
    pixels_per_mm=pixels_per_mm,
    wire_diameter_mm=0.9,
    max_gap_user=max_gap_input,
    demo_mode=use_demo_threshold
)

confidence_dict = calculate_confidence(
    detected_count=len(positions),
    expected_count=expected_wires,
    pitches_px=gap_eval.get('pitches_px', []),
    raw_candidate_count=len(all_raw_candidates),
    seg_quality=seg_quality,
    tracking_support_score=robust_res['mean_support_score']
)

annotated_img = draw_wire_annotations(
    processed_roi,
    positions,
    gap_eval,
    orientation=orientation,
    confidence_dict=confidence_dict,
    raw_candidates=np.array(all_raw_candidates, dtype=int),
    rejected_candidates=np.array(all_rejected_candidates, dtype=int),
    debug_mode=debug_mode
)

processing_time_ms = (time.time() - start_time) * 1000.0

# --- DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Inspection Dashboard",
    "🔬 Pipeline Inspection & Debug",
    "📋 Gap Data Table",
    "🧪 Synthetic Test Suite",
    "⚡ Benchmarks & Evaluation"
])

# === TAB 1: INSPECTION DASHBOARD ===
with tab1:
    if pixels_per_mm is None:
        st.markdown(
            '<div class="calibration-banner">⚠️ <b>Calibration Required:</b> System is operating in uncalibrated pixel mode. Measurements are displayed in pixels (px). Enter <i>Pixels per mm</i> in sidebar for millimeter (mm) conversion.</div>',
            unsafe_allow_html=True
        )

    overall_status = gap_eval['overall_status']
    abnormal_count = gap_eval['abnormal_count']
    detected_count = len(positions)

    if overall_status == "ABNORMAL":
        st.markdown(f'<div class="status-abnormal">🚨 ALERT: ABNORMAL WIRE GAP DETECTED — {abnormal_count} Pair(s) Exceed Threshold ({gap_eval["threshold_label"]})</div>', unsafe_allow_html=True)
    elif detected_count != expected_wires:
        st.markdown(f'<div class="status-warning">⚠️ WARNING: WIRE COUNT MISMATCH — Detected {detected_count} of {expected_wires} Expected Wires</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-normal">✅ SYSTEM NORMAL — All Wire Gaps Within Permissible Parameters</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Overall Status", overall_status)
    with col2:
        st.metric("Detected Wires", f"{detected_count} / {expected_wires}")
    with col3:
        st.metric("Max Measured Gap", gap_eval['max_gap_display'])
    with col4:
        st.metric("Spacing Consistency", f"{confidence_dict['spacing_consistency_pct']}%")
    with col5:
        st.metric("Confidence Rating", confidence_dict['confidence_rating'])

    st.markdown("---")

    # Display Mode A vs Mode B Info
    if "Mode A" in processing_mode:
        st.info("ℹ️ **Mode A (Wire Detection Verification)**: Verifying 48-wire candidate identification and tracking support.")
    else:
        st.info("ℹ️ **Mode B (Measurement & Gap Analysis)**: Evaluating adjacent pitch, physical empty gap, and abnormality thresholds.")

    st.subheader(f"Annotated Inspection Visual Overlay ({image_title})")
    st_image_compat(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

    if abnormal_count > 0:
        st.error(f"### 🚨 Flagged Abnormal Wire Pairs ({abnormal_count})")
        ab_df = pd.DataFrame(gap_eval['abnormal_pairs'])
        st.table(ab_df[['pair_id', 'pos1_px', 'pos2_px', 'pitch_display', 'physical_gap_display']])

# === TAB 2: PIPELINE INSPECTION & DEBUG ===
with tab2:
    st.subheader("Step-by-Step Computer Vision Processing Pipeline")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### 1. Input Image & ROI Extraction")
        st_image_compat(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption=f"Original Input ({input_image.shape[1]}x{input_image.shape[0]})")
        st.info(f"Tight Wire-Bundle ROI: {roi_box} | Orientation: **{orientation}**")
        
    with col_p2:
        st.markdown("#### 2. Cropped & Preprocessed ROI")
        st_image_compat(cv2.cvtColor(processed_roi, cv2.COLOR_BGR2RGB), caption=f"Cropped ROI ({processed_roi.shape[1]}x{processed_roi.shape[0]})")
        st.info(f"Representation Channel: **{selected_mode}** | Polarity: **{matched_polarity}** | Deskew: **{deskew_angle:.2f}°**")

    st.markdown("---")

    col_p3, col_p4 = st.columns(2)
    with col_p3:
        st.markdown("#### 3. Feature Channel (Selected Representation)")
        st_image_compat(feature_channel, caption=f"Feature Channel ({selected_mode} - Polarity: {matched_polarity})")
        
    with col_p4:
        st.markdown("#### 4. Binary Segmented Mask")
        st_image_compat(binary_mask, caption="Otsu Binarized Mask")

    st.markdown("---")
    st.markdown("#### 5. Local Multi-Strip 1D Profiles & Tracking Diagnostics")
    
    st.info(f"**TRACKING DIAGNOSTICS**: Strips: **5** | Ref Pitch: **{ref_pitch:.1f} px** | Tracking Support Score: **{robust_res['mean_support_score']*100:.1f}%** | Composite Quality Score: **{confidence_dict['composite_score']}**")

    fig = plot_1d_projection_profile(
        profiles[2]['profile'],
        positions,
        orientation=orientation,
        raw_candidates=all_raw_candidates,
        rejected_candidates=all_rejected_candidates,
        estimated_pitch=ref_pitch,
        debug_mode=debug_mode
    )
    st.pyplot(fig)

# === TAB 3: MEASUREMENT LOG ===
with tab3:
    st.subheader("Complete Adjacent Wire Gap & Pitch Measurement Log")
    
    if len(gap_eval['pairs_data']) > 0:
        df_gaps = pd.DataFrame(gap_eval['pairs_data'])
        
        st_dataframe_compat(
            df_gaps[['pair_id', 'pos1_px', 'pos2_px', 'pitch_px', 'pitch_display', 'physical_gap_display', 'relative_dev_pct', 'threshold_display', 'status']]
        )
        
        csv_bytes = df_gaps.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Measurement Log (CSV)",
            data=csv_bytes,
            file_name="wire_gap_measurements.csv",
            mime="text/csv"
        )
    else:
        st.warning("Insufficient wires detected to generate gap log.")

# === TAB 4: SYNTHETIC TEST SUITE ===
with tab4:
    st.subheader("Synthetic Demonstration Suite — 8 Edge-Case Scenarios")
    st.markdown("Select a scenario from the sidebar to evaluate edge cases:")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.success("✅ **Scenario 1: Normal (48 Wires)**\n\nUniform pitch -> 48 Wires, Status NORMAL.")
        st.error("🚨 **Scenario 2: Artificial Gap**\n\n1 Displaced wire -> Status ABNORMAL.")
        st.warning("⚠️ **Scenario 3: Missing Wire**\n\n47 wires -> WARNING: Count Mismatch.")
    with col_s2:
        st.info("ℹ️ **Scenario 4: Illumination Noise**\n\nSensor noise -> Maintains stable wire detection.")
        st.info("ℹ️ **Scenario 5: Curved Wires**\n\nSinusoidal curvature -> Multi-strip tracking succeeds.")
        st.error("🚨 **Scenario 6: Displaced Local Wire**\n\nLocal portion shift -> Local abnormal gap identified.")
    with col_s3:
        st.info("ℹ️ **Scenario 7: Reflection Artifact**\n\nBright glare -> Monotonic tracking rejects false glare.")
        st.info("ℹ️ **Scenario 8: Partial Obscuration**\n\nShadow block -> Maintains tracking support.")

# === TAB 5: BENCHMARKS & EVALUATION ===
with tab5:
    st.subheader("Real-Time Performance & System Latency Metrics")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.metric("Frame Processing Time", f"{processing_time_ms:.1f} ms")
    with col_b2:
        estimated_fps = 1000.0 / max(1.0, processing_time_ms)
        st.metric("Estimated Max FPS", f"{estimated_fps:.1f} FPS")
    with col_b3:
        st.metric("Target Line Speed", "400 - 450 m/min")

    st.markdown("---")
    st.markdown("### Prototype Target Metrics & System Evaluation")
    st.markdown(r"""
    - **Stage 2 ROI Localization Bounding Box Recall**: $\ge 95\%$
    - **Wire Detection Precision**: $\ge 95\%$
    - **Wire Detection Recall**: $\ge 95\%$
    - **Monotonic Tracking Support Score**: $\ge 90\%$
    - **Mean Wire Center Error**: $< 1.0\text{ px}$
    """)

# Summary Footer
st.markdown("---")
st.caption("Real-Time-Capable Wire Gap Detection System | Computer Vision Prototype | Python 3, OpenCV, SciPy, Streamlit")
