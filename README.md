# Real-Time Wire Gap Detection System (Computer Vision Prototype)

An industrial-grade computer vision prototype for automatic wire belt inspection, gap measurement, and real-time abnormality detection using classical computer vision (Python, OpenCV, NumPy, SciPy, Streamlit).

---

## 📌 System Context & Objectives

The system inspects a continuous industrial wire belt consisting of **48 parallel wires** (approx. **0.9 mm diameter**). It automatically detects wire positions, calculates adjacent wire gaps, evaluates abnormality thresholds, and alerts operators to physical defects caused by wire displacement, mechanical vibration, guide wear, or wire breakage.

### Core Capabilities
1. **Automated ROI Detection**: Isolates the wire belt region from machine backgrounds and red LED backlighting.
2. **Orientation Detection**: Automatically detects whether wires are arranged **Horizontally** or **Vertically**.
3. **Adaptive Deskewing**: Estimates tilt angle via Hough line transform and levels wires if segmentation quality is poor.
4. **1D Intensity Projection Profiling**: Integrates intensity along wire length to yield a 1D profile.
5. **Sub-Pixel Peak/Valley Finding**: Uses `scipy.signal.find_peaks` to identify wire centers ($x_1..x_{48}$ or $y_1..y_{48}$).
6. **48 Wire Validation & Warning System**: Checks detected wire count against expected target ($N=48$).
7. **Pixel-to-Millimeter Calibration**: Starts in an uncalibrated mode (`pixels_per_mm = None`) with explicit warnings, supported by an interactive calibration tool.
8. **Detection Confidence Engine**: Evaluates system reliability (`HIGH`, `MEDIUM`, `LOW`) based on spacing consistency %, wire count match, and peak prominence.
9. **Visual Overlay & Alert Banners**: Draws wire centerlines, numbers wires 1–48, draws green/red gap indicators, and displays alert banners.
10. **Interactive Streamlit Dashboard**: Provides tabbed views for inspection, step-by-step pipeline inspection, measurement data table with CSV export, synthetic edge-case demo suite, and FPS benchmarks.

---

## 📐 System Architecture

```text
                    INPUT
                      │
        ┌─────────────┼──────────────┐
        │             │              │
      Image         Camera        Synthetic
        │             │              │
        └─────────────┼──────────────┘
                      ↓
                 ROI Detection
                      ↓
               Orientation Detection
                      ↓
             ┌──────────────────┐
             │  Preprocessing   │
             │                  │
             │ R-G / R / Gray   │
             │ Blur / Contrast  │
             │ Threshold        │
             └────────┬─────────┘
                      ↓
              Segmentation Quality
                   Check
                      │
               ┌──────┴──────┐
               │             │
             GOOD          POOR
               │             │
               │          Deskew /
               │          alternative
               │          preprocessing
               │             │
               └──────┬──────┘
                      ↓
                Projection Profile
                      ↓
                 Peak Detection
                      ↓
              Wire Position Detection
                      ↓
               Wire Count Validation
                      │
             ┌────────┴─────────┐
             │                  │
          48 wires          ≠ 48 wires
             │                  │
             │              WARNING: Count Mismatch
             ↓
          Gap Calculation
             ↓
       Pixel → mm Calibration Check
             │
       ┌─────┴─────┐
       │           │
   Calibrated  Uncalibrated (px only)
       └─────┬─────┘
             ↓
      Threshold Evaluation
             ↓
       ┌─────┴─────┐
       │           │
     NORMAL     ABNORMAL
       │           │
       ↓           ↓
    GREEN       RED ALERT
                   │
                   ↓
          Gap + Wire Pair + Location
                   │
                   ↓
              Dashboard
```

---

## 📁 Repository Directory Structure

```text
d:/SEM5/ML/CP/
├── app.py                      # Streamlit Dashboard Interface
├── config.py                   # System thresholds & default parameters
├── requirements.txt            # Python dependencies
├── image1.jpeg - image7.jpeg   # 7 Industrial test images
├── src/
│   ├── __init__.py
│   ├── roi.py                  # Auto-ROI detection & manual cropping
│   ├── image_preprocessing.py  # Feature channel extraction, binarization, adaptive deskewing
│   ├── wire_detection.py       # Orientation detection, 1D projection profile, peak finding
│   ├── confidence.py           # Confidence engine (HIGH/MED/LOW, consistency %)
│   ├── gap_measurement.py      # Adjacent gap math, calibration evaluation, thresholding
│   ├── calibration.py          # Calibration helpers (px/mm conversion)
│   ├── visualization.py        # OpenCV visual overlays, alert banners, Matplotlib plots
│   ├── camera.py               # Live camera stream & latency benchmarking
│   └── synthetic.py            # Controlled synthetic wire-belt generator
└── tests/
    └── test_detection.py       # Pytest unit test suite for 5 core scenarios
```

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites
Ensure Python 3.8+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
To run unit tests verifying all 5 test scenarios (Normal Spacing, Abnormal Gap, Missing Wire, Noise Stability, Slight Shift):
```bash
python tests/test_detection.py
```

### 4. Launch the Streamlit Inspection Dashboard
```bash
streamlit run app.py
```
The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## ⚙️ Key Concepts & Design Rules

1. **Uncalibrated Default State**:
   - `DEFAULT_PIXELS_PER_MM = None`.
   - Measurements display in pixels (`px`) until calibration is entered.
   - Prevents unvalidated pixel-to-millimeter claims.
2. **Demo Threshold Disclaimer**:
   - `DEFAULT_DEMO_MAX_GAP_MM = 2.0` is explicitly labeled as `Demo threshold — not an engineering specification`.
3. **Parameter Independence**:
   - `Wire Diameter = 0.9 mm` (physical wire thickness).
   - `Expected Wire Pitch = Configurable`.
   - `Maximum Permissible Gap = Configurable`.
   - Emphasizes: `wire diameter ≠ spacing between wire centers`.
4. **Synthetic Demo Mode**:
   - Allows instant controlled demonstration of:
     - `Normal` (48 wires, uniform pitch $\to$ NORMAL / GREEN).
     - `Abnormal Gap` (wire displaced $\to$ RED ALERT + Wire Pair ID & Location).
     - `Missing Wire` (47 wires $\to$ WARNING: Count Mismatch).
     - `Noisy Illumination` (Tests noise stability).
     - `Slight Shift` (Below threshold $\to$ NORMAL).

---

## 🏭 Production Scope & Future Hardware Considerations

For production line deployment:
- **Global-Shutter Industrial Camera**: Prevents motion blur at high belt speeds.
- **Triggered Strobe Backlighting**: Synchronizes LED lighting pulses with frame capture.
- **PLC & Digital I/O Integration**: Immediate machine stop signal triggered on RED ALERT status.
