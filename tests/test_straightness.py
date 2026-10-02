import sys, os
import cv2
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.synthetic import generate_synthetic_wire_belt
from src.roi import detect_backlight_roi, crop_roi
from src.image_preprocessing import adaptive_preprocess_pipeline
from src.strip_measurement import extract_local_strips, compute_strip_profiles
from src.wire_detection import detect_orientation, detect_wire_positions
from src.wire_tracking import track_wires_across_strips
from src.straightness import evaluate_wire_straightness, fit_straight_line

def run_straightness_pipeline(img, pixels_per_mm=None, max_straightness_tol=None):
    orientation = detect_orientation(img)
    roi_box = detect_backlight_roi(img, orientation=orientation, trim_rails=True)
    roi = crop_roi(img, roi_box)

    prep = adaptive_preprocess_pipeline(roi, orientation=orientation, channel_mode='AUTO')
    ch = prep['feature_channel']

    strips = extract_local_strips(roi, orientation=orientation)
    profiles = compute_strip_profiles(strips, orientation=orientation, channel=ch)

    strip_detections = []
    for p_info in profiles:
        det = detect_wire_positions(p_info['profile'], detection_mode='valleys')
        strip_detections.append(det)

    ref_pitch = strip_detections[0]['estimated_pitch'] if len(strip_detections) > 0 else 15.0
    track_res = track_wires_across_strips(strip_detections, reference_pitch=ref_pitch)

    straightness_res = evaluate_wire_straightness(
        track_res['trajectories'],
        strips,
        orientation=orientation,
        pixels_per_mm=pixels_per_mm,
        max_straightness_tol=max_straightness_tol
    )

    return straightness_res

def test_straight_line_fitting():
    # Perfect horizontal line
    u = np.array([100, 200, 300, 400, 500])
    v = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
    res = fit_straight_line(u, v)
    assert res['max_dev_px'] == 0.0, f"Expected 0.0 dev, got {res['max_dev_px']}"
    assert res['peak_to_valley_px'] == 0.0

    # Line with slight slope + 3px bow at center
    v_bowed = np.array([50.0, 51.0, 55.0, 53.0, 54.0])
    res_bowed = fit_straight_line(u, v_bowed)
    assert res_bowed['max_dev_px'] > 1.0, "Expected deflection > 1.0 px"

def test_normal_wires_are_straight():
    img, _ = generate_synthetic_wire_belt(scenario='normal', num_wires=48)
    res = run_straightness_pipeline(img, max_straightness_tol=2.0)
    assert res['overall_status'] == 'NORMAL', f"Expected NORMAL, got {res['overall_status']}"
    assert res['bent_wire_count'] == 0, f"Expected 0 bent wires, got {res['bent_wire_count']}"
    assert res['max_belt_dev_px'] < 2.0, f"Deviation unexpectedly high: {res['max_belt_dev_px']} px"

def test_curved_wires_detected():
    img, _ = generate_synthetic_wire_belt(scenario='curved_wires', num_wires=48)
    res = run_straightness_pipeline(img, max_straightness_tol=2.0)
    assert res['overall_status'] == 'ABNORMAL', f"Expected ABNORMAL for curved wires, got {res['overall_status']}"
    assert res['bent_wire_count'] > 10, f"Expected many bent wires, got {res['bent_wire_count']}"
    assert res['max_belt_dev_px'] > 2.0

def test_straightness_trajectory_evaluation():
    # Test controlled multi-strip trajectories with 1 bent wire and multiple straight wires
    mock_strips = [{'center_coord': c} for c in [100, 200, 300, 400, 500]]
    mock_trajectories = []

    # 5 Straight wires
    for w in range(1, 6):
        base_pos = w * 15.0
        mock_trajectories.append({
            'wire_id': w,
            'strip_positions': {s: base_pos for s in range(5)}
        })

    # Wire 6 is bent with a 4.0 px deflection at center strip
    mock_trajectories.append({
        'wire_id': 6,
        'strip_positions': {0: 90.0, 1: 91.5, 2: 94.0, 3: 91.5, 4: 90.0}
    })

    res = evaluate_wire_straightness(mock_trajectories, mock_strips, orientation='VERTICAL', max_straightness_tol=2.0)
    assert res['overall_status'] == 'ABNORMAL', f"Expected ABNORMAL, got {res['overall_status']}"
    assert res['bent_wire_count'] == 1, f"Expected 1 bent wire, got {res['bent_wire_count']}"
    assert res['bent_wires'][0]['wire_id'] == 6, f"Expected wire 6 bent, got {res['bent_wires'][0]['wire_id']}"
    assert res['bent_wires'][0]['max_dev_px'] > 2.0, "Expected deflection > 2.0 px"

def test_calibrated_straightness():
    img, _ = generate_synthetic_wire_belt(scenario='normal', num_wires=48)
    # 10 px/mm scale
    res = run_straightness_pipeline(img, pixels_per_mm=10.0, max_straightness_tol=0.25)
    assert res['is_calibrated'] is True
    assert res['max_belt_dev_mm'] is not None
    assert res['max_belt_dev_mm'] < 0.25
    assert "mm" in res['max_belt_dev_display']

if __name__ == '__main__':
    all_tests = [
        ('Straight Line Fitting Math', test_straight_line_fitting),
        ('Normal Wires Straightness (All Straight)', test_normal_wires_are_straight),
        ('Curved Wires Defect Detection', test_curved_wires_detected),
        ('Multi-Strip Trajectory Straightness Evaluation', test_straightness_trajectory_evaluation),
        ('Calibrated Metric Straightness (mm)', test_calibrated_straightness),
    ]
    passed = 0
    for name, test_func in all_tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    print(f"\nStraightness Test Results: {passed}/{len(all_tests)} passed.")
    if passed < len(all_tests):
        sys.exit(1)
