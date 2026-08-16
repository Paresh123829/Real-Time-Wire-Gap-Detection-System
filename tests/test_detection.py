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
from src.wire_detection import detect_orientation, detect_wire_positions, validate_wire_count
from src.wire_tracking import track_wires_across_strips, compute_robust_wire_positions
from src.confidence import calculate_confidence
from src.gap_measurement import evaluate_gaps

def run_pipeline_on_image(img, orientation_override=None, pixels_per_mm=None, max_gap_threshold=None, demo_mode=True):
    orientation = orientation_override or detect_orientation(img)
    roi_box = detect_backlight_roi(img, orientation=orientation, trim_rails=True)
    roi = crop_roi(img, roi_box)
    
    prep = adaptive_preprocess_pipeline(roi, orientation=orientation, channel_mode='AUTO')
    ch = prep['feature_channel']
    
    strips = extract_local_strips(roi, orientation=orientation)
    profiles = compute_strip_profiles(strips, orientation=orientation, channel=ch)
    
    strip_detections = []
    all_raw_count = 0
    for p_info in profiles:
        det = detect_wire_positions(p_info['profile'], detection_mode='valleys')
        strip_detections.append(det)
        all_raw_count += len(det['raw_candidates'])

    ref_pitch = strip_detections[0]['estimated_pitch'] if len(strip_detections) > 0 else 15.0
    
    track_res = track_wires_across_strips(strip_detections, reference_pitch=ref_pitch)
    robust_res = compute_robust_wire_positions(track_res['trajectories'])
    
    positions = robust_res['positions']
    
    count_val = validate_wire_count(len(positions), expected_count=48)
    eval_results = evaluate_gaps(positions, pixels_per_mm=pixels_per_mm, max_gap_user=max_gap_threshold, demo_mode=demo_mode)
    
    confidence = calculate_confidence(
        detected_count=len(positions),
        expected_count=48,
        pitches_px=eval_results.get('pitches_px', []),
        raw_candidate_count=all_raw_count,
        seg_quality=prep['segmentation_quality'],
        tracking_support_score=robust_res['mean_support_score']
    )
    
    return {
        'positions': positions,
        'count_val': count_val,
        'eval_results': eval_results,
        'confidence': confidence,
        'orientation': orientation,
        'track_res': track_res
    }

def test_scenario_1_normal_spacing():
    img, _ = generate_synthetic_wire_belt(scenario='normal', num_wires=48)
    res = run_pipeline_on_image(img)
    assert len(res['positions']) == 48, f"Expected 48 wires, got {len(res['positions'])}"
    assert res['eval_results']['overall_status'] == 'NORMAL', f"Expected NORMAL, got {res['eval_results']['overall_status']}"
    assert res['confidence']['confidence_rating'] == 'HIGH', f"Expected HIGH confidence, got {res['confidence']['confidence_rating']}"

def test_scenario_2_abnormal_gap():
    img, _ = generate_synthetic_wire_belt(scenario='abnormal_gap', num_wires=48)
    res = run_pipeline_on_image(img, demo_mode=True)
    assert res['eval_results']['overall_status'] == 'ABNORMAL', f"Expected ABNORMAL, got {res['eval_results']['overall_status']}"

def test_scenario_3_missing_wire():
    img, _ = generate_synthetic_wire_belt(scenario='missing_wire', num_wires=48)
    res = run_pipeline_on_image(img)
    assert res['count_val']['is_match'] is False, "Expected count mismatch"
    assert res['count_val']['detected_count'] < 48, f"Expected < 48 wires, got {res['count_val']['detected_count']}"

def test_scenario_4_noise_stability():
    img, _ = generate_synthetic_wire_belt(scenario='noisy_illumination', num_wires=48)
    res = run_pipeline_on_image(img)
    assert len(res['positions']) >= 45, f"Too few wires detected under noise: {len(res['positions'])}"

def test_scenario_5_curved_wires():
    img, _ = generate_synthetic_wire_belt(scenario='curved_wires', num_wires=48)
    res = run_pipeline_on_image(img)
    assert len(res['positions']) >= 45, f"Multi-strip failed on curved wires: {len(res['positions'])}"

def test_scenario_6_displaced_local_wire():
    img, _ = generate_synthetic_wire_belt(scenario='displaced_local_wire', num_wires=48)
    res = run_pipeline_on_image(img)
    assert len(res['positions']) >= 45, f"Failed on local displaced wire: {len(res['positions'])}"

def test_scenario_7_reflection_artifact():
    img, _ = generate_synthetic_wire_belt(scenario='reflection_artifact', num_wires=48)
    res = run_pipeline_on_image(img)
    assert len(res['positions']) == 48, f"Reflection created false detection: {len(res['positions'])}"

def test_scenario_8_partial_obscuration():
    img, _ = generate_synthetic_wire_belt(scenario='partial_obscuration', num_wires=48)
    res = run_pipeline_on_image(img)
    assert len(res['positions']) >= 45, f"Partial obscuration failed: {len(res['positions'])}"

if __name__ == '__main__':
    all_tests = [
        ('Scenario 1: Normal Spacing (48 Wires)', test_scenario_1_normal_spacing),
        ('Scenario 2: Artificial Abnormal Gap', test_scenario_2_abnormal_gap),
        ('Scenario 3: Missing Wire', test_scenario_3_missing_wire),
        ('Scenario 4: Noise Stability', test_scenario_4_noise_stability),
        ('Scenario 5: Curved Wires', test_scenario_5_curved_wires),
        ('Scenario 6: Displaced Local Wire', test_scenario_6_displaced_local_wire),
        ('Scenario 7: Reflection Artifact', test_scenario_7_reflection_artifact),
        ('Scenario 8: Partial Obscuration', test_scenario_8_partial_obscuration),
    ]
    passed = 0
    for name, test_func in all_tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    print(f"\nResults: {passed}/{len(all_tests)} passed.")
    if passed < len(all_tests):
        sys.exit(1)
