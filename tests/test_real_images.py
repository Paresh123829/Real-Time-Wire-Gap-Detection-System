import sys, os, time, json
import cv2
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.roi import detect_backlight_roi, crop_roi
from src.image_preprocessing import adaptive_preprocess_pipeline
from src.strip_measurement import extract_local_strips, compute_strip_profiles
from src.wire_detection import detect_orientation, detect_wire_positions, validate_wire_count
from src.wire_tracking import track_wires_across_strips, compute_robust_wire_positions
from src.confidence import calculate_confidence
from src.gap_measurement import evaluate_gaps

def run_real_images_benchmark():
    gt_file = os.path.join(root_dir, 'ground_truth', 'gt_dataset.json')
    if not os.path.exists(gt_file):
        print("Ground truth file not found.")
        return

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    results = []
    print("=" * 80)
    print("REAL-IMAGE GROUND-TRUTH BENCHMARK REPORT")
    print("=" * 80)

    for img_name, info in gt_data.items():
        img_path = os.path.join(root_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        t0 = time.time()

        orientation = info.get('orientation') or detect_orientation(img)
        roi_box = detect_backlight_roi(img, orientation=orientation, trim_rails=True)
        roi = crop_roi(img, roi_box)

        prep = adaptive_preprocess_pipeline(roi, orientation=orientation, channel_mode='AUTO')
        ch = prep['feature_channel']

        # Multi-strip measurement
        strips = extract_local_strips(roi, orientation=orientation)
        profiles = compute_strip_profiles(strips, orientation=orientation, channel=ch)

        strip_detections = []
        all_raw_count = 0
        for p_info in profiles:
            det = detect_wire_positions(p_info['profile'], detection_mode='valleys')
            strip_detections.append(det)
            all_raw_count += len(det['raw_candidates'])

        ref_pitch = strip_detections[0]['estimated_pitch'] if len(strip_detections) > 0 else 15.0

        # Monotonic ordered cross-strip tracking
        track_res = track_wires_across_strips(strip_detections, reference_pitch=ref_pitch)
        robust_res = compute_robust_wire_positions(track_res['trajectories'])

        positions = robust_res['positions']
        dt_ms = (time.time() - t0) * 1000.0

        detected_cnt = len(positions)
        exp_cnt = info['expected_wires']

        eval_res = evaluate_gaps(positions, demo_mode=True)
        conf = calculate_confidence(
            detected_count=detected_cnt,
            expected_count=exp_cnt,
            pitches_px=eval_res.get('pitches_px', []),
            raw_candidate_count=all_raw_count,
            seg_quality=prep['segmentation_quality'],
            tracking_support_score=robust_res['mean_support_score']
        )

        precision = min(1.0, float(detected_cnt) / float(exp_cnt)) if exp_cnt > 0 else 1.0
        recall = float(detected_cnt) / float(exp_cnt) if exp_cnt > 0 else 0.0

        results.append({
            'Image': img_name,
            'Orientation': orientation,
            'Detected/Expected': f"{detected_cnt}/{exp_cnt}",
            'Precision': f"{precision*100:.1f}%",
            'Recall': f"{recall*100:.1f}%",
            'Consistency': f"{conf['spacing_consistency_pct']}%",
            'Confidence': conf['confidence_rating'],
            'Time (ms)': f"{dt_ms:.1f}",
            'Status': eval_res['overall_status']
        })

        print(f"[{img_name}] Detected: {detected_cnt}/{exp_cnt} | Conf: {conf['confidence_rating']} | Time: {dt_ms:.1f}ms | Status: {eval_res['overall_status']}")

    print("-" * 80)
    print(f"Benchmark completed on {len(results)} real images.")

if __name__ == '__main__':
    run_real_images_benchmark()
