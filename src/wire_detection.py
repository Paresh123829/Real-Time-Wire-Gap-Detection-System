import sys, os
import cv2
import numpy as np
import scipy.signal as signal

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config

EXPECTED_WIRES = config.EXPECTED_WIRES
MIN_DISTANCE_FACTOR = config.MIN_DISTANCE_FACTOR

def detect_orientation(img):
    """
    Automatically detects wire orientation (HORIZONTAL vs VERTICAL)
    by comparing directional Sobel gradients in central region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    
    cy_start, cy_end = int(h * 0.2), int(h * 0.8)
    cx_start, cx_end = int(w * 0.2), int(w * 0.8)
    crop = gray[cy_start:cy_end, cx_start:cx_end]
    
    grad_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
    
    energy_x = np.mean(np.abs(grad_x))  # High X gradient -> Vertical wires
    energy_y = np.mean(np.abs(grad_y))  # High Y gradient -> Horizontal wires
    
    if energy_x > energy_y:
        return 'VERTICAL'
    else:
        return 'HORIZONTAL'

def compute_projection_profile(channel, orientation='HORIZONTAL', remove_background=True):
    """
    Generates 1D intensity projection profile along measurement axis.
    Applies high-pass baseline removal to eliminate background lighting gradients.
    """
    if orientation == 'HORIZONTAL':
        profile = np.mean(channel.astype(np.float32), axis=1).ravel()
    else:
        profile = np.mean(channel.astype(np.float32), axis=0).ravel()

    if remove_background and len(profile) > 30:
        ksize = max(31, (len(profile) // 4) | 1)
        if ksize % 2 == 0:
            ksize += 1
        bg = cv2.GaussianBlur(profile[:, None].astype(np.float32), (ksize, 1), 0).ravel()
        hp = profile - bg
        
        hp_min, hp_max = hp.min(), hp.max()
        if (hp_max - hp_min) > 1e-3:
            norm_profile = (hp - hp_min) / (hp_max - hp_min) * 255.0
            return norm_profile.astype(np.float32)
            
    return profile.astype(np.float32)

def evaluate_strip_quality(positions, expected_count=EXPECTED_WIRES):
    """
    Evaluates strip detection quality score and classifies into HIGH, MEDIUM, or LOW QUALITY.
    - Candidate count plausibility
    - Spacing consistency (CV)
    """
    cnt = len(positions)
    if cnt == 0:
        return 'LOW QUALITY', 0.0

    # Count score
    diff = abs(cnt - expected_count)
    count_score = max(0.0, 1.0 - (diff / float(expected_count)))

    # Spacing score
    if cnt > 1:
        pitches = np.diff(np.sort(positions))
        mean_p = float(np.mean(pitches))
        std_p = float(np.std(pitches))
        cv = std_p / (mean_p + 1e-5)
        spacing_score = float(1.0 / (1.0 + 5.0 * cv))
    else:
        cv = 1.0
        spacing_score = 0.0

    composite = 0.5 * count_score + 0.5 * spacing_score

    if composite >= 0.75 and cv < 0.25:
        category = 'HIGH QUALITY'
    elif composite >= 0.45 and cv < 0.40:
        category = 'MEDIUM QUALITY'
    else:
        category = 'LOW QUALITY'

    return category, composite

def estimate_coarse_pitch(profile, detection_mode='valleys'):
    """
    Computes initial coarse reference pitch from raw signal extrema.
    Used ONLY for setting coarse min_distance control parameter.
    """
    profile = profile.ravel()
    n = len(profile)
    if n < 20:
        return 15.0

    target = -profile if detection_mode == 'valleys' else profile
    p_std = float(np.std(target))
    prom = max(0.2, p_std * 0.08)

    raw_peaks, _ = signal.find_peaks(target, prominence=prom, distance=4)
    if len(raw_peaks) > 1:
        pitches = np.diff(raw_peaks)
        med_p = float(np.median(pitches))
        if 4.0 <= med_p <= 100.0:
            return round(med_p, 2)

    return float(n) / float(EXPECTED_WIRES)

def refine_subpixel_center(profile, integer_x):
    """
    Parabolic 3-Point Sub-Pixel Refinement with Safe Zero-Denominator Safeguard:
    x_sub = x_int + (y_right - y_left) / (2 * (2 * y_center - y_left - y_right))
    """
    n = len(profile)
    x = int(round(integer_x))
    
    if x <= 0 or x >= (n - 1):
        return float(x)
        
    y_center = float(profile[x])
    y_left = float(profile[x - 1])
    y_right = float(profile[x + 1])
    
    denom = 2.0 * (2.0 * y_center - y_left - y_right)
    
    if abs(denom) < 1e-5:
        return float(x)
        
    delta = (y_right - y_left) / denom
    delta = max(-0.5, min(0.5, delta))
    
    return float(x + delta)

def detect_wire_positions(
    profile,
    detection_mode='valleys',
    smoothing_ksize=5,
    min_distance_factor=MIN_DISTANCE_FACTOR,
    manual_prominence=None,
    coarse_pitch_hint=None
):
    """
    Coarse-to-Refined Two-Stage Wire Detection with Physical-Wire Level Deduplication:
    1. Bootstrap coarse pitch P_coarse.
    2. Set refined min_distance = min_distance_factor * P_coarse.
    3. Extract raw candidate extrema matching detection_mode polarity ('peaks' vs 'valleys').
    4. Local window shoulder-contrast validation.
    5. Physical-Wire Level Candidate Deduplication (envelope ~ 0.4 * P_coarse).
    6. Sub-pixel center refinement (with safe zero-denominator guard).
    """
    profile = profile.ravel()
    n = len(profile)
    if n == 0:
        return {
            'positions': np.array([]),
            'subpixel_positions': np.array([]),
            'raw_candidates': np.array([]),
            'rejected_candidates': np.array([]),
            'estimated_pitch': 15.0,
            'min_distance': 5,
            'quality_category': 'LOW QUALITY',
            'smoothed_profile': np.array([]),
            'detection_mode': detection_mode
        }

    if smoothing_ksize > 1:
        if smoothing_ksize % 2 == 0:
            smoothing_ksize += 1
        smoothed = cv2.GaussianBlur(profile[:, None].astype(np.float32), (smoothing_ksize, 1), 0).ravel()
    else:
        smoothed = profile.astype(np.float32)

    # 1. Coarse Pitch Bootstrap
    if coarse_pitch_hint is not None and coarse_pitch_hint > 2.0:
        estimated_pitch = float(coarse_pitch_hint)
    else:
        estimated_pitch = estimate_coarse_pitch(smoothed, detection_mode=detection_mode)

    min_distance = max(4, int(round(min_distance_factor * estimated_pitch)))
    
    # Target signal matched to feature polarity
    target_signal = -smoothed if detection_mode == 'valleys' else smoothed
    
    profile_std = float(np.std(target_signal))
    prominence = manual_prominence if manual_prominence is not None else max(0.2, profile_std * 0.08)

    raw_peaks, props = signal.find_peaks(
        target_signal,
        prominence=prominence,
        distance=max(2, min_distance // 2)
    )

    prominences = props.get('prominences', np.zeros(len(raw_peaks)))

    candidates = []
    rejected = []
    half_w = max(2, int(estimated_pitch * 0.4))
    
    for idx, p in enumerate(raw_peaks):
        prom = prominences[idx]
        if p < half_w or p >= (n - half_w):
            rejected.append((p, "Border Boundary"))
            continue
            
        center_val = smoothed[p]
        left_shoulder = np.mean(smoothed[max(0, p - half_w): max(0, p - half_w // 2)])
        right_shoulder = np.mean(smoothed[min(n, p + half_w // 2): min(n, p + half_w)])
        
        if detection_mode == 'valleys':
            is_valid_contrast = (center_val < left_shoulder) and (center_val < right_shoulder)
        else:
            is_valid_contrast = (center_val > left_shoulder) and (center_val > right_shoulder)
            
        if is_valid_contrast:
            candidates.append({
                'pos': p,
                'prominence': prom,
                'center_val': center_val,
                'score': prom
            })
        else:
            rejected.append((p, "Poor Contrast"))

    candidates = sorted(candidates, key=lambda c: c['pos'])
    
    # 5. Physical-Wire Level Candidate Deduplication
    wire_envelope_px = max(3, int(round(estimated_pitch * 0.42)))
    final_wire_centers = []
    i = 0
    while i < len(candidates):
        curr = candidates[i]
        duplicate_group = [curr]
        j = i + 1
        while j < len(candidates) and (candidates[j]['pos'] - curr['pos']) < wire_envelope_px:
            duplicate_group.append(candidates[j])
            j += 1
            
        if len(duplicate_group) == 1:
            final_wire_centers.append(curr['pos'])
            i += 1
        else:
            best = max(duplicate_group, key=lambda c: c['score'])
            final_wire_centers.append(best['pos'])
            for dup in duplicate_group:
                if dup['pos'] != best['pos']:
                    rejected.append((dup['pos'], "Duplicate Wire Center"))
            i = j

    int_positions = np.array(final_wire_centers, dtype=int)
    
    # Sub-pixel refinement applied AFTER candidate deduplication
    subpixel_positions = np.array([refine_subpixel_center(smoothed, x) for x in int_positions], dtype=np.float32)
    rejected_positions = np.array([r[0] for r in rejected], dtype=int) if len(rejected) > 0 else np.array([], dtype=int)

    quality_cat, q_score = evaluate_strip_quality(int_positions)

    return {
        'positions': int_positions,
        'subpixel_positions': subpixel_positions,
        'raw_candidates': np.array(raw_peaks, dtype=int),
        'rejected_candidates': rejected_positions,
        'rejected_details': rejected,
        'prominences': prominences,
        'estimated_pitch': estimated_pitch,
        'min_distance': min_distance,
        'quality_category': quality_cat,
        'quality_score': q_score,
        'smoothed_profile': smoothed,
        'detection_mode': detection_mode
    }

def validate_wire_count(detected_count, expected_count=EXPECTED_WIRES):
    """
    Validates detected wire count against expected target (default 48).
    """
    is_match = (detected_count == expected_count)
    if is_match:
        warning = None
    elif detected_count < expected_count:
        warning = f"WARNING: Expected {expected_count} wires, detected {detected_count} wires. Possible missing wire or obscured region."
    else:
        warning = f"WARNING: Expected {expected_count} wires, detected {detected_count} wires. Possible duplicate candidate artifact."
        
    return {
        'is_match': is_match,
        'detected_count': detected_count,
        'expected_count': expected_count,
        'warning_message': warning
    }
