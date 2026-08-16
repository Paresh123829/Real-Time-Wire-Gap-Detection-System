import sys, os
import cv2
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from src.wire_detection import compute_projection_profile, detect_wire_positions

POLARITY_MAP = config.REPRESENTATION_POLARITY

def apply_directional_blackhat(roi, orientation='VERTICAL'):
    """
    Applies directional Black-Hat morphological transform matched to wire orientation:
    - Vertical wires -> Vertical kernel (1, 9)
    - Horizontal wires -> Horizontal kernel (9, 1)
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    if orientation == 'VERTICAL':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def evaluate_best_channel(roi, orientation='VERTICAL', expected_wires=48):
    """
    Evaluates candidate representation channels (R-G, RED, GRAY, INVERT_GRAY, BLACK_HAT)
    using matched feature polarity (Peak vs Valley) and selects representation maximizing
    actual wire-detection quality score (candidate count plausibility + spacing consistency).
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    r_channel = roi[:, :, 2] if len(roi.shape) == 3 else roi
    g_channel = roi[:, :, 1] if len(roi.shape) == 3 else roi
    
    r_g_diff = cv2.subtract(r_channel, g_channel)
    invert_gray = cv2.bitwise_not(gray)
    blackhat = apply_directional_blackhat(roi, orientation=orientation)
    
    channels = {
        'R-G': r_g_diff,
        'RED': r_channel,
        'GRAY': gray,
        'INVERT_GRAY': invert_gray,
        'BLACK_HAT': blackhat
    }
    
    channel_scores = {}
    
    for ch_name, ch_img in channels.items():
        pol = POLARITY_MAP.get(ch_name, 'valleys')
        p_profile = compute_projection_profile(ch_img, orientation=orientation)
        
        det = detect_wire_positions(p_profile, detection_mode=pol)
        positions = det['positions']
        cnt = len(positions)
        
        # Count plausibility score
        diff = abs(cnt - expected_wires)
        count_score = max(0.0, 1.0 - (diff / float(expected_wires)))
        
        # Spacing consistency score
        if cnt > 1:
            pitches = np.diff(np.sort(positions))
            cv = float(np.std(pitches)) / (float(np.mean(pitches)) + 1e-5)
            spacing_score = float(1.0 / (1.0 + 5.0 * cv))
        else:
            spacing_score = 0.0
            
        composite_score = 0.5 * count_score + 0.5 * spacing_score
        channel_scores[ch_name] = {
            'score': composite_score,
            'channel_img': ch_img,
            'polarity': pol,
            'detected_count': cnt
        }

    best_mode = max(channel_scores.keys(), key=lambda k: channel_scores[k]['score'])
    best_info = channel_scores[best_mode]
    
    return {
        'best_mode': best_mode,
        'best_channel': best_info['channel_img'],
        'polarity': best_info['polarity'],
        'quality_score': round(best_info['score'], 3),
        'all_scores': channel_scores
    }

def adaptive_preprocess_pipeline(roi, orientation='VERTICAL', channel_mode='AUTO', force_deskew=False):
    """
    Adaptive Preprocessing Pipeline:
    1. Deskewing / alignment (if forced or detected).
    2. Representation selection (R-G, RED, GRAY, INVERT_GRAY, BLACK_HAT) with matched polarity.
    3. Otsu binarization & segmentation quality metric.
    """
    processed_roi = roi.copy()
    deskew_angle = 0.0

    if force_deskew:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            angles = []
            flat_lines = lines.reshape(-1, 4)
            for line in flat_lines:
                x1, y1, x2, y2 = line
                if orientation == 'VERTICAL':
                    ang = np.degrees(np.arctan2(x2 - x1, y2 - y1))
                else:
                    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(ang) < 15.0:
                    angles.append(ang)
                    
            if len(angles) > 0:
                deskew_angle = float(np.median(angles))
                if abs(deskew_angle) > 0.3:
                    h, w = roi.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), deskew_angle, 1.0)
                    processed_roi = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if channel_mode == 'AUTO':
        eval_res = evaluate_best_channel(processed_roi, orientation=orientation)
        selected_mode = eval_res['best_mode']
        feature_channel = eval_res['best_channel']
        polarity = eval_res['polarity']
    else:
        selected_mode = channel_mode
        polarity = POLARITY_MAP.get(selected_mode, 'valleys')
        if selected_mode == 'BLACK_HAT':
            feature_channel = apply_directional_blackhat(processed_roi, orientation=orientation)
        elif selected_mode == 'R-G':
            feature_channel = cv2.subtract(processed_roi[:, :, 2], processed_roi[:, :, 1])
        elif selected_mode == 'RED':
            feature_channel = processed_roi[:, :, 2]
        elif selected_mode == 'INVERT_GRAY':
            gray = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2GRAY) if len(processed_roi.shape) == 3 else processed_roi
            feature_channel = cv2.bitwise_not(gray)
        else: # GRAY
            feature_channel = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2GRAY) if len(processed_roi.shape) == 3 else processed_roi

    # Otsu binarization
    blurred_ch = cv2.GaussianBlur(feature_channel, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Segmentation quality metric
    fg_mean = float(np.mean(feature_channel[binary_mask > 0])) if np.sum(binary_mask > 0) > 0 else 0.0
    bg_mean = float(np.mean(feature_channel[binary_mask == 0])) if np.sum(binary_mask == 0) > 0 else 0.0
    contrast_ratio = float(abs(fg_mean - bg_mean) / (bg_mean + 1e-5))

    return {
        'processed_roi': processed_roi,
        'feature_channel': feature_channel,
        'binary': binary_mask,
        'selected_mode': selected_mode,
        'polarity': polarity,
        'deskew_angle': deskew_angle,
        'segmentation_quality': round(contrast_ratio, 2)
    }
