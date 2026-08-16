import numpy as np
import config

WEIGHTS = config.CONFIDENCE_WEIGHTS

def calculate_confidence(
    detected_count,
    expected_count=config.EXPECTED_WIRES,
    pitches_px=None,
    raw_candidate_count=None,
    seg_quality=1.0,
    tracking_support_score=1.0
):
    """
    Normalized Multi-Factor Confidence Engine (Independent of Abnormality Status):
    Computes measurement quality (0.0 to 1.0) across 6 components:
    1. Count Score (S_count)
    2. Spacing Consistency Score (S_spacing)
    3. Peak Quality Score (S_peak)
    4. Segmentation Contrast Score (S_segmentation)
    5. Tracking Support Score (S_tracking)
    6. Multi-Strip Agreement Score (S_agreement)
    """
    # 1. Count Score
    diff_c = abs(detected_count - expected_count)
    s_count = max(0.0, 1.0 - (diff_c / float(expected_count)))

    # 2. Spacing Consistency Score
    if pitches_px is not None and len(pitches_px) > 1:
        mean_p = float(np.mean(pitches_px))
        std_p = float(np.std(pitches_px))
        cv = std_p / (mean_p + 1e-5)
        s_spacing = max(0.0, 1.0 - (cv / 0.35))
    else:
        cv = 1.0
        s_spacing = 0.0

    # 3. Peak Quality Score
    if raw_candidate_count is not None and raw_candidate_count > 0:
        ratio = float(detected_count) / float(raw_candidate_count)
        s_peak = min(1.0, max(0.0, ratio))
    else:
        s_peak = 0.8

    # 4. Segmentation Contrast Score
    s_segmentation = min(1.0, max(0.0, float(seg_quality) / 2.0))

    # 5. Tracking Support Score
    s_tracking = min(1.0, max(0.0, float(tracking_support_score)))

    # 6. Agreement Score
    s_agreement = 1.0 if abs(detected_count - expected_count) <= 2 else 0.5

    composite_score = (
        WEIGHTS['count'] * s_count +
        WEIGHTS['spacing'] * s_spacing +
        WEIGHTS['peak'] * s_peak +
        WEIGHTS['segmentation'] * s_segmentation +
        WEIGHTS['tracking'] * s_tracking +
        WEIGHTS['agreement'] * s_agreement
    )

    composite_score = float(np.clip(composite_score, 0.0, 1.0))

    if composite_score >= 0.80:
        rating = 'HIGH'
    elif composite_score >= 0.60:
        rating = 'MEDIUM'
    else:
        rating = 'LOW'

    return {
        'composite_score': round(composite_score, 3),
        'confidence_rating': rating,
        'spacing_consistency_pct': round((1.0 - min(1.0, cv)) * 100.0, 1),
        'cv': round(cv, 3),
        'component_scores': {
            'count': round(s_count, 3),
            'spacing': round(s_spacing, 3),
            'peak': round(s_peak, 3),
            'segmentation': round(s_segmentation, 3),
            'tracking': round(s_tracking, 3),
            'agreement': round(s_agreement, 3)
        }
    }
