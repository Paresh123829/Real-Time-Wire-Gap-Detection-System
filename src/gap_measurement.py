import numpy as np
import config

DEFAULT_WIRE_DIAMETER_MM = config.DEFAULT_WIRE_DIAMETER_MM
DEFAULT_DEMO_MAX_GAP_MM = config.DEFAULT_DEMO_MAX_GAP_MM

def evaluate_gaps(
    final_wire_positions,
    pixels_per_mm=config.DEFAULT_PIXELS_PER_MM,
    wire_diameter_mm=DEFAULT_WIRE_DIAMETER_MM,
    max_gap_user=None,
    demo_mode=True
):
    """
    Evaluates adjacent wire pitch and physical empty gap AFTER tracking:
    - FINAL Pitch Recalculation: final_pitches_px = diff(final_wire_positions).
    - Uncalibrated Mode (pixels_per_mm is None):
        Reports Pitch in px, Reference Pitch in px, Relative Pitch Deviation in %.
        Physical Empty Gap = NOT AVAILABLE (Calibration required).
    - Calibrated Mode:
        P_i,mm = P_i,px / pixels_per_mm
        G_i,mm = P_i,mm - 0.90 mm
        Evaluates max permissible threshold limit.
    """
    positions = np.sort(np.array(final_wire_positions, dtype=np.float32))
    num_wires = len(positions)

    if num_wires < 2:
        return {
            'is_calibrated': False,
            'num_pairs': 0,
            'pairs_data': [],
            'pitches_px': [],
            'final_reference_pitch_px': 0.0,
            'gaps_mm': [],
            'overall_status': 'INSUFFICIENT_WIRES',
            'max_gap_val': 0.0,
            'max_gap_display': 'N/A',
            'abnormal_count': 0,
            'abnormal_pairs': [],
            'threshold_label': 'N/A'
        }

    is_calibrated = (pixels_per_mm is not None) and (pixels_per_mm > 0)
    
    # FINAL Pitch calculation AFTER tracking
    pitches_px = np.diff(positions)
    final_reference_pitch_px = float(np.median(pitches_px))

    pairs_data = []
    gaps_mm = []
    abnormal_pairs = []

    if is_calibrated:
        scale = float(pixels_per_mm)
        ref_pitch_mm = final_reference_pitch_px / scale
        threshold_val = max_gap_user if max_gap_user is not None else DEFAULT_DEMO_MAX_GAP_MM
        threshold_label = f"{threshold_val:.2f} mm"
        
        for i in range(num_wires - 1):
            p1 = float(positions[i])
            p2 = float(positions[i + 1])
            pitch_px = float(p2 - p1)
            pitch_mm = pitch_px / scale
            gap_mm = pitch_mm - wire_diameter_mm
            gaps_mm.append(gap_mm)

            rel_dev = (abs(pitch_px - final_reference_pitch_px) / final_reference_pitch_px) * 100.0
            is_abnormal = (gap_mm > threshold_val)
            status_pair = 'ABNORMAL' if is_abnormal else 'NORMAL'

            p_data = {
                'pair_id': f"W{i+1}-W{i+2}",
                'wire1': i + 1,
                'wire2': i + 2,
                'pos1_px': round(p1, 2),
                'pos2_px': round(p2, 2),
                'pitch_px': round(pitch_px, 2),
                'pitch_mm': round(pitch_mm, 3),
                'pitch_display': f"{pitch_mm:.3f} mm",
                'physical_gap_mm': round(gap_mm, 3),
                'physical_gap_display': f"{gap_mm:.3f} mm",
                'relative_dev_pct': round(rel_dev, 1),
                'threshold_display': threshold_label,
                'status': status_pair
            }
            pairs_data.append(p_data)
            if is_abnormal:
                abnormal_pairs.append(p_data)

        max_gap_val = float(np.max(gaps_mm)) if gaps_mm else 0.0
        max_gap_display = f"{max_gap_val:.3f} mm"

    else: # UNCALIBRATED MODE
        threshold_val = max_gap_user if max_gap_user is not None else (final_reference_pitch_px * 1.35)
        threshold_label = f"{threshold_val:.1f} px"

        for i in range(num_wires - 1):
            p1 = float(positions[i])
            p2 = float(positions[i + 1])
            pitch_px = float(p2 - p1)
            rel_dev = (abs(pitch_px - final_reference_pitch_px) / final_reference_pitch_px) * 100.0
            is_abnormal = (pitch_px > threshold_val)
            status_pair = 'ABNORMAL' if is_abnormal else 'NORMAL'

            p_data = {
                'pair_id': f"W{i+1}-W{i+2}",
                'wire1': i + 1,
                'wire2': i + 2,
                'pos1_px': round(p1, 2),
                'pos2_px': round(p2, 2),
                'pitch_px': round(pitch_px, 2),
                'pitch_mm': None,
                'pitch_display': f"{pitch_px:.1f} px",
                'physical_gap_mm': None,
                'physical_gap_display': "NOT AVAILABLE (Calibration Required)",
                'relative_dev_pct': round(rel_dev, 1),
                'threshold_display': threshold_label,
                'status': status_pair
            }
            pairs_data.append(p_data)
            if is_abnormal:
                abnormal_pairs.append(p_data)

        max_gap_val = float(np.max(pitches_px)) if len(pitches_px) > 0 else 0.0
        max_gap_display = f"{max_gap_val:.1f} px"

    abnormal_count = len(abnormal_pairs)
    overall_status = 'ABNORMAL' if abnormal_count > 0 else 'NORMAL'

    return {
        'is_calibrated': is_calibrated,
        'num_pairs': len(pairs_data),
        'pairs_data': pairs_data,
        'pitches_px': pitches_px,
        'final_reference_pitch_px': final_reference_pitch_px,
        'gaps_mm': gaps_mm,
        'overall_status': overall_status,
        'max_gap_val': max_gap_val,
        'max_gap_display': max_gap_display,
        'abnormal_count': abnormal_count,
        'abnormal_pairs': abnormal_pairs,
        'threshold_label': threshold_label
    }
