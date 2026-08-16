import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_wire_annotations(
    roi,
    positions,
    gap_eval_results,
    orientation='HORIZONTAL',
    confidence_dict=None,
    raw_candidates=None,
    rejected_candidates=None,
    debug_mode=False
):
    """
    Draws clear visual inspection overlay on ROI:
    - Neutral wire centerlines (W1, W2, ... W48)
    - Highlight ONLY abnormal gaps in bold red callout boxes
    - Optional debug overlay (raw candidate peaks & rejected peaks)
    """
    annotated = roi.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    h, w = annotated.shape[:2]

    # Draw neutral wire centerlines
    for idx, pos in enumerate(positions):
        p_int = int(round(pos))
        w_label = f"W{idx+1}"

        if orientation == 'HORIZONTAL':
            cv2.line(annotated, (0, p_int), (w, p_int), (0, 255, 0), 1)
            cv2.putText(annotated, w_label, (5, max(12, p_int - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        else:
            cv2.line(annotated, (p_int, 0), (p_int, h), (0, 255, 0), 1)
            cv2.putText(annotated, w_label, (max(2, p_int - 8), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Optional DEBUG mode candidates
    if debug_mode:
        if raw_candidates is not None:
            for p in raw_candidates:
                p_int = int(round(p))
                if orientation == 'HORIZONTAL':
                    cv2.circle(annotated, (w // 2, p_int), 3, (0, 255, 255), -1)
                else:
                    cv2.circle(annotated, (p_int, h // 2), 3, (0, 255, 255), -1)

        if rejected_candidates is not None:
            for p in rejected_candidates:
                p_int = int(round(p))
                if orientation == 'HORIZONTAL':
                    cv2.drawMarker(annotated, (w // 2 + 30, p_int), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
                else:
                    cv2.drawMarker(annotated, (p_int, h // 2 + 30), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

    # Highlight ONLY ABNORMAL GAPS in bold red callout boxes
    abnormal_pairs = gap_eval_results.get('abnormal_pairs', [])
    for ab in abnormal_pairs:
        pair_key = ab.get('pair_id') or ab.get('wire_pair') or 'W1-W2'
        gap_text = ab.get('physical_gap_display') or ab.get('gap_disp') or ab.get('pitch_display') or 'ABNORMAL'

        w1_val = ab.get('wire1')
        w2_val = ab.get('wire2')

        if w1_val is not None and w2_val is not None:
            w1_i = int(w1_val) - 1
            w2_i = int(w2_val) - 1
        else:
            parts = pair_key.split('-')
            w1_i = int(parts[0].replace('W', '')) - 1
            w2_i = int(parts[1].replace('W', '')) - 1

        if 0 <= w1_i < len(positions) and 0 <= w2_i < len(positions):
            p1 = int(round(positions[w1_i]))
            p2 = int(round(positions[w2_i]))
            mid_p = (p1 + p2) // 2

            if orientation == 'HORIZONTAL':
                y1, y2 = min(p1, p2), max(p1, p2)
                cv2.rectangle(annotated, (w // 4, y1), (3 * w // 4, y2), (0, 0, 255), 2)
                label = f"ALERT: {pair_key} GAP: {gap_text}"
                cv2.putText(annotated, label, (w // 4 + 10, mid_p + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                x1, x2 = min(p1, p2), max(p1, p2)
                cv2.rectangle(annotated, (x1, h // 4), (x2, 3 * h // 4), (0, 0, 255), 2)
                label = f"ALERT: {pair_key} GAP: {gap_text}"
                cv2.putText(annotated, label, (mid_p - 40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Header banner info
    status_str = f"Status: {gap_eval_results.get('overall_status', 'NORMAL')} | Wires: {len(positions)}/48"
    cv2.putText(annotated, status_str, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated

def plot_1d_projection_profile(
    profile,
    positions,
    orientation='HORIZONTAL',
    raw_candidates=None,
    rejected_candidates=None,
    estimated_pitch=15.0,
    debug_mode=False
):
    """
    Plots 1D projection profile with detected wire positions and debug overlay using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)

    x_vals = np.arange(len(profile))
    ax.plot(x_vals, profile, color='#1E293B', linewidth=1.5, label='1D Profile')

    # Draw detected wire centers
    for idx, pos in enumerate(positions):
        ax.axvline(x=pos, color='#10B981', linestyle='--', alpha=0.7, label='Wire Center' if idx == 0 else "")

    if debug_mode:
        if raw_candidates is not None:
            for p in raw_candidates:
                if 0 <= p < len(profile):
                    ax.plot(p, profile[p], 'o', color='#F59E0B', markersize=5, label='Raw Candidate' if p == raw_candidates[0] else "")

        if rejected_candidates is not None:
            for p in rejected_candidates:
                if 0 <= p < len(profile):
                    ax.plot(p, profile[p], 'x', color='#EF4444', markersize=7, label='Rejected Candidate' if p == rejected_candidates[0] else "")

    ax.set_title(f"1D Projection Profile & Detected Wire Centers (Ref Pitch: {estimated_pitch:.1f} px)", fontsize=11)
    ax.set_xlabel("Pixel Coordinate along Measurement Axis")
    ax.set_ylabel("Intensity")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    return fig
