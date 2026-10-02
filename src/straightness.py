import numpy as np
import config

DEFAULT_MAX_STRAIGHTNESS_DEV_PX = getattr(config, 'DEFAULT_MAX_STRAIGHTNESS_DEV_PX', 2.0)
DEFAULT_DEMO_MAX_STRAIGHTNESS_MM = getattr(config, 'DEFAULT_DEMO_MAX_STRAIGHTNESS_MM', 0.25)

def fit_straight_line(coords_u, coords_v):
    """
    Fits a linear reference model v = m * u + c using least-squares linear regression.
    Computes orthogonal/transverse straightness residuals and peak-to-valley deflection.

    Parameters:
        coords_u (array-like): Longitudinal wire coordinates along belt travel direction.
        coords_v (array-like): Transverse wire coordinates along measurement axis.

    Returns:
        dict: Fitted line slope, intercept, fitted values, residuals, max deviation,
              peak-to-valley runout, and residual standard deviation.
    """
    u = np.array(coords_u, dtype=np.float64)
    v = np.array(coords_v, dtype=np.float64)
    n = len(u)

    if n < 2:
        return {
            'slope': 0.0,
            'intercept': float(v[0]) if n == 1 else 0.0,
            'residuals': np.zeros(n),
            'max_dev_px': 0.0,
            'peak_to_valley_px': 0.0,
            'mean_dev_px': 0.0,
            'std_dev_px': 0.0,
            'fitted_v': v.copy()
        }

    # Linear least-squares fit: v = m * u + c
    # Handles uniform u safeguard
    u_span = float(np.ptp(u))
    if u_span < 1e-5:
        mean_v = float(np.mean(v))
        res = v - mean_v
        max_d = float(np.max(np.abs(res)))
        ptv = float(np.ptp(res))
        return {
            'slope': 0.0,
            'intercept': mean_v,
            'residuals': res,
            'max_dev_px': max_d,
            'peak_to_valley_px': ptv,
            'mean_dev_px': float(np.mean(np.abs(res))),
            'std_dev_px': float(np.std(res)),
            'fitted_v': np.full_like(v, mean_v)
        }

    slope, intercept = np.polyfit(u, v, 1)
    fitted_v = slope * u + intercept
    residuals = v - fitted_v

    max_dev_px = float(np.max(np.abs(residuals)))
    if max_dev_px < 1e-10:
        max_dev_px = 0.0
        residuals = np.zeros_like(residuals)

    peak_to_valley_px = float(np.ptp(residuals))
    if peak_to_valley_px < 1e-10:
        peak_to_valley_px = 0.0

    mean_dev_px = float(np.mean(np.abs(residuals)))
    if mean_dev_px < 1e-10:
        mean_dev_px = 0.0

    std_dev_px = float(np.std(residuals))
    if std_dev_px < 1e-10:
        std_dev_px = 0.0

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'residuals': residuals,
        'max_dev_px': max_dev_px,
        'peak_to_valley_px': peak_to_valley_px,
        'mean_dev_px': mean_dev_px,
        'std_dev_px': std_dev_px,
        'fitted_v': fitted_v
    }

def evaluate_wire_straightness(
    trajectories,
    strips,
    orientation='HORIZONTAL',
    pixels_per_mm=None,
    max_straightness_tol=None
):
    """
    Evaluates straightness of each individual wire across multi-strip tracking positions:
    - Extracts multi-point longitudinal trajectories for each wire.
    - Fits reference straight line to filter overall belt skew/tilt.
    - Measures peak-to-valley runout and max deflection residual per wire.
    - Compares against maximum permissible tolerance limit (px or mm).
    - Classifies wire status: STRAIGHT vs BENT.
    - Returns comprehensive per-wire analytics and overall belt straightness assessment.

    Parameters:
        trajectories (list): List of wire trajectory dicts from track_wires_across_strips.
        strips (list): List of strip metadata dicts containing center_coord.
        orientation (str): 'HORIZONTAL' or 'VERTICAL'.
        pixels_per_mm (float or None): Calibration factor for metric conversion.
        max_straightness_tol (float or None): Permissible threshold in px (uncalibrated)
                                              or mm (calibrated).

    Returns:
        dict: Complete straightness evaluation including overall status, bent wire lists,
              and detailed per-wire metrics.
    """
    is_calibrated = (pixels_per_mm is not None) and (pixels_per_mm > 0)
    scale = float(pixels_per_mm) if is_calibrated else None

    # Determine threshold
    if is_calibrated:
        threshold_val = max_straightness_tol if max_straightness_tol is not None else DEFAULT_DEMO_MAX_STRAIGHTNESS_MM
        threshold_label = f"{threshold_val:.3f} mm"
    else:
        threshold_val = max_straightness_tol if max_straightness_tol is not None else DEFAULT_MAX_STRAIGHTNESS_DEV_PX
        threshold_label = f"{threshold_val:.2f} px"

    wires_data = []
    bent_wires = []

    for traj in trajectories:
        w_id = traj.get('wire_id', 0)
        strip_positions = traj.get('strip_positions', {})

        coords_u = []  # Longitudinal axis
        coords_v = []  # Transverse / measurement axis
        points_2d = [] # (x, y) coordinates in ROI space

        sorted_strips = sorted(strip_positions.items(), key=lambda item: item[0])

        for s_idx, pos_val in sorted_strips:
            if s_idx < len(strips):
                longitudinal_coord = strips[s_idx]['center_coord']
            else:
                longitudinal_coord = s_idx * 50  # Fallback

            if orientation == 'HORIZONTAL':
                # Longitudinal axis is X, measurement axis is Y
                x_coord = float(longitudinal_coord)
                y_coord = float(pos_val)
                coords_u.append(x_coord)
                coords_v.append(y_coord)
                points_2d.append((round(x_coord, 1), round(y_coord, 2)))
            else:
                # Longitudinal axis is Y, measurement axis is X
                y_coord = float(longitudinal_coord)
                x_coord = float(pos_val)
                coords_u.append(y_coord)
                coords_v.append(x_coord)
                points_2d.append((round(x_coord, 2), round(y_coord, 1)))

        num_points = len(coords_u)

        if num_points >= 2:
            fit_res = fit_straight_line(coords_u, coords_v)
            max_dev_px = fit_res['max_dev_px']
            ptv_px = fit_res['peak_to_valley_px']
            mean_dev_px = fit_res['mean_dev_px']
            std_dev_px = fit_res['std_dev_px']
            slope = fit_res['slope']
            angle_deg = float(np.degrees(np.arctan(slope)))

            if is_calibrated:
                max_dev_mm = max_dev_px / scale
                ptv_mm = ptv_px / scale
                mean_dev_mm = mean_dev_px / scale
                is_bent = bool(max_dev_mm > threshold_val)
                max_dev_display = f"{max_dev_mm:.3f} mm"
                ptv_display = f"{ptv_mm:.3f} mm"
            else:
                max_dev_mm = None
                ptv_mm = None
                mean_dev_mm = None
                is_bent = bool(max_dev_px > threshold_val)
                max_dev_display = f"{max_dev_px:.2f} px"
                ptv_display = f"{ptv_px:.2f} px"

            status = 'BENT' if is_bent else 'STRAIGHT'

            wire_info = {
                'wire_id': w_id,
                'wire_label': f"W{w_id}",
                'points_count': num_points,
                'points_2d': points_2d,
                'max_dev_px': round(max_dev_px, 2),
                'max_dev_mm': round(max_dev_mm, 4) if max_dev_mm is not None else None,
                'peak_to_valley_px': round(ptv_px, 2),
                'peak_to_valley_mm': round(ptv_mm, 4) if ptv_mm is not None else None,
                'mean_dev_px': round(mean_dev_px, 2),
                'mean_dev_mm': round(mean_dev_mm, 4) if mean_dev_mm is not None else None,
                'std_dev_px': round(std_dev_px, 2),
                'max_dev_display': max_dev_display,
                'ptv_display': ptv_display,
                'slope_angle_deg': round(angle_deg, 2),
                'threshold_display': threshold_label,
                'is_bent': is_bent,
                'status': status
            }
            wires_data.append(wire_info)
            if is_bent:
                bent_wires.append(wire_info)

        else:
            # Insufficient multi-strip points for curvature assessment
            wires_data.append({
                'wire_id': w_id,
                'wire_label': f"W{w_id}",
                'points_count': num_points,
                'points_2d': points_2d,
                'max_dev_px': 0.0,
                'max_dev_mm': 0.0 if is_calibrated else None,
                'peak_to_valley_px': 0.0,
                'peak_to_valley_mm': 0.0 if is_calibrated else None,
                'mean_dev_px': 0.0,
                'mean_dev_mm': 0.0 if is_calibrated else None,
                'std_dev_px': 0.0,
                'max_dev_display': 'N/A',
                'ptv_display': 'N/A',
                'slope_angle_deg': 0.0,
                'threshold_display': threshold_label,
                'is_bent': False,
                'status': 'INSUFFICIENT_POINTS'
            })

    bent_count = len(bent_wires)
    overall_status = 'ABNORMAL' if bent_count > 0 else 'NORMAL'

    all_devs_px = [w['max_dev_px'] for w in wires_data if w['points_count'] >= 2]
    max_belt_dev_px = float(np.max(all_devs_px)) if all_devs_px else 0.0
    mean_belt_dev_px = float(np.mean(all_devs_px)) if all_devs_px else 0.0

    if is_calibrated:
        max_belt_dev_mm = max_belt_dev_px / scale
        max_belt_dev_display = f"{max_belt_dev_mm:.3f} mm"
    else:
        max_belt_dev_mm = None
        max_belt_dev_display = f"{max_belt_dev_px:.2f} px"

    return {
        'is_calibrated': is_calibrated,
        'overall_status': overall_status,
        'bent_wire_count': bent_count,
        'bent_wires': bent_wires,
        'wires_data': wires_data,
        'total_evaluated_wires': len(wires_data),
        'max_belt_dev_px': round(max_belt_dev_px, 2),
        'max_belt_dev_mm': round(max_belt_dev_mm, 4) if max_belt_dev_mm is not None else None,
        'mean_belt_dev_px': round(mean_belt_dev_px, 2),
        'max_belt_dev_display': max_belt_dev_display,
        'threshold_val': threshold_val,
        'threshold_label': threshold_label
    }
