import cv2
import numpy as np
import config

def extract_local_strips(
    roi,
    orientation='HORIZONTAL',
    active_start=config.ACTIVE_REGION_START,
    active_end=config.ACTIVE_REGION_END,
    relative_positions=config.STRIP_RELATIVE_POSITIONS,
    strip_thickness=config.STRIP_THICKNESS_PX
):
    """
    Extracts local measurement strips along the wire-active longitudinal range.
    - active_start=0.20, active_end=0.80 (20% to 80% along wire length).
    - relative_positions=[0.0, 0.25, 0.50, 0.75, 1.0] maps to 20%, 35%, 50%, 65%, 80%.
    - Measurement axis is perpendicular to wire axis.
    """
    h, w = roi.shape[:2]
    half_t = max(1, strip_thickness // 2)
    strips = []

    active_len_ratio = active_end - active_start

    for rel_pos in relative_positions:
        ratio = active_start + rel_pos * active_len_ratio

        if orientation == 'HORIZONTAL':
            # Wires run X. Longitudinal axis is X (width). Measurement axis is Y (height).
            center_x = int(w * ratio)
            x_start = max(0, center_x - half_t)
            x_end = min(w, center_x + half_t + 1)
            strip_img = roi[:, x_start:x_end]
            strips.append({
                'ratio': ratio,
                'rel_pos': rel_pos,
                'center_coord': center_x,
                'bbox': (x_start, 0, x_end - x_start, h),
                'img': strip_img
            })
        else:
            # Wires run Y. Longitudinal axis is Y (height). Measurement axis is X (width).
            center_y = int(h * ratio)
            y_start = max(0, center_y - half_t)
            y_end = min(h, center_y + half_t + 1)
            strip_img = roi[y_start:y_end, :]
            strips.append({
                'ratio': ratio,
                'rel_pos': rel_pos,
                'center_coord': center_y,
                'bbox': (0, y_start, w, y_end - y_start),
                'img': strip_img
            })

    return strips

def compute_strip_profiles(strips, orientation='HORIZONTAL', channel=None):
    """
    Computes 1D intensity projection profile along the measurement axis for each strip.
    Profile length equals ROI height for horizontal wires, ROI width for vertical wires.
    """
    profiles = []
    for strip in strips:
        strip_img = strip['img']
        if channel is not None:
            x, y, sw, sh = strip['bbox']
            crop_ch = channel[y:y+sh, x:x+sw]
        else:
            crop_ch = strip_img

        if len(crop_ch.shape) == 3:
            crop_ch = cv2.cvtColor(crop_ch, cv2.COLOR_BGR2GRAY)

        if orientation == 'HORIZONTAL':
            # Profile along Y (measurement axis)
            p = np.mean(crop_ch.astype(np.float32), axis=1).ravel()
        else:
            # Profile along X (measurement axis)
            p = np.mean(crop_ch.astype(np.float32), axis=0).ravel()

        profiles.append({
            'ratio': strip['ratio'],
            'rel_pos': strip['rel_pos'],
            'center_coord': strip['center_coord'],
            'profile': p.astype(np.float32)
        })

    return profiles
