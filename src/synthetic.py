import cv2
import numpy as np

def generate_synthetic_wire_belt(scenario='normal', num_wires=48, image_size=(800, 800), orientation='VERTICAL'):
    """
    Generates realistic synthetic 48-wire belt images covering diverse industrial scenarios.
    """
    h, w = image_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Base red backlighting
    y_coords, x_coords = np.indices((h, w))
    cy, cx = h // 2, w // 2
    dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    backlight = 255.0 - (dist_from_center / max_dist) * 40.0
    backlight = np.clip(backlight, 150, 255).astype(np.uint8)

    canvas[:, :, 2] = backlight
    canvas[:, :, 1] = (backlight * 0.15).astype(np.uint8)
    canvas[:, :, 0] = (backlight * 0.10).astype(np.uint8)

    # Frame borders
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (20, 20, 20), 12)
    
    # Mounting end rails
    if orientation == 'VERTICAL':
        cv2.rectangle(canvas, (12, 15), (w - 12, 45), (40, 40, 40), -1)
        cv2.rectangle(canvas, (12, h - 45), (w - 12, h - 15), (40, 40, 40), -1)
        usable_len = w - 80
        start_pos = 40
    else:
        cv2.rectangle(canvas, (15, 12), (45, h - 12), (40, 40, 40), -1)
        cv2.rectangle(canvas, (w - 45, 12), (w - 15, h - 12), (40, 40, 40), -1)
        usable_len = h - 80
        start_pos = 40

    nominal_pitch = float(usable_len) / float(num_wires)
    wire_positions = [start_pos + i * nominal_pitch for i in range(num_wires)]

    # Apply scenario modifications
    active_positions = list(wire_positions)

    if scenario == 'abnormal_gap':
        active_positions[15] += nominal_pitch * 0.85
    elif scenario == 'missing_wire':
        active_positions.pop(20)
    elif scenario == 'slight_displacement':
        active_positions[10] += nominal_pitch * 0.15
    elif scenario == 'multiple_abnormal_gaps':
        active_positions[12] += nominal_pitch * 0.75
        active_positions[30] += nominal_pitch * 0.80
    elif scenario == 'multiple_broken_wires':
        # Remove 3 wires
        if len(active_positions) > 35:
            active_positions.pop(35)
            active_positions.pop(22)
            active_positions.pop(10)
    elif scenario == 'guide_wear_group_shift':
        # Shift a group of wires
        for idx in range(20, 26):
            if idx < len(active_positions):
                active_positions[idx] += nominal_pitch * 0.70

    wire_radius = 4

    # Draw wires
    for idx, pos in enumerate(active_positions):
        base_coord = int(round(pos))

        if orientation == 'VERTICAL':
            if scenario == 'curved_wires':
                pts = []
                for y_val in range(45, h - 45, 5):
                    curve_x = int(base_coord + 6.0 * np.sin(y_val / 40.0))
                    pts.append((curve_x, y_val))
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i+1], (15, 15, 15), wire_radius * 2)
            elif scenario == 'vibration_jitter':
                pts = []
                for y_val in range(45, h - 45, 4):
                    jitter = int(np.random.normal(0, 1.5))
                    pts.append((base_coord + jitter, y_val))
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i+1], (15, 15, 15), wire_radius * 2)
            elif scenario == 'displaced_local_wire' and idx == 18:
                pts = []
                for y_val in range(45, h - 45, 5):
                    shift = 15 if (h*0.35 <= y_val <= h*0.65) else 0
                    pts.append((base_coord + shift, y_val))
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i+1], (15, 15, 15), wire_radius * 2)
            else:
                cv2.line(canvas, (base_coord, 45), (base_coord, h - 45), (15, 15, 15), wire_radius * 2)
                cv2.line(canvas, (base_coord - 1, 45), (base_coord - 1, h - 45), (70, 70, 70), 1)
        else:
            if scenario == 'curved_wires':
                pts = []
                for x_val in range(45, w - 45, 5):
                    curve_y = int(base_coord + 6.0 * np.sin(x_val / 40.0))
                    pts.append((x_val, curve_y))
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i+1], (15, 15, 15), wire_radius * 2)
            else:
                cv2.line(canvas, (45, base_coord), (w - 45, base_coord), (15, 15, 15), wire_radius * 2)
                cv2.line(canvas, (45, base_coord - 1), (w - 45, base_coord - 1), (70, 70, 70), 1)

    # Artifact overlays
    if scenario == 'noisy_illumination':
        noise = np.random.normal(0, 18, (h, w, 3)).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif scenario == 'reflection_artifact':
        ref_x = int(round(active_positions[14])) + int(round(nominal_pitch * 0.5))
        cv2.line(canvas, (ref_x, 80), (ref_x, h - 80), (255, 230, 230), 2)
    elif scenario == 'partial_obscuration':
        obs_x = int(round(active_positions[23]))
        cv2.rectangle(canvas, (obs_x - 15, h//2 - 40), (obs_x + 15, h//2 + 40), (10, 10, 10), -1)
    elif scenario == 'dust_debris':
        # Add random dust spots
        for _ in range(30):
            rx, ry = np.random.randint(50, w-50), np.random.randint(50, h-50)
            cv2.circle(canvas, (rx, ry), np.random.randint(2, 6), (15, 15, 15), -1)

    return canvas, active_positions
