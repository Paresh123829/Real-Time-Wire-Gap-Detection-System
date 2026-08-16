def compute_pixels_per_mm(measured_pixels, known_mm):
    """
    Calculates calibration scale factor (pixels per mm).
    """
    if known_mm is None or known_mm <= 0 or measured_pixels is None or measured_pixels <= 0:
        return None
    return float(measured_pixels) / float(known_mm)

def px_to_mm(val_px, pixels_per_mm):
    """
    Converts pixel value to millimeters if calibration is available.
    Returns (converted_val, unit_string, is_calibrated_bool)
    """
    if pixels_per_mm is None or pixels_per_mm <= 0:
        return float(val_px), "px", False
    val_mm = float(val_px) / float(pixels_per_mm)
    return round(val_mm, 2), "mm", True

def format_gap_value(val_px, pixels_per_mm):
    """
    Returns formatted string with appropriate units.
    """
    val, unit, is_calibrated = px_to_mm(val_px, pixels_per_mm)
    if is_calibrated:
        return f"{val:.2f} mm"
    else:
        return f"{val_px:.1f} px"
