import cv2
import numpy as np

def detect_backlight_roi(img, orientation='HORIZONTAL', margin_pct=0.01, trim_rails=True, trim_pct=0.08):
    """
    Stage 1: Backlight Inspection Box ROI Localization.
    Isolates the red illuminated area using R - G difference & red thresholding.
    """
    h, w = img.shape[:2]
    r_channel = img[:, :, 2].astype(np.float32)
    g_channel = img[:, :, 1].astype(np.float32)
    
    diff = cv2.subtract(img[:, :, 2], img[:, :, 1])
    _, thresh_diff = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    
    if np.sum(thresh_diff > 0) < (h * w * 0.05):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        thresh_diff = cv2.dilate(edges, kernel)
        
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        large_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.05)]
        if large_contours:
            all_pts = np.vstack(large_contours)
            x, y, bw, bh = cv2.boundingRect(all_pts)
        else:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            
        inset_x = int(bw * 0.015)
        inset_y = int(bh * 0.015)
        
        x = x + inset_x
        y = y + inset_y
        bw = max(10, bw - 2 * inset_x)
        bh = max(10, bh - 2 * inset_y)
        
        if bw < (w * 0.3) or bh < (h * 0.3):
            x, y, bw, bh = 0, 0, w, h
    else:
        x, y, bw, bh = 0, 0, w, h

    if trim_rails:
        if orientation == 'HORIZONTAL':
            cut_x = int(bw * trim_pct)
            x = x + cut_x
            bw = max(10, bw - 2 * cut_x)
        else:
            cut_y = int(bh * trim_pct)
            y = y + cut_y
            bh = max(10, bh - 2 * cut_y)

    return (x, y, bw, bh)

def detect_two_stage_roi(img, orientation='VERTICAL', trim_rails=True):
    """
    Two-Stage ROI Localization Pipeline:
    Order of operations: Orientation is calculated FIRST.
    Stage 1: Backlight Inspection Box ROI
    Stage 2: Directional Wire-Bundle ROI Localization using sustained line density response.
             - Measure directional Sobel energy along measurement axis.
             - Project line energy -> Gaussian smooth -> contiguous high-response region
               -> aspect-ratio & size validation (excludes mounting plate edge false ROI).
    """
    # Stage 1: Backlight ROI
    bx, by, bw, bh = detect_backlight_roi(img, orientation=orientation, trim_rails=trim_rails)
    stage1_roi = img[by:by+bh, bx:bx+bw]
    sh, sw = stage1_roi.shape[:2]
    
    gray = cv2.cvtColor(stage1_roi, cv2.COLOR_BGR2GRAY) if len(stage1_roi.shape) == 3 else stage1_roi

    # Stage 2: Directional Line Energy Along Measurement Axis
    if orientation == 'VERTICAL':
        # Wires run Y (vertical). Measurement axis is X (horizontal).
        # Compute Sobel X gradient
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        energy_proj = np.mean(np.abs(grad), axis=0) # 1D profile along X
        
        # Smooth line density profile
        smoothed = cv2.GaussianBlur(energy_proj[:, None], (31, 1), 0).ravel()
        thresh_val = np.mean(smoothed) + 0.25 * np.std(smoothed)
        active_indices = np.where(smoothed > thresh_val)[0]
        
        if len(active_indices) > 20:
            x_min, x_max = active_indices[0], active_indices[-1]
            # Aspect ratio & size validation
            bundle_w = x_max - x_min
            if bundle_w > (sw * 0.40):
                margin = int(bundle_w * 0.02)
                x_start = max(0, x_min - margin)
                x_end = min(sw, x_max + margin)
                tight_x = bx + x_start
                tight_w = x_end - x_start
                return (tight_x, by, tight_w, bh)

    else: # HORIZONTAL
        # Wires run X (horizontal). Measurement axis is Y (vertical).
        grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy_proj = np.mean(np.abs(grad), axis=1) # 1D profile along Y
        
        smoothed = cv2.GaussianBlur(energy_proj[:, None], (31, 1), 0).ravel()
        thresh_val = np.mean(smoothed) + 0.25 * np.std(smoothed)
        active_indices = np.where(smoothed > thresh_val)[0]
        
        if len(active_indices) > 20:
            y_min, y_max = active_indices[0], active_indices[-1]
            bundle_h = y_max - y_min
            if bundle_h > (sh * 0.40):
                margin = int(bundle_h * 0.02)
                y_start = max(0, y_min - margin)
                y_end = min(sh, y_max + margin)
                tight_y = by + y_start
                tight_h = y_end - y_start
                return (bx, tight_y, bw, tight_h)

    # Fallback to Stage 1 ROI if Stage 2 energy validation is inconclusive
    return (bx, by, bw, bh)

def crop_roi(img, bbox):
    """
    Crops image to specified ROI bounding box (x, y, w, h).
    """
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]
    
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    return img[y:y+h, x:x+w]
