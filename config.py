# Configuration Parameters for Real-Time-Capable Wire Gap Detection System

# Physical Wire Belt Parameters
EXPECTED_WIRES = 48                   # Total expected parallel wires
DEFAULT_WIRE_DIAMETER_MM = 0.9        # Nominal physical wire thickness in mm

# Feature Polarity Mapping
REPRESENTATION_POLARITY = {
    'R-G': 'valleys',
    'RED': 'valleys',
    'GRAY': 'valleys',
    'INVERT_GRAY': 'peaks',
    'BLACK_HAT': 'peaks'
}

# Multi-Strip Wire-Active Region & Positioning
ACTIVE_REGION_START = 0.20            # 20% along wire length (trims mounting end-caps)
ACTIVE_REGION_END = 0.80              # 80% along wire length
STRIP_RELATIVE_POSITIONS = [0.0, 0.25, 0.50, 0.75, 1.0] # 5 relative positions inside active range
STRIP_THICKNESS_PX = 15              # Width of measurement strip in pixels

# Tracking Jump Safety
MAX_JUMP_ALPHA = 0.4                  # Pitch-relative jump limit factor
MIN_JUMP_PX = 3                       # Minimum jump limit floor in pixels

# Calibration & Limits
DEFAULT_PIXELS_PER_MM = None          # Uncalibrated mode by default
DEFAULT_DEMO_MAX_GAP_MM = 2.0         # Demo threshold limit (NOT an engineering spec)

# Peak & Detection Parameters
MIN_DISTANCE_FACTOR = 0.55            # Minimum peak distance factor relative to pitch
SMOOTHING_KSIZE = 5                   # Gaussian blur kernel for projection profile

# Normalized Confidence Engine Weights
CONFIDENCE_WEIGHTS = {
    'count': 0.25,
    'spacing': 0.20,
    'peak': 0.15,
    'segmentation': 0.15,
    'tracking': 0.15,
    'agreement': 0.10
}
