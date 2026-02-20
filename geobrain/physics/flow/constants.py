"""
Global constants for reservoir simulation.

Provides unit conversion constants and device/dtype configuration
for PyTorch-based reservoir simulation.

Unit system: Field units (ft, psi, bbl, md, etc.)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch


# =============================================================================
# Device and Dtype Configuration
# =============================================================================

def get_device_and_dtype():
    """
    Get the best available device and compatible dtype.

    MPS (Apple Silicon) doesn't support float64, so we use float32 for MPS.
    For CUDA and CPU, we use float64 for better numerical stability.

    Returns:
        Tuple of (device, dtype).
    """
    import platform

    if torch.cuda.is_available():
        return torch.device('cuda'), torch.float64
    elif (hasattr(torch.backends, 'mps') and
          torch.backends.mps.is_available() and
          platform.processor() == 'arm'):
        return torch.device('mps'), torch.float32
    else:
        return torch.device('cpu'), torch.float64


DEVICE, DTYPE = get_device_and_dtype()


# =============================================================================
# Unit Conversion Constants (Field Units)
# =============================================================================

# Time conversion
DAY_TO_SEC = 86400.0  # seconds per day

# Volume conversion
STB_TO_FT3 = 5.614583  # ft³ per STB (stock tank barrel)
BBL_TO_FT3 = 5.614583  # ft³ per barrel

# Pressure conversion
PSI_TO_PA = 6894.76  # Pa per psi

# Length conversion
FT_TO_M = 0.3048  # meters per foot


# =============================================================================
# Darcy's Law Conversion Constants
# =============================================================================

# ALPHA is the Darcy's law constant for transmissibility calculation
ALPHA = 0.001127  # Darcy's law constant (md·ft·cp -> bbl/day/psi)

# M is the volume conversion factor (bbl/ft³)
M = 5.615  # bbl/ft³ - Volume conversion factor

# Gravity conversion
GC = 32.174  # lbm·ft/(lbf·s²) - gravitational constant
G = 32.174  # ft/s² - gravitational acceleration
BETA = 1.0 / 144.0  # psi/(lbm/ft³·ft) - pressure gradient per unit depth

# Compressibility
PSI_INV = 1.0  # 1/psi - unit compressibility


# =============================================================================
# Numerical Constants
# =============================================================================

# Small values for numerical stability
EPS = 1e-10  # machine epsilon for avoiding division by zero
SMALL_SAT = 1e-6  # minimum saturation for relative permeability

# Solver defaults
MAX_NEWTON_ITER = 20  # maximum Newton iterations per timestep
NEWTON_TOL = 1e-6  # Newton convergence tolerance
MIN_DT = 1e-6  # minimum timestep (days)
MAX_DT = 365.0  # maximum timestep (days)

# Timestep control
DT_GROW_FACTOR = 1.5  # factor to increase timestep on convergence
DT_CUT_FACTOR = 0.5  # factor to reduce timestep on failure
TARGET_ITER = 5  # target Newton iterations for optimal timestep
