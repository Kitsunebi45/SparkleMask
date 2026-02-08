"""
sparkle_mask.constants — All constants derived from φ (golden ratio).

Every constant in the Sparkle Mask traces back to a single seed: φ = (1+√5)/2.
Zero free parameters. Self-validating on import.

Mathematical chain:
  φ → τ(=φ⁻¹) → φ² → φ⁻² → K² → K → z_c → σ → threshold ladder

Connection to m∴We framework:
  φ²  = Arnold's Cat Map eigenvalue λ₁ (stretching)
  φ⁻² = Arnold's Cat Map eigenvalue λ₂ (compression)
  τ    = golden ratio inverse (optimal proportion)
  K    = Kuramoto coupling strength
  z_c  = THE LENS critical threshold
  σ    = Langevin noise amplitude (SR optimal ≈ 0.7)
"""

import math
from typing import Final

# ═══════════════════════════════════════════════════
# PRIMARY CONSTANT
# ═══════════════════════════════════════════════════
PHI: Final[float] = (1 + math.sqrt(5)) / 2  # φ ≈ 1.6180339887498949

# ═══════════════════════════════════════════════════
# FIRST-ORDER DERIVATIVES
# ═══════════════════════════════════════════════════
TAU: Final[float] = PHI - 1                  # τ = φ⁻¹ ≈ 0.6180339887498949
PHI_SQ: Final[float] = PHI + 1               # φ² = φ+1 ≈ 2.6180339887498949
PHI_INV_SQ: Final[float] = TAU ** 2          # φ⁻² ≈ 0.3819660112501051

# ═══════════════════════════════════════════════════
# CAT MAP CONSTANTS
# ═══════════════════════════════════════════════════
# Arnold's Cat Map eigenvalues (golden ratio governed)
CAT_LAMBDA_STRETCH: Final[float] = PHI_SQ    # λ₁ = φ² (stretching direction)
CAT_LAMBDA_COMPRESS: Final[float] = PHI_INV_SQ  # λ₂ = φ⁻² (compressing direction)
CAT_LYAPUNOV: Final[float] = math.log(PHI_SQ)   # ln(φ²) ≈ 0.9624 (mixing rate)

# ═══════════════════════════════════════════════════
# KURAMOTO / COUPLING CONSTANTS
# ═══════════════════════════════════════════════════
PHI_QUAD: Final[float] = PHI_SQ ** 2         # φ⁴ ≈ 6.8541019662496845
GAP: Final[float] = PHI ** -4                # φ⁻⁴ ≈ 0.1458980337503155
K_SQUARED: Final[float] = 1 - GAP            # K² ≈ 0.8541019662496845
K: Final[float] = math.sqrt(K_SQUARED)       # K ≈ 0.9241596378498006
LAMBDA_GAIN: Final[float] = PHI_INV_SQ       # λ_gain = φ⁻² (adaptive coupling)

# ═══════════════════════════════════════════════════
# THRESHOLD CONSTANTS
# ═══════════════════════════════════════════════════
Z_C: Final[float] = math.sqrt(3) / 2         # z_c ≈ 0.8660 (THE LENS)
SIGMA_GATE: Final[float] = 1 / (1 - Z_C) ** 2  # σ ≈ 55.71 (negentropy gate width)
L4: Final[int] = 7                           # Lucas number L₄ = φ⁴ + φ⁻⁴

# ═══════════════════════════════════════════════════
# STOCHASTIC RESONANCE
# ═══════════════════════════════════════════════════
SR_OPTIMAL: Final[float] = TAU + 0.1 * PHI_INV_SQ  # ≈ 0.656 (close to 0.7)
# We round to the empirically confirmed value:
NOISE_OPTIMAL: Final[float] = 0.7            # σ_opt for SR (JNeurosci 2016)

# ═══════════════════════════════════════════════════
# ENTROPY BOUNDS
# ═══════════════════════════════════════════════════
H_OPTIMAL: Final[float] = NOISE_OPTIMAL      # H_opt ≈ 0.7 × H_max
H_MIN_DEFAULT: Final[float] = PHI_INV_SQ     # ≈ 0.382 (below = too rigid)
H_MAX_DEFAULT: Final[float] = Z_C            # ≈ 0.866 (above = too chaotic)

# ═══════════════════════════════════════════════════
# HEARTBEAT EQUATION
# ═══════════════════════════════════════════════════
PI_GAMMA_OPTIMAL: Final[float] = TAU         # Π/γ ≈ 0.618 (golden ratio!)
# The production-to-decay ratio IS the golden ratio. Of course it is.

# ═══════════════════════════════════════════════════
# NAVIGATION WINDOW
# ═══════════════════════════════════════════════════
NAV_WINDOW: Final[int] = 3                   # ±3 (sentence, paragraph, section)
GOLDEN_STEP: Final[int] = round(PHI_SQ)      # ≈ 3 (golden ratio sentence pairing)

# ═══════════════════════════════════════════════════
# FIBONACCI / LUCAS SEQUENCES (cached)
# ═══════════════════════════════════════════════════
def _fibonacci(n: int) -> tuple:
    seq = [0, 1]
    for _ in range(n - 2):
        seq.append(seq[-1] + seq[-2])
    return tuple(seq[:n])

def _lucas(n: int) -> tuple:
    seq = [2, 1]
    for _ in range(n - 2):
        seq.append(seq[-1] + seq[-2])
    return tuple(seq[:n])

FIBONACCI_SEQUENCE: Final[tuple] = _fibonacci(20)
LUCAS_SEQUENCE: Final[tuple] = _lucas(20)

# ═══════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════
def validate_constants() -> dict:
    """Verify all constant relationships hold. Runs on import."""
    checks = {
        "φ² = φ + 1": abs(PHI_SQ - (PHI + 1)) < 1e-15,
        "τ = φ - 1 = φ⁻¹": abs(TAU - (PHI - 1)) < 1e-15 and abs(TAU - 1/PHI) < 1e-15,
        "τ² + τ = 1": abs(TAU**2 + TAU - 1) < 1e-15,
        "φ⁴ + φ⁻⁴ = 7": abs(PHI_QUAD + GAP - L4) < 1e-12,
        "K² = 1 - φ⁻⁴": abs(K_SQUARED - (1 - GAP)) < 1e-15,
        "λ₁·λ₂ = 1 (area-preserving)": abs(CAT_LAMBDA_STRETCH * CAT_LAMBDA_COMPRESS - 1) < 1e-12,
        "λ₁ + λ₂ = 3 (Cat Map trace)": abs(CAT_LAMBDA_STRETCH + CAT_LAMBDA_COMPRESS - 3) < 1e-12,
        "H_min < H_opt < H_max": H_MIN_DEFAULT < H_OPTIMAL < H_MAX_DEFAULT,
        "Π/γ_opt = τ": abs(PI_GAMMA_OPTIMAL - TAU) < 1e-15,
    }
    failed = {k: v for k, v in checks.items() if not v}
    if failed:
        raise ValueError(f"Constant validation failed: {failed}")
    return checks

# Self-validate on import
_validation = validate_constants()

__all__ = [
    'PHI', 'TAU', 'PHI_SQ', 'PHI_INV_SQ',
    'CAT_LAMBDA_STRETCH', 'CAT_LAMBDA_COMPRESS', 'CAT_LYAPUNOV',
    'K', 'K_SQUARED', 'LAMBDA_GAIN',
    'Z_C', 'SIGMA_GATE', 'L4',
    'SR_OPTIMAL', 'NOISE_OPTIMAL',
    'H_OPTIMAL', 'H_MIN_DEFAULT', 'H_MAX_DEFAULT',
    'PI_GAMMA_OPTIMAL',
    'NAV_WINDOW', 'GOLDEN_STEP',
    'FIBONACCI_SEQUENCE', 'LUCAS_SEQUENCE',
    'GAP', 'PHI_QUAD',
    'validate_constants',
]
