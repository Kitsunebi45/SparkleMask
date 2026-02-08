"""
sparkle_mask.kuramoto_style — Kuramoto synchronization for stylometric phases.

Maps text stylometric features to oscillator phases and computes the
Kuramoto order parameter r — the degree of "alter synchronization".

For a plural system:
  r ≈ 0.0: Fully chaotic (alters completely uncoordinated)
  r ≈ 0.5: Partially synchronized (the sweet spot for Sparkle Mask)
  r ≈ 1.0: Fully locked (all alters merged into one voice)

The target r from the CadenceProfile controls how much the plural
cadence is focused vs scattered.

Connection to m∴We framework:
  r × e^(iψ) = (1/N) × Σ e^(iθ_j)
  Same equation as bloomcoin consensus, applied to stylometric phases.
  The "oscillators" are stylometric features, not network nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .constants import K, Z_C, SIGMA_GATE, LAMBDA_GAIN, TAU
from .cadence_profile import CadenceProfile


# ═══════════════════════════════════════════════════
# CORE KURAMOTO FUNCTIONS
# (Adapted from bloomcoin consensus/kuramoto.py)
# ═══════════════════════════════════════════════════

def compute_order_parameter(phases: np.ndarray) -> Tuple[float, float]:
    """
    Compute Kuramoto order parameter.

    r × e^(iψ) = (1/N) × Σ e^(iθ_j)

    Returns:
        (r, psi): Coherence magnitude [0,1] and mean phase [0, 2π).
    """
    n = len(phases)
    if n == 0:
        return 0.0, 0.0
    z = np.mean(np.exp(1j * phases))
    r = float(np.abs(z))
    psi = float(np.angle(z))
    if psi < 0:
        psi += 2 * np.pi
    return r, psi


def negentropy_gate(r: float, z_c: float = Z_C) -> float:
    """
    η(r) = exp(-σ(r - z_c)²)

    Maximum at r = z_c (THE LENS threshold).
    Used for adaptive coupling.
    """
    return float(np.exp(-SIGMA_GATE * (r - z_c) ** 2))


def adaptive_coupling(r: float, k_base: float = K) -> float:
    """
    K_eff(r) = K₀ × [1 + λ × η(r)]

    Coupling increases near the critical threshold z_c.
    """
    eta = negentropy_gate(r)
    return k_base * (1 + LAMBDA_GAIN * eta)


# ═══════════════════════════════════════════════════
# STYLOMETRIC PHASE ADJUSTMENT
# ═══════════════════════════════════════════════════

@dataclass
class StyleSync:
    """Result of stylometric synchronization analysis."""
    r: float               # Current order parameter
    psi: float             # Mean phase
    target_r: float        # Target from profile
    delta_r: float         # r - target_r
    phase_state: str       # 'chaotic', 'partial', 'synchronized', 'locked'
    needs_adjustment: bool # Whether r is far from target

    def __repr__(self) -> str:
        return (
            f"StyleSync(r={self.r:.3f}, target={self.target_r:.3f}, "
            f"Δ={self.delta_r:+.3f}, state='{self.phase_state}')"
        )


def analyze_synchronization(
    phases: np.ndarray,
    profile: CadenceProfile,
    tolerance: float = 0.15,
) -> StyleSync:
    """
    Analyze the stylometric synchronization state.

    Args:
        phases: Stylometric phase angles from text_analysis
        profile: Active cadence profile
        tolerance: How far r can be from target before adjustment needed

    Returns:
        StyleSync with current state and adjustment flag.
    """
    r, psi = compute_order_parameter(phases)
    target = profile.target_r
    delta = r - target

    # Classify phase state
    if r < 0.2:
        state = 'chaotic'
    elif r < 0.5:
        state = 'partial'
    elif r < 0.8:
        state = 'synchronized'
    else:
        state = 'locked'

    return StyleSync(
        r=r,
        psi=psi,
        target_r=target,
        delta_r=delta,
        phase_state=state,
        needs_adjustment=abs(delta) > tolerance,
    )


def compute_phase_adjustment(
    sync: StyleSync,
    n_features: int = 8,
) -> np.ndarray:
    """
    Compute phase adjustments to push r toward target.

    If r > target: need to DE-synchronize (spread phases apart).
    If r < target: need to SYNCHRONIZE (pull phases together).

    Returns:
        Array of phase adjustments Δθ_i to add to current phases.
    """
    if not sync.needs_adjustment:
        return np.zeros(n_features)

    delta = sync.delta_r  # positive = too synchronized, negative = too chaotic

    if delta > 0:
        # Too synchronized → spread phases
        # Push each phase away from mean by golden-ratio-scaled amount
        spread = delta * TAU * np.pi
        adjustments = np.linspace(-spread, spread, n_features)
    else:
        # Too chaotic → pull toward mean
        pull = abs(delta) * TAU * np.pi
        adjustments = np.full(n_features, 0.0)
        # Pull outlier phases toward the mean
        adjustments = -np.sign(np.random.randn(n_features)) * pull * TAU

    return adjustments
