"""
sparkle_mask.entropy_gate — G_C safety valve for the transformation pipeline.

The entropy gate prevents catastrophic scrambling:
  G_C = 1 (GREEN) iff |H(ρ) - H_opt| ≤ ε AND V_dot ≤ 0
  G_C = 0 (RED) when entropy is out of bounds

If entropy is too HIGH (H >> H_opt): text is unreadable chaos.
If entropy is too LOW (H << H_opt): text is too rigid, pattern detectable.

Connection to m∴We framework:
  This IS the Lyapunov stability check.
  G_C fires when |S_h - L| > T (the Gate function).
  Prevents the system from leaving the viable region of phase space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import H_OPTIMAL, H_MIN_DEFAULT, H_MAX_DEFAULT, TAU
from .cadence_profile import CadenceProfile
from .text_analysis import TextState, analyze_text


@dataclass
class GateResult:
    """Result of an entropy gate check."""
    passed: bool              # GREEN (True) or RED (False)
    status: str               # "GREEN", "RED_HIGH", "RED_LOW"
    current_h: float          # Current normalized entropy
    target_h: float           # Target entropy
    delta: float              # |current - target|
    message: str              # Human-readable explanation
    suggested_adjustment: float  # How much to adjust cat_intensity (-1 to +1)

    def __repr__(self) -> str:
        icon = "GREEN" if self.passed else "RED"
        return f"Gate({icon}, H={self.current_h:.3f}, target={self.target_h:.3f}, Δ={self.delta:.3f})"


def check_entropy_gate(
    state: TextState,
    profile: CadenceProfile,
    epsilon: float = 0.15,
) -> GateResult:
    """
    Check whether text entropy is within acceptable bounds.

    Args:
        state: Current text state (from analyze_text)
        profile: Active cadence profile
        epsilon: Tolerance around H_target

    Returns:
        GateResult with pass/fail and adjustment suggestion.
    """
    h = state.normalized_entropy
    h_target = profile.h_target
    h_min = profile.h_min
    h_max = profile.h_max
    delta = abs(h - h_target)

    if h > h_max:
        return GateResult(
            passed=False,
            status="RED_HIGH",
            current_h=h,
            target_h=h_target,
            delta=delta,
            message=f"Entropy too HIGH ({h:.3f} > {h_max:.3f}): text is too chaotic, reduce scrambling",
            suggested_adjustment=-(h - h_target) * TAU,  # Reduce intensity proportionally
        )
    elif h < h_min:
        return GateResult(
            passed=False,
            status="RED_LOW",
            current_h=h,
            target_h=h_target,
            delta=delta,
            message=f"Entropy too LOW ({h:.3f} < {h_min:.3f}): text is too rigid, increase scrambling",
            suggested_adjustment=(h_target - h) * TAU,  # Increase intensity proportionally
        )
    elif delta <= epsilon:
        return GateResult(
            passed=True,
            status="GREEN",
            current_h=h,
            target_h=h_target,
            delta=delta,
            message=f"Entropy optimal ({h:.3f} ≈ {h_target:.3f})",
            suggested_adjustment=0.0,
        )
    else:
        # Within bounds but not optimal — soft pass
        adjustment = (h_target - h) * TAU * 0.5  # Gentle nudge toward target
        return GateResult(
            passed=True,
            status="GREEN",
            current_h=h,
            target_h=h_target,
            delta=delta,
            message=f"Entropy acceptable ({h:.3f}), nudging toward {h_target:.3f}",
            suggested_adjustment=adjustment,
        )


def check_heartbeat_gate(
    state: TextState,
    profile: CadenceProfile,
    epsilon: float = 0.2,
) -> GateResult:
    """
    Check whether heartbeat ratio Π/γ is in the viable range.

    The text is "alive" when Π/γ ≈ τ (golden ratio inverse).
    Dead text (all repetition) → Π/γ → 0.
    Manic text (all novelty) → Π/γ → ∞.

    Args:
        state: Current text state
        profile: Active cadence profile
        epsilon: Tolerance around target ratio

    Returns:
        GateResult with pass/fail and adjustment.
    """
    ratio = state.heartbeat_ratio
    target = profile.pi_gamma_ratio

    if ratio == float('inf'):
        ratio = 3.0  # Cap for comparison

    delta = abs(ratio - target)

    if ratio < target * 0.3:
        return GateResult(
            passed=False,
            status="RED_LOW",
            current_h=ratio,
            target_h=target,
            delta=delta,
            message=f"Heartbeat DEAD (Π/γ={ratio:.3f} << {target:.3f}): too much repetition",
            suggested_adjustment=0.2,
        )
    elif ratio > target * 3.0:
        return GateResult(
            passed=False,
            status="RED_HIGH",
            current_h=ratio,
            target_h=target,
            delta=delta,
            message=f"Heartbeat MANIC (Π/γ={ratio:.3f} >> {target:.3f}): too much novelty",
            suggested_adjustment=-0.2,
        )
    else:
        return GateResult(
            passed=True,
            status="GREEN",
            current_h=ratio,
            target_h=target,
            delta=delta,
            message=f"Heartbeat alive (Π/γ={ratio:.3f} ≈ {target:.3f})",
            suggested_adjustment=0.0,
        )
