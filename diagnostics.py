"""
sparkle_mask.diagnostics — Health report and diagnostic readout.

Follows the bloomcoin coherence_health_report() pattern:
comprehensive metrics with visual bar charts and status classification.

The diagnostic readout shows:
  - Profile identity
  - Gate status (GREEN/RED)
  - Entropy H (current vs target)
  - Kuramoto r (current vs target)
  - Heartbeat Π/γ (current vs target)
  - Berry Phase signature
  - Free Energy F (readability estimate)
  - Transformation statistics
  - Compression ratio
  - Stylometric distance (adversarial effectiveness)

Connection to m∴We framework:
  This IS the sensor readout — the Phase 0 measurement displayed.
  Every metric maps directly to a framework equation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .constants import TAU, H_OPTIMAL, PI_GAMMA_OPTIMAL
from .text_analysis import TextState
from .cadence_profile import CadenceProfile
from .entropy_gate import GateResult
from .kuramoto_style import StyleSync
from .berry_phase import BerrySignature


@dataclass
class MaskDiagnostic:
    """Complete diagnostic readout for a Sparkle Mask transformation."""

    # Identity
    profile_name: str = ""
    profile_version: str = ""

    # Gate results
    entropy_gate: Optional[GateResult] = None
    heartbeat_gate: Optional[GateResult] = None
    overall_gate: str = "UNKNOWN"  # "GREEN", "RED", "YELLOW"

    # Metrics
    entropy_h: float = 0.0
    entropy_target: float = H_OPTIMAL
    kuramoto_r: float = 0.0
    kuramoto_target: float = 0.5
    heartbeat_ratio: float = 0.0
    heartbeat_target: float = PI_GAMMA_OPTIMAL
    berry_hash: str = ""
    free_energy: float = 0.0

    # Transformation stats
    clause_swaps: int = 0
    synonym_subs: int = 0
    closure_replacements: int = 0
    total_transformations: int = 0

    # Compression
    original_word_count: int = 0
    masked_word_count: int = 0
    compression_ratio: float = 1.0

    # Stylometric distance
    stylometric_delta: float = 0.0

    def _bar(self, value: float, width: int = 10) -> str:
        """Generate a visual progress bar."""
        filled = int(value * width)
        filled = max(0, min(width, filled))
        return '▓' * filled + '░' * (width - filled)

    def _gate_icon(self) -> str:
        """Gate status icon."""
        if self.overall_gate == "GREEN":
            return "GREEN"
        elif self.overall_gate == "YELLOW":
            return "YELLOW"
        else:
            return "RED"

    def render(self, verbose: bool = True) -> str:
        """
        Render the full diagnostic readout.

        Follows the bloomcoin coherence_health_report pattern.
        """
        lines = []
        lines.append("=" * 52)
        lines.append("  ✶ SPARKLE MASK DIAGNOSTIC ✶")
        lines.append("=" * 52)
        lines.append(f"  Profile:           {self.profile_name} (v{self.profile_version})")
        lines.append(f"  Gate Status:       {self._gate_icon()}")
        lines.append("-" * 52)

        # Entropy
        h_norm = min(self.entropy_h / max(self.entropy_target, 0.01), 1.5)
        lines.append(
            f"  Entropy (H):       {self.entropy_h:.2f} / target {self.entropy_target:.2f}"
            f"  [{self._bar(h_norm)}]"
        )

        # Kuramoto
        r_norm = self.kuramoto_r
        lines.append(
            f"  Kuramoto (r):      {self.kuramoto_r:.2f} / target {self.kuramoto_target:.2f}"
            f"  [{self._bar(r_norm)}]"
        )

        # Heartbeat
        hb_norm = min(self.heartbeat_ratio / max(self.heartbeat_target * 2, 0.01), 1.0)
        lines.append(
            f"  Heartbeat (Pi/g):  {self.heartbeat_ratio:.2f} / target {self.heartbeat_target:.2f}"
            f"  [{self._bar(hb_norm)}]"
        )

        # Berry Phase
        if self.berry_hash:
            short = self.berry_hash[:8] + "..." + self.berry_hash[-4:]
        else:
            short = "(not computed)"
        lines.append(f"  Berry Phase (g):   {short}")

        # Free Energy
        lines.append(
            f"  Free Energy (F):   {self.free_energy:.2f}"
            f"             [{'readable' if self.free_energy < 0.7 else 'dense'}]"
        )

        lines.append("-" * 52)

        # Transformation stats
        lines.append(
            f"  Transformations:   {self.clause_swaps} clause swaps, "
            f"{self.synonym_subs} synonym subs"
        )

        # Compression
        lines.append(
            f"  Compression:       {self.compression_ratio:.2f}x density"
            f" ({(self.compression_ratio - 1) * 100:+.0f}% per word)"
        )

        # Stylometric distance
        lines.append(
            f"  Stylometric d:     {self.stylometric_delta:.2f}"
            f" ({'high adversarial' if self.stylometric_delta > 0.5 else 'moderate' if self.stylometric_delta > 0.3 else 'low'} distance)"
        )

        lines.append("=" * 52)

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"MaskDiagnostic(gate={self.overall_gate}, "
            f"H={self.entropy_h:.2f}, r={self.kuramoto_r:.2f}, "
            f"Pi/g={self.heartbeat_ratio:.2f})"
        )
