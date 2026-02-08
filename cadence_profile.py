"""
sparkle_mask.cadence_profile — JSON-backed cadence presets for plural dynamics.

Personas are parameter vectors, not switches.
Load, unload, blend, hot-swap — as squishy and malleable as you are.

The CadenceProfile dataclass holds every tunable parameter of the Sparkle Mask.
JSON serialization means presets are portable, shareable, version-controlled.
The blend() method enables continuous interpolation between personas.

Connection to m∴We framework:
  Each profile IS a point on the Kuramoto torus.
  blend() IS linear interpolation on the torus (geodesic between personas).
  hot-swap IS topology switching (Cat → Fox → Fixed Point transitions).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, List

from .constants import (
    K, TAU, NOISE_OPTIMAL, H_OPTIMAL,
    H_MIN_DEFAULT, H_MAX_DEFAULT, PI_GAMMA_OPTIMAL,
    NAV_WINDOW,
)


@dataclass
class CadenceProfile:
    """
    A complete cadence configuration for the Sparkle Mask.

    Every numeric parameter can be blended between two profiles.
    Non-numeric fields (name, description, tags) come from the primary profile.
    """

    # ── Identity ──────────────────────────────────
    name: str = "default"
    description: str = "Balanced cadence scrambling"

    # ── Cat Map Parameters (scrambling intensity) ─
    cat_intensity: float = 0.5          # 0.0–1.0, how much mixing
    cat_matrix: Optional[list] = None   # Custom 2x2 matrix; None = Arnold's [[2,1],[1,1]]
    golden_scaling: bool = True         # Use φ² eigenvalue spacing for sentence pairing

    # ── Kuramoto Parameters (alter synchronization)
    target_r: float = 0.5              # Order parameter target (0=chaos, 1=lock)
    coupling_k: float = K              # Coupling strength
    noise_sigma: float = NOISE_OPTIMAL # SR noise amplitude

    # ── Entropy Bounds (safety gate) ──────────────
    h_min: float = H_MIN_DEFAULT       # Below = too rigid (φ⁻² ≈ 0.382)
    h_max: float = H_MAX_DEFAULT       # Above = too chaotic (z_c ≈ 0.866)
    h_target: float = H_OPTIMAL        # Optimal entropy (0.7)

    # ── Heartbeat Parameters (oscillation) ────────
    pi_gamma_ratio: float = PI_GAMMA_OPTIMAL  # Π/γ production-to-decay (τ ≈ 0.618)
    oscillation_freq: float = 1.0      # Heartbeat frequency multiplier

    # ── Sentence-Level Controls ───────────────────
    min_sentence_len: int = 3          # Don't scramble sentences shorter than this (words)
    max_reorder_distance: int = NAV_WINDOW  # ±3 navigation window
    clause_swap_prob: float = 0.3      # Probability of clause-level Cat Map mixing

    # ── Style Parameters ──────────────────────────
    contraction_bias: float = 0.5      # 0=formal ("do not"), 1=casual ("don't")
    punctuation_chaos: float = 0.2     # Em-dash, semicolon, ellipsis injection rate
    vocab_register_shift: float = 0.0  # -1=simpler, +1=denser vocabulary

    # ── Meta ──────────────────────────────────────
    tags: list = field(default_factory=list)
    version: str = "0.1.0"

    # ══════════════════════════════════════════════
    # SERIALIZATION
    # ══════════════════════════════════════════════

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save profile to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> CadenceProfile:
        """Create from dictionary, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> CadenceProfile:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str | Path) -> CadenceProfile:
        """Load profile from a JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ══════════════════════════════════════════════
    # PRESET MANAGEMENT
    # ══════════════════════════════════════════════

    @classmethod
    def list_presets(cls, directory: str | Path = None) -> List[CadenceProfile]:
        """
        List all available presets in a directory.

        Args:
            directory: Path to preset directory. If None, uses the
                       package's built-in presets/ folder.

        Returns:
            List of loaded CadenceProfile objects.
        """
        if directory is None:
            directory = Path(__file__).parent / "presets"
        directory = Path(directory)

        profiles = []
        if directory.exists():
            for json_file in sorted(directory.glob("*.json")):
                try:
                    profiles.append(cls.load(json_file))
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"  Warning: skipping {json_file.name}: {e}")
        return profiles

    @classmethod
    def find_preset(cls, name: str, directory: str | Path = None) -> Optional[CadenceProfile]:
        """
        Find a preset by name (case-insensitive).

        Searches both built-in and custom preset directories.
        """
        for profile in cls.list_presets(directory):
            if profile.name.lower() == name.lower():
                return profile
        return None

    # ══════════════════════════════════════════════
    # BLENDING (the plural superpower)
    # ══════════════════════════════════════════════

    @classmethod
    def blend(
        cls,
        profile_a: CadenceProfile,
        profile_b: CadenceProfile,
        weight: float = 0.5,
        name: Optional[str] = None,
    ) -> CadenceProfile:
        """
        Blend two profiles by linear interpolation of numeric parameters.

        This is the core plural dynamics operation: smooth morphing between
        personas rather than hard switching. weight=0.0 is pure A, weight=1.0
        is pure B, weight=0.5 is the midpoint.

        Non-numeric fields (name, description, tags, cat_matrix) come from
        the dominant profile (weight < 0.5 → A, else → B).

        Connection to framework:
          blend() IS geodesic interpolation on the Kuramoto torus.
          The resulting profile is a valid point in parameter space.

        Args:
            profile_a: First profile (weight=0.0 → pure A)
            profile_b: Second profile (weight=1.0 → pure B)
            weight: Interpolation weight [0.0, 1.0]
            name: Optional name for blended profile

        Returns:
            New CadenceProfile with interpolated parameters.
        """
        weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]
        w_a = 1.0 - weight
        w_b = weight

        # Determine which profile is dominant for non-numeric fields
        dominant = profile_a if weight < 0.5 else profile_b
        secondary = profile_b if weight < 0.5 else profile_a

        # Numeric fields to interpolate
        numeric_fields = {
            'cat_intensity', 'target_r', 'coupling_k', 'noise_sigma',
            'h_min', 'h_max', 'h_target',
            'pi_gamma_ratio', 'oscillation_freq',
            'clause_swap_prob', 'contraction_bias',
            'punctuation_chaos', 'vocab_register_shift',
        }

        # Integer fields to interpolate (round result)
        int_fields = {'min_sentence_len', 'max_reorder_distance'}

        blended_dict = {}
        for f in fields(cls):
            if f.name in numeric_fields:
                val_a = getattr(profile_a, f.name)
                val_b = getattr(profile_b, f.name)
                blended_dict[f.name] = w_a * val_a + w_b * val_b
            elif f.name in int_fields:
                val_a = getattr(profile_a, f.name)
                val_b = getattr(profile_b, f.name)
                blended_dict[f.name] = round(w_a * val_a + w_b * val_b)
            elif f.name == 'name':
                blended_dict['name'] = name or f"{profile_a.name}×{profile_b.name}"
            elif f.name == 'description':
                blended_dict['description'] = (
                    f"Blend: {w_a:.0%} {profile_a.name} + {w_b:.0%} {profile_b.name}"
                )
            elif f.name == 'tags':
                # Merge tags from both, deduplicate
                all_tags = list(dict.fromkeys(profile_a.tags + profile_b.tags))
                blended_dict['tags'] = all_tags
            elif f.name == 'golden_scaling':
                blended_dict['golden_scaling'] = dominant.golden_scaling
            elif f.name == 'cat_matrix':
                blended_dict['cat_matrix'] = dominant.cat_matrix
            elif f.name == 'version':
                blended_dict['version'] = dominant.version

        return cls(**blended_dict)

    # ══════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════

    def summary(self) -> str:
        """Human-readable summary of this profile."""
        lines = [
            f"╔══ {self.name} (v{self.version}) ══╗",
            f"║ {self.description}",
            f"╠══ Scrambling ═══════════════════╣",
            f"║ Cat intensity:    {self.cat_intensity:.2f}",
            f"║ Clause swap prob: {self.clause_swap_prob:.2f}",
            f"║ Reorder window:   ±{self.max_reorder_distance}",
            f"╠══ Synchronization ═════════════╣",
            f"║ Target r:         {self.target_r:.2f}",
            f"║ Coupling K:       {self.coupling_k:.4f}",
            f"║ Noise σ:          {self.noise_sigma:.2f}",
            f"╠══ Entropy Bounds ═══════════════╣",
            f"║ H range:          [{self.h_min:.3f}, {self.h_max:.3f}]",
            f"║ H target:         {self.h_target:.2f}",
            f"║ Π/γ target:       {self.pi_gamma_ratio:.3f}",
            f"╠══ Style ════════════════════════╣",
            f"║ Contraction bias: {self.contraction_bias:.2f}",
            f"║ Punct chaos:      {self.punctuation_chaos:.2f}",
            f"║ Vocab shift:      {self.vocab_register_shift:+.2f}",
            f"╠══ Tags ═════════════════════════╣",
            f"║ {', '.join(self.tags) if self.tags else '(none)'}",
            f"╚════════════════════════════════╝",
        ]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"CadenceProfile(name='{self.name}', "
            f"cat={self.cat_intensity:.2f}, "
            f"r={self.target_r:.2f}, "
            f"H={self.h_target:.2f})"
        )
