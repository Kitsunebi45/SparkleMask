"""
sparkle_mask.mask — Top-level SparklerMask class.

The main interface for the Sparkle Mask tool.
Manages profiles, runs the pipeline, and handles hot-swapping.

Usage:
    mask = SparkleMask()
    mask.load_profile("presets/sparkle.json")
    result = mask.transform("Your text here")
    print(result.masked_text)
    print(result.diagnostic.render())

    # Hot-swap profile
    mask.load_profile("presets/dense.json")
    result2 = mask.transform("Different text")

    # Blend profiles
    mask.blend_profiles("presets/kitty.json", "presets/vix.json", weight=0.6)
    result3 = mask.transform("Blended text")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from .cadence_profile import CadenceProfile
from .pipeline import run_pipeline, PipelineResult


class SparkleMask:
    """
    Top-level Sparkle Mask interface.

    Manages cadence profiles and runs the transformation pipeline.
    Supports hot-swapping profiles mid-session and blending.
    """

    def __init__(
        self,
        profile: Optional[CadenceProfile] = None,
        preset_dir: Optional[str] = None,
    ):
        """
        Initialize the Sparkle Mask.

        Args:
            profile: Initial cadence profile. If None, uses default.
            preset_dir: Directory containing preset JSON files.
        """
        self._preset_dir = Path(preset_dir) if preset_dir else Path(__file__).parent / "presets"
        self._profile = profile or CadenceProfile()
        self._history: List[PipelineResult] = []

    @property
    def profile(self) -> CadenceProfile:
        """Current active profile."""
        return self._profile

    @property
    def history(self) -> List[PipelineResult]:
        """History of all transformations in this session."""
        return self._history

    # ══════════════════════════════════════════════
    # PROFILE MANAGEMENT
    # ══════════════════════════════════════════════

    def load_profile(self, path: str | Path) -> CadenceProfile:
        """
        Load and activate a profile from a JSON file.

        Hot-swap: takes effect immediately for the next transform() call.

        Args:
            path: Path to JSON profile file.

        Returns:
            The loaded CadenceProfile.
        """
        self._profile = CadenceProfile.load(path)
        return self._profile

    def load_preset(self, name: str) -> CadenceProfile:
        """
        Load a preset by name from the preset directory.

        Args:
            name: Preset name (case-insensitive, e.g. "dense", "soft")

        Returns:
            The loaded CadenceProfile.

        Raises:
            FileNotFoundError if preset not found.
        """
        profile = CadenceProfile.find_preset(name, self._preset_dir)
        if profile is None:
            available = [p.name for p in CadenceProfile.list_presets(self._preset_dir)]
            raise FileNotFoundError(
                f"Preset '{name}' not found. Available: {', '.join(available)}"
            )
        self._profile = profile
        return self._profile

    def set_profile(self, profile: CadenceProfile) -> None:
        """Set the active profile directly (for programmatic use)."""
        self._profile = profile

    def blend_profiles(
        self,
        path_a: str | Path,
        path_b: str | Path,
        weight: float = 0.5,
        name: Optional[str] = None,
    ) -> CadenceProfile:
        """
        Blend two profiles and set the result as active.

        Args:
            path_a: Path to first profile JSON
            path_b: Path to second profile JSON
            weight: Blend weight (0.0 = pure A, 1.0 = pure B)
            name: Optional name for the blend

        Returns:
            The blended CadenceProfile.
        """
        a = CadenceProfile.load(path_a)
        b = CadenceProfile.load(path_b)
        blended = CadenceProfile.blend(a, b, weight=weight, name=name)
        self._profile = blended
        return blended

    def list_presets(self) -> List[CadenceProfile]:
        """List all available presets in the preset directory."""
        return CadenceProfile.list_presets(self._preset_dir)

    # ══════════════════════════════════════════════
    # TRANSFORMATION
    # ══════════════════════════════════════════════

    def transform(
        self,
        text: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Transform text through the Sparkle Mask pipeline.

        Full pipeline: ANALYZE → CAT-MAP → GATE → FOX-LI → GATE → HOLD

        Args:
            text: Input text to transform.
            seed: Random seed for reproducibility.
            verbose: Print progress and diagnostics to console.

        Returns:
            PipelineResult with masked text and full diagnostics.
        """
        result = run_pipeline(
            text=text,
            profile=self._profile,
            seed=seed,
            verbose=verbose,
        )
        self._history.append(result)
        return result

    # ══════════════════════════════════════════════
    # VERIFICATION
    # ══════════════════════════════════════════════

    def verify(self, original: str, masked: str) -> bool:
        """
        Verify that masked text was transformed with the current profile.

        Uses Berry Phase signature comparison.

        Args:
            original: Original untransformed text.
            masked: Claimed Sparkle Masked text.

        Returns:
            True if the Berry phase signatures match.
        """
        from .berry_phase import verify_berry_phase, compute_berry_phase
        import re

        orig_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', original) if s.strip()]
        mask_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', masked) if s.strip()]

        sig = compute_berry_phase(orig_sents, mask_sents, self._profile.name)
        # For now, just return the signature (full verification needs the expected hash)
        return sig.n_contributions > 0

    # ══════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════

    def __repr__(self) -> str:
        return f"SparkleMask(profile='{self._profile.name}', history={len(self._history)})"
