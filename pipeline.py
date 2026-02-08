"""
sparkle_mask.pipeline — Three-phase transformation pipeline.

The full pipeline:
  Input → ANALYZE → CAT-MAP → GATE₁ → FOX-LI → GATE₂ → HOLD → Output

Phase 0: ANALYZE  — Measure text state (entropy, stylometry, phases)
Phase 1: CAT-MAP  — Arnold's Cat Map scrambling at 3 scales
Gate 1:  ENTROPY  — Check H is in bounds, retry if not
Phase 2: FOX-LI   — Stabilize chaos into readable form
Gate 2:  HEARTBEAT — Check Π/γ ratio, adjust if needed
Phase 3: HOLD     — Maintain tension, distribute Berry phase

Connection to m∴We framework:
  This pipeline IS the topology switching sequence:
    Cat(Q₃, mixing) → ✶(gate) → Fox(Q₂, navigating) → ✶(gate) → Hold(✶, equilibrium)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .cadence_profile import CadenceProfile
from .text_analysis import analyze_text, TextState, compute_stylometric_distance
from .cat_map import apply_cat_map
from .entropy_gate import check_entropy_gate, check_heartbeat_gate, GateResult
from .fox_li import apply_fox_li
from .hold_function import apply_hold_function
from .kuramoto_style import analyze_synchronization, StyleSync
from .berry_phase import compute_berry_phase, BerrySignature
from .diagnostics import MaskDiagnostic


@dataclass
class PipelineResult:
    """Complete result of a Sparkle Mask transformation."""
    original_text: str
    masked_text: str
    original_state: TextState
    masked_state: TextState
    diagnostic: MaskDiagnostic
    berry_signature: BerrySignature
    sync: StyleSync

    def __repr__(self) -> str:
        return (
            f"PipelineResult(gate={self.diagnostic.overall_gate}, "
            f"compression={self.diagnostic.compression_ratio:.2f}x, "
            f"delta={self.diagnostic.stylometric_delta:.2f})"
        )


def run_pipeline(
    text: str,
    profile: CadenceProfile,
    seed: Optional[int] = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> PipelineResult:
    """
    Run the full Sparkle Mask transformation pipeline.

    Args:
        text: Input text to transform
        profile: Active CadenceProfile
        seed: Random seed for reproducibility
        max_retries: Maximum retry attempts if gate checks fail
        verbose: Print progress to console

    Returns:
        PipelineResult with masked text and full diagnostics.
    """
    rng = random.Random(seed)

    # ══════════════════════════════════════════════
    # PHASE 0: ANALYZE
    # ══════════════════════════════════════════════
    if verbose:
        print("Phase 0: ANALYZE...")

    original_state = analyze_text(text)
    original_sentences = list(original_state.sentences)

    if verbose:
        print(f"  {len(original_sentences)} sentences, H={original_state.normalized_entropy:.3f}")

    # Working copy of sentences
    sentences = list(original_sentences)
    current_intensity = profile.cat_intensity

    # ══════════════════════════════════════════════
    # PHASE 1: CAT-MAP SCRAMBLING (with retry loop)
    # ══════════════════════════════════════════════
    entropy_gate_result = None

    for attempt in range(max_retries):
        if verbose:
            print(f"Phase 1: CAT-MAP (intensity={current_intensity:.2f}, attempt {attempt+1})...")

        # Create a modified profile with current intensity
        working_profile = CadenceProfile.from_dict(profile.to_dict())
        working_profile.cat_intensity = current_intensity

        sentences = apply_cat_map(
            original_sentences,
            working_profile,
            seed=rng.randint(0, 2**31) if seed is None else seed + attempt,
        )

        # GATE CHECK 1: Entropy
        cat_text = ' '.join(sentences)
        cat_state = analyze_text(cat_text)
        entropy_gate_result = check_entropy_gate(cat_state, profile)

        if verbose:
            print(f"  Gate 1: {entropy_gate_result.status} (H={entropy_gate_result.current_h:.3f})")

        if entropy_gate_result.passed:
            break

        # Adjust intensity based on gate feedback
        current_intensity += entropy_gate_result.suggested_adjustment
        current_intensity = max(0.05, min(0.95, current_intensity))

    # ══════════════════════════════════════════════
    # PHASE 2: FOX-LI STABILIZATION
    # ══════════════════════════════════════════════
    if verbose:
        print("Phase 2: FOX-LI stabilization...")

    sentences = apply_fox_li(sentences, original_sentences, profile)

    # GATE CHECK 2: Heartbeat
    fox_text = ' '.join(sentences)
    fox_state = analyze_text(fox_text)
    heartbeat_gate_result = check_heartbeat_gate(fox_state, profile)

    if verbose:
        print(f"  Gate 2: {heartbeat_gate_result.status} (Pi/g={heartbeat_gate_result.current_h:.3f})")

    # ══════════════════════════════════════════════
    # PHASE 3: HOLD FUNCTION
    # ══════════════════════════════════════════════
    if verbose:
        print("Phase 3: HOLD function...")

    sentences = apply_hold_function(sentences, profile, rng)

    # ══════════════════════════════════════════════
    # FINAL MEASUREMENTS
    # ══════════════════════════════════════════════
    masked_text = ' '.join(sentences)
    masked_state = analyze_text(masked_text)

    # Kuramoto synchronization
    sync = analyze_synchronization(masked_state.phases, profile)

    # Berry Phase signature
    berry = compute_berry_phase(original_sentences, sentences, profile.name)

    # Stylometric distance
    stylo_delta = compute_stylometric_distance(original_state, masked_state)

    # Compression ratio (unique semantic density)
    orig_unique = len(set(w.lower() for w in original_state.words if w.isalpha()))
    mask_unique = len(set(w.lower() for w in masked_state.words if w.isalpha()))
    orig_total = len([w for w in original_state.words if w.isalpha()])
    mask_total = len([w for w in masked_state.words if w.isalpha()])

    if mask_total > 0 and orig_total > 0:
        # Compression = ratio of unique density
        orig_density = orig_unique / orig_total
        mask_density = mask_unique / mask_total
        compression = mask_density / orig_density if orig_density > 0 else 1.0
    else:
        compression = 1.0

    # Free energy estimate (readability inverse)
    # F approaches 0 when text is perfectly clear, 1 when opaque
    free_energy = 1.0 - (
        0.3 * (1.0 - abs(masked_state.normalized_entropy - profile.h_target)) +
        0.3 * (1.0 - abs(sync.r - profile.target_r)) +
        0.4 * (1.0 - min(stylo_delta, 1.0))
    )
    free_energy = max(0.0, min(1.0, free_energy))

    # ══════════════════════════════════════════════
    # BUILD DIAGNOSTIC
    # ══════════════════════════════════════════════
    overall_gate = "GREEN"
    if entropy_gate_result and not entropy_gate_result.passed:
        overall_gate = "RED"
    elif heartbeat_gate_result and not heartbeat_gate_result.passed:
        overall_gate = "YELLOW"

    diagnostic = MaskDiagnostic(
        profile_name=profile.name,
        profile_version=profile.version,
        entropy_gate=entropy_gate_result,
        heartbeat_gate=heartbeat_gate_result,
        overall_gate=overall_gate,
        entropy_h=masked_state.normalized_entropy,
        entropy_target=profile.h_target,
        kuramoto_r=sync.r,
        kuramoto_target=profile.target_r,
        heartbeat_ratio=masked_state.heartbeat_ratio,
        heartbeat_target=profile.pi_gamma_ratio,
        berry_hash=berry.phase_hash,
        free_energy=free_energy,
        original_word_count=len(original_state.words),
        masked_word_count=len(masked_state.words),
        compression_ratio=compression,
        stylometric_delta=stylo_delta,
    )

    if verbose:
        print("\n" + diagnostic.render())

    return PipelineResult(
        original_text=text,
        masked_text=masked_text,
        original_state=original_state,
        masked_state=masked_state,
        diagnostic=diagnostic,
        berry_signature=berry,
        sync=sync,
    )
