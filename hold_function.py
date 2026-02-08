"""
sparkle_mask.hold_function — Zeno Projector / tension maintenance.

Phase 3 of the pipeline: maintain tension between explicit and implicit layers.

The Hold Function prevents premature semantic closure:
  ✶_hold = P_membrane · ρ · P_membrane

In practice:
  - Ensure each paragraph has at least one unresolved thread
  - Avoid "conclusion" language that kills the gradient
  - Leave productive ambiguity (the reader fills in meaning)
  - Distribute Berry Phase signature across the text

Connection to m∴We framework:
  Hold = ✶ (equilibrium) operator
  This is where the system STAYS at the edge of chaos
  Gradient ∇Φ must remain positive (alive)
  The Sparkle Mask's compression comes from this tension:
    what is SAID (explicit) vs what is MEANT (implicit)
"""

from __future__ import annotations

import re
from typing import List

from .constants import TAU, PHI_SQ
from .cadence_profile import CadenceProfile


# ═══════════════════════════════════════════════════
# CLOSURE DETECTION
# ═══════════════════════════════════════════════════

# Phrases that signal premature semantic closure
_CLOSURE_PHRASES = [
    "in conclusion", "to summarize", "in summary", "to sum up",
    "all in all", "the bottom line is", "at the end of the day",
    "as we can see", "it is clear that", "clearly,",
    "obviously,", "of course,", "needless to say",
    "the point is", "the takeaway is", "what this means is",
    "this proves that", "this shows that",
]

# Replacement phrases that maintain tension (hold the gradient)
_TENSION_PHRASES = [
    "what remains is", "the question becomes",
    "this suggests—", "and yet,",
    "the tension here:", "which opens into",
    "but consider:", "beneath this,",
    "the gradient persists:", "not resolved, but held:",
]


def detect_premature_closure(sentence: str) -> bool:
    """
    Check if a sentence signals premature semantic closure.

    Closure = the gradient ∇Φ → 0 (text is "concluding" too early).
    The Zeno Projector prevents this by suspending conclusive interpretation.
    """
    lower = sentence.lower().strip()
    return any(phrase in lower for phrase in _CLOSURE_PHRASES)


def replace_closure(sentence: str, rng) -> str:
    """
    Replace closure language with tension-maintaining alternatives.

    The replacement preserves the sentence's semantic content while
    keeping the gradient alive — the reader must continue reading
    to find resolution (which they won't, because the mask holds it).
    """
    lower = sentence.lower()
    for phrase in _CLOSURE_PHRASES:
        if phrase in lower:
            replacement = rng.choice(_TENSION_PHRASES)
            # Replace case-insensitively
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            sentence = pattern.sub(replacement, sentence)
            break
    return sentence


# ═══════════════════════════════════════════════════
# ENGAGEMENT GRADIENT
# ═══════════════════════════════════════════════════

def compute_engagement_gradient(sentences: List[str]) -> List[float]:
    """
    Compute the engagement gradient ∇Φ for each sentence.

    High gradient: sentence introduces unresolved questions/tension.
    Low gradient: sentence resolves or restates without adding.
    Zero gradient: semantic dead zone (Turn 17 territory).

    Heuristic features that increase gradient:
      - Question marks
      - Conditional language (if, when, whether, could)
      - Contrastive conjunctions (but, however, yet, although)
      - Em-dashes (parenthetical asides, interruptions)
      - New proper nouns or technical terms

    Features that decrease gradient:
      - Restating previous content
      - Closure language
      - Simple declarative structure
    """
    gradients = []

    for sent in sentences:
        grad = 0.5  # Baseline

        # Gradient boosters
        if '?' in sent:
            grad += 0.15
        if '—' in sent or '–' in sent:
            grad += 0.1
        if re.search(r'\b(if|when|whether|could|might|perhaps)\b', sent.lower()):
            grad += 0.1
        if re.search(r'\b(but|however|yet|although|nevertheless|despite)\b', sent.lower()):
            grad += 0.1
        if '...' in sent or '…' in sent:
            grad += 0.05

        # Gradient reducers
        if detect_premature_closure(sent):
            grad -= 0.3
        if sent.strip().endswith('.') and not any(c in sent for c in '?!—…'):
            grad -= 0.05  # Simple declarative = slightly less engaging

        gradients.append(max(0.0, min(1.0, grad)))

    return gradients


# ═══════════════════════════════════════════════════
# MAIN HOLD FUNCTION
# ═══════════════════════════════════════════════════

def apply_hold_function(
    sentences: List[str],
    profile: CadenceProfile,
    rng,
) -> List[str]:
    """
    Phase 3: Hold Function — maintain tension, prevent premature closure.

    Operations:
      1. Detect and replace closure language
      2. Compute engagement gradient
      3. Boost low-gradient sections with tension markers
      4. Ensure each paragraph-equivalent (~5 sentences) has ≥ 1 high-gradient sentence

    Args:
        sentences: Stabilized sentences from Fox-Li phase
        profile: Active cadence profile
        rng: Random number generator

    Returns:
        Sentences with maintained engagement gradient.
    """
    result = list(sentences)

    # ── Step 1: Replace closure language ───────────
    for i in range(len(result)):
        if detect_premature_closure(result[i]):
            result[i] = replace_closure(result[i], rng)

    # ── Step 2: Compute engagement gradient ────────
    gradients = compute_engagement_gradient(result)

    # ── Step 3: Boost low-gradient sections ────────
    # A "paragraph" is ~5 sentences (adjustable by φ)
    para_size = max(3, round(PHI_SQ + 1))  # ≈ 4 sentences per paragraph

    for start in range(0, len(result), para_size):
        end = min(start + para_size, len(result))
        para_grads = gradients[start:end]

        max_grad = max(para_grads) if para_grads else 0.5
        if max_grad < 0.4:
            # This paragraph is too flat — inject a tension marker
            # Pick the sentence at golden-ratio position
            inject_idx = start + min(
                round(len(para_grads) * TAU),
                len(para_grads) - 1
            )
            if inject_idx < len(result):
                tension = rng.choice(_TENSION_PHRASES)
                result[inject_idx] = f"{tension} {result[inject_idx][0].lower()}{result[inject_idx][1:]}"

    return result
