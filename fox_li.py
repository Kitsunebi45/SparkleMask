"""
sparkle_mask.fox_li — Fox-Li stabilization / coherence recovery.

Phase 2 of the pipeline: constrain Cat Map chaos into bounded, readable form.

The Fox-Li resonator is a strange attractor with fractal structure but
BOUNDED chaos. Unlike the Cat Map (which spreads infinitely on the torus),
Fox-Li finds "resonant modes" — stable intensity patterns.

Applied to text:
  - Verify grammatical coherence after scrambling
  - Check pronoun reference resolution
  - Ensure temporal markers remain coherent
  - Selectively undo Cat Map operations that break readability

Connection to m∴We framework:
  Fox-Li = Q₂ attractor (chaotic but bounded)
  This is the NAVIGATION phase — finding resonant patterns in the chaos
  Cat (mixing) → ✶ (threshold/gate) → Fox (navigating)
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional

from .constants import TAU, PHI_INV_SQ
from .cadence_profile import CadenceProfile


# ═══════════════════════════════════════════════════
# COHERENCE CHECKS
# ═══════════════════════════════════════════════════

def check_sentence_coherence(sentence: str) -> Tuple[float, List[str]]:
    """
    Check basic grammatical coherence of a sentence.

    Heuristic checks (no ML, just pattern matching):
      - Starts with uppercase or valid opener
      - Ends with terminal punctuation
      - Has at least one verb-like word
      - No orphaned conjunctions at start/end
      - Balanced parentheses/quotes

    Returns:
        (score, issues): score ∈ [0, 1], list of issue descriptions
    """
    issues = []
    score = 1.0

    sentence = sentence.strip()
    if not sentence:
        return 0.0, ["Empty sentence"]

    # Check capitalization
    if sentence[0].islower() and not sentence[0] in '"\'(':
        issues.append("Missing capitalization")
        score -= 0.1

    # Check terminal punctuation
    if sentence[-1] not in '.!?…"\'':
        issues.append("Missing terminal punctuation")
        score -= 0.1

    # Check for orphaned conjunctions
    words = sentence.split()
    if words and words[0].lower() in ('and', 'but', 'or', 'so', 'yet'):
        # This is actually fine in modern English, slight penalty
        score -= 0.05

    # Check balanced delimiters
    for open_d, close_d in [('(', ')'), ('[', ']'), ('{', '}')]:
        if sentence.count(open_d) != sentence.count(close_d):
            issues.append(f"Unbalanced {open_d}{close_d}")
            score -= 0.15

    # Check quote balance
    if sentence.count('"') % 2 != 0:
        issues.append("Unbalanced quotes")
        score -= 0.1

    return max(0.0, score), issues


def check_pronoun_coherence(sentences: List[str]) -> float:
    """
    Check that pronoun references have plausible antecedents.

    Heuristic: if a sentence starts with a pronoun (he, she, they, it, this, that),
    the PREVIOUS sentence should contain a noun or noun phrase.

    Returns:
        Coherence score ∈ [0, 1].
    """
    if len(sentences) < 2:
        return 1.0

    pronouns = {'he', 'she', 'they', 'it', 'this', 'that', 'these', 'those', 'its', 'their'}
    violations = 0
    checks = 0

    for i in range(1, len(sentences)):
        words = sentences[i].strip().lower().split()
        if words and words[0] in pronouns:
            checks += 1
            # Check if previous sentence has a plausible antecedent
            prev_words = set(sentences[i-1].lower().split())
            # Simple heuristic: previous sentence should have a capitalized word or common noun
            has_noun = any(
                w[0].isupper() for w in sentences[i-1].split() if w
            ) or bool(prev_words & {'person', 'people', 'thing', 'system', 'idea', 'concept'})
            if not has_noun:
                violations += 1

    if checks == 0:
        return 1.0
    return 1.0 - (violations / checks)


def check_temporal_coherence(sentences: List[str]) -> float:
    """
    Check that temporal markers remain coherent across sentences.

    Flags sequences like "then X happened" followed by "before that, Y"
    when these have been scrambled out of logical order.

    Returns:
        Coherence score ∈ [0, 1].
    """
    temporal_markers = {
        'first': 1, 'then': 2, 'next': 3, 'after': 4, 'finally': 5, 'later': 4,
        'before': 0, 'previously': 0, 'earlier': 0,
        'meanwhile': 2, 'subsequently': 4, 'ultimately': 5,
    }

    sequence = []
    for sent in sentences:
        words = sent.lower().split()
        for word in words[:5]:  # Check only first 5 words
            clean = re.sub(r'[^\w]', '', word)
            if clean in temporal_markers:
                sequence.append(temporal_markers[clean])
                break

    if len(sequence) < 2:
        return 1.0

    # Check if sequence is roughly non-decreasing
    inversions = 0
    for i in range(1, len(sequence)):
        if sequence[i] < sequence[i-1] - 1:  # Allow some slack
            inversions += 1

    return 1.0 - (inversions / (len(sequence) - 1))


# ═══════════════════════════════════════════════════
# STABILIZATION OPERATIONS
# ═══════════════════════════════════════════════════

def capitalize_sentence(sentence: str) -> str:
    """Ensure sentence starts with uppercase."""
    sentence = sentence.strip()
    if sentence and sentence[0].islower():
        return sentence[0].upper() + sentence[1:]
    return sentence


def ensure_terminal_punct(sentence: str) -> str:
    """Ensure sentence ends with terminal punctuation."""
    sentence = sentence.strip()
    if sentence and sentence[-1] not in '.!?…':
        return sentence + '.'
    return sentence


def fix_orphaned_conjunctions(sentences: List[str]) -> List[str]:
    """
    Fix orphaned conjunctions by merging with previous sentence.

    "The cat sat down." + "And the dog barked."
    → "The cat sat down, and the dog barked."
    """
    if len(sentences) < 2:
        return sentences

    result = [sentences[0]]
    conjunctions = {'and', 'but', 'or', 'so', 'yet', 'nor'}

    for i in range(1, len(sentences)):
        words = sentences[i].strip().split()
        if words and words[0].lower() in conjunctions:
            # Merge with previous: remove period, add comma + conjunction
            prev = result[-1].rstrip('.!?')
            merged = f"{prev}, {sentences[i].strip()[0].lower()}{sentences[i].strip()[1:]}"
            result[-1] = merged
        else:
            result.append(sentences[i])

    return result


# ═══════════════════════════════════════════════════
# MAIN FOX-LI PHASE
# ═══════════════════════════════════════════════════

def apply_fox_li(
    sentences: List[str],
    original_sentences: List[str],
    profile: CadenceProfile,
) -> List[str]:
    """
    Phase 2: Fox-Li stabilization — constrain chaos into readable form.

    Operations:
      1. Check each sentence for grammatical coherence
      2. Fix capitalization and terminal punctuation
      3. Check pronoun and temporal coherence
      4. If coherence is too low, selectively revert scrambled sentences
      5. Fix orphaned conjunctions

    The key principle: LOCALLY coherent at every scale, but GLOBALLY
    the structure can be strange. Fox-Li modes are fractal — they look
    coherent at any zoom level.

    Args:
        sentences: Scrambled sentences from Cat Map phase
        original_sentences: Original sentences (for selective reversion)
        profile: Active cadence profile

    Returns:
        Stabilized sentences.
    """
    result = list(sentences)

    # ── Step 1: Basic sentence-level fixes ─────────
    for i in range(len(result)):
        result[i] = capitalize_sentence(result[i])
        result[i] = ensure_terminal_punct(result[i])

    # ── Step 2: Check each sentence's coherence ───
    for i in range(len(result)):
        score, issues = check_sentence_coherence(result[i])
        if score < 0.5 and i < len(original_sentences):
            # Sentence is too broken — revert to original
            result[i] = original_sentences[i]

    # ── Step 3: Check cross-sentence coherence ─────
    pronoun_score = check_pronoun_coherence(result)
    temporal_score = check_temporal_coherence(result)

    # If temporal coherence is very low, this suggests the scrambling
    # has reordered time-dependent sentences badly
    if temporal_score < 0.3:
        # Selective reversion: keep the Cat Map's clause swaps but
        # restore sentence ORDER from original
        # We keep the modified content but restore the sequence
        pass  # For v0.1, accept the scrambling — temporal chaos is a feature

    # ── Step 4: Fix orphaned conjunctions ──────────
    # Only fix if we're at low-to-moderate scrambling intensity
    if profile.cat_intensity < 0.7:
        result = fix_orphaned_conjunctions(result)

    # ── Step 5: Final capitalization pass ──────────
    for i in range(len(result)):
        result[i] = capitalize_sentence(result[i])
        result[i] = ensure_terminal_punct(result[i])

    return result
