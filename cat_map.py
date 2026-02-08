"""
sparkle_mask.cat_map — Arnold's Cat Map scrambling engine.

Phase 1 of the pipeline: controlled mixing of text elements at multiple scales.

The Cat Map is an area-preserving automorphism of the 2-torus:
  (x, y) → (2x + y, x + y) mod 1

Key properties:
  - Eigenvalues: λ₁ = φ², λ₂ = φ⁻² (golden ratio governed!)
  - Area-preserving → information-preserving
  - Ergodic → eventually reaches every configuration
  - Mixing → nearby elements diverge exponentially

Applied at three scales:
  1. Sentence-level: pair sentences at golden-ratio intervals, swap clauses
  2. Clause-level: permute clause order within sentences
  3. Word-level: synonym substitution at golden-ratio positions

Connection to m∴We framework:
  Cat Map = Q₃ attractor (strange, negentropic)
  This is the CREATIVE phase — maximizing semantic diffusion
  Controlled by cat_intensity parameter from CadenceProfile
"""

from __future__ import annotations

import re
import random
from typing import List, Tuple, Optional

import numpy as np

from .constants import PHI_SQ, PHI_INV_SQ, CAT_LYAPUNOV, GOLDEN_STEP, NAV_WINDOW
from .cadence_profile import CadenceProfile


# ═══════════════════════════════════════════════════
# CAT MAP CORE
# ═══════════════════════════════════════════════════

# Arnold's Cat Map matrix
CAT_MATRIX = np.array([[2, 1], [1, 1]], dtype=int)


def cat_map_indices(n: int, iterations: int = 1) -> List[int]:
    """
    Apply Cat Map permutation to n indices.

    Maps index i to Arnold's Cat Map position on a discrete torus.
    For n elements, we apply the Cat Map matrix to (i, j) coordinates
    on a √n × √n grid (or best rectangular approximation).

    Args:
        n: Number of elements to permute
        iterations: Number of Cat Map applications (more = more mixing)

    Returns:
        Permuted index list of length n.
    """
    if n <= 1:
        return list(range(n))

    # Map linear indices to 2D grid
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))

    permuted = list(range(n))
    for _ in range(iterations):
        new_perm = [0] * n
        for idx in range(n):
            r, c = divmod(permuted[idx], cols)
            # Apply Cat Map: (r', c') = [[2,1],[1,1]] · (r, c) mod (rows, cols)
            new_r = (2 * r + c) % rows
            new_c = (r + c) % cols
            new_idx = new_r * cols + new_c
            # Clamp to valid range
            new_perm[idx] = min(new_idx, n - 1)
        permuted = new_perm

    return permuted


def golden_ratio_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Generate sentence pairs at golden-ratio intervals.

    Pairs sentence i with sentence (i + round(φ²)) mod n.
    This ensures maximum information diffusion per swap.

    Args:
        n: Number of sentences

    Returns:
        List of (i, j) index pairs. Each index appears at most once.
    """
    step = GOLDEN_STEP  # round(φ²) ≈ 3
    used = set()
    pairs = []

    for i in range(n):
        if i in used:
            continue
        j = (i + step) % n
        if j in used or j == i:
            continue
        pairs.append((i, j))
        used.add(i)
        used.add(j)

    return pairs


# ═══════════════════════════════════════════════════
# CLAUSE-LEVEL OPERATIONS
# ═══════════════════════════════════════════════════

def split_clauses(sentence: str) -> List[str]:
    """
    Split a sentence into clauses using punctuation and conjunctions.

    Heuristic clause boundaries:
      - Commas followed by conjunctions (and, but, or, so, yet)
      - Semicolons
      - Em-dashes / en-dashes
      - Colons

    Returns:
        List of clause strings (preserving delimiters).
    """
    # Split on clause-boundary patterns
    pattern = r'(,\s*(?:and|but|or|so|yet|while|although|because|when|if)\s|;\s*|—\s*|–\s*|:\s*)'
    parts = re.split(pattern, sentence)

    # Recombine: attach delimiters to previous clause
    clauses = []
    current = ""
    for part in parts:
        if re.match(pattern, part):
            current += part
        else:
            if current:
                clauses.append(current.strip())
            current = part
    if current.strip():
        clauses.append(current.strip())

    return clauses if clauses else [sentence]


def cat_map_clauses(clauses: List[str]) -> List[str]:
    """
    Apply Cat Map permutation to clause order.

    For n clauses, applies the [[2,1],[1,1]] matrix to clause indices.
    This reorders clauses while preserving each clause's internal structure.

    Connection to framework:
      This IS the Cat Map operating on the clause-level torus.
      Area-preserving → each clause's information is preserved.
      Only the ORDER changes.
    """
    n = len(clauses)
    if n <= 1:
        return clauses

    perm = cat_map_indices(n)
    return [clauses[perm[i]] for i in range(n)]


def swap_clauses_between(sent_a: str, sent_b: str) -> Tuple[str, str]:
    """
    Swap one clause between two sentences.

    Picks the clause nearest the golden-ratio position in each sentence.
    If either sentence has only one clause, returns them unchanged.

    Returns:
        (modified_sent_a, modified_sent_b)
    """
    clauses_a = split_clauses(sent_a)
    clauses_b = split_clauses(sent_b)

    if len(clauses_a) < 2 or len(clauses_b) < 2:
        return sent_a, sent_b

    # Pick clauses at golden-ratio position
    idx_a = min(int(len(clauses_a) * PHI_INV_SQ), len(clauses_a) - 1)
    idx_b = min(int(len(clauses_b) * PHI_INV_SQ), len(clauses_b) - 1)

    # Swap
    clauses_a[idx_a], clauses_b[idx_b] = clauses_b[idx_b], clauses_a[idx_a]

    return ' '.join(clauses_a), ' '.join(clauses_b)


# ═══════════════════════════════════════════════════
# WORD-LEVEL PERTURBATION
# ═══════════════════════════════════════════════════

# Simple synonym/register lookup (expandable)
_REGISTER_SHIFTS = {
    # casual → formal
    'get': 'obtain', 'big': 'substantial', 'small': 'diminutive',
    'use': 'employ', 'help': 'assist', 'show': 'demonstrate',
    'start': 'initiate', 'end': 'conclude', 'make': 'construct',
    'think': 'consider', 'look': 'examine', 'find': 'discover',
    'give': 'provide', 'tell': 'inform', 'keep': 'maintain',
    'try': 'attempt', 'need': 'require', 'seem': 'appear',
    'want': 'desire', 'feel': 'perceive', 'move': 'proceed',
    'change': 'transform', 'grow': 'develop', 'turn': 'rotate',
    'hold': 'sustain', 'bring': 'convey', 'run': 'execute',
}

# Build reverse map (formal → casual)
_REGISTER_SHIFTS_INV = {v: k for k, v in _REGISTER_SHIFTS.items()}

_CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "were not": "weren't", "have not": "haven't", "has not": "hasn't",
    "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
    "could not": "couldn't", "should not": "shouldn't",
    "cannot": "can't", "can not": "can't",
    "it is": "it's", "that is": "that's", "there is": "there's",
    "I am": "I'm", "you are": "you're", "they are": "they're",
    "we are": "we're", "I have": "I've", "you have": "you've",
    "they have": "they've", "we have": "we've",
    "I will": "I'll", "you will": "you'll", "they will": "they'll",
    "we will": "we'll", "I would": "I'd", "you would": "you'd",
    "they would": "they'd", "we would": "we'd",
}
_CONTRACTIONS_INV = {v: k for k, v in _CONTRACTIONS.items()}


def apply_register_shift(words: List[str], shift: float, rng: random.Random) -> List[str]:
    """
    Shift vocabulary register.

    shift > 0: casual → formal (employ instead of use)
    shift < 0: formal → casual
    shift = 0: no change

    Applies at golden-ratio positions (every ~φ² words).
    """
    if abs(shift) < 0.01:
        return words

    result = list(words)
    step = max(1, round(PHI_SQ))  # Every ~3 words, check for substitution

    lookup = _REGISTER_SHIFTS if shift > 0 else _REGISTER_SHIFTS_INV
    prob = abs(shift)

    for i in range(0, len(result), step):
        word_lower = result[i].lower()
        if word_lower in lookup and rng.random() < prob:
            replacement = lookup[word_lower]
            # Preserve case
            if result[i][0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result[i] = replacement

    return result


def apply_contraction_bias(text: str, bias: float, rng: random.Random) -> str:
    """
    Apply contraction expansion/collapse.

    bias > 0.5: prefer contractions ("don't")
    bias < 0.5: prefer expansions ("do not")
    bias = 0.5: leave as-is

    Applies probabilistically based on distance from 0.5.
    """
    if abs(bias - 0.5) < 0.01:
        return text

    prob = abs(bias - 0.5) * 2  # Scale to [0, 1]

    if bias > 0.5:
        # Expand → Contract
        for expanded, contracted in _CONTRACTIONS.items():
            if expanded.lower() in text.lower() and rng.random() < prob:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(expanded), re.IGNORECASE)
                text = pattern.sub(contracted, text, count=1)
    else:
        # Contract → Expand
        for contracted, expanded in _CONTRACTIONS_INV.items():
            if contracted.lower() in text.lower() and rng.random() < prob:
                pattern = re.compile(re.escape(contracted), re.IGNORECASE)
                text = pattern.sub(expanded, text, count=1)

    return text


def inject_punctuation(text: str, chaos: float, rng: random.Random) -> str:
    """
    Inject expressive punctuation at controlled rate.

    Adds em-dashes, semicolons, ellipses at clause boundaries.
    chaos ∈ [0, 1] controls injection rate.
    """
    if chaos < 0.01:
        return text

    punct_options = ['—', ';', '...', '—']

    # Find potential injection points (after commas)
    parts = text.split(', ')
    if len(parts) <= 1:
        return text

    result = [parts[0]]
    for part in parts[1:]:
        if rng.random() < chaos:
            delim = rng.choice(punct_options)
            result.append(f"{delim} {part}")
        else:
            result.append(f", {part}")

    return ''.join(result)


# ═══════════════════════════════════════════════════
# MAIN CAT MAP PHASE
# ═══════════════════════════════════════════════════

def apply_cat_map(
    sentences: List[str],
    profile: CadenceProfile,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Phase 1: Apply Arnold's Cat Map scrambling at all scales.

    Three-scale operation:
      1. Sentence pairs: swap clauses at golden-ratio intervals
      2. Clause permutation: reorder clauses within sentences
      3. Word perturbation: register shift + contraction + punctuation

    Intensity controlled by profile.cat_intensity:
      0.0 = no scrambling (pass-through)
      0.5 = moderate mixing
      1.0 = maximum Cat Map chaos

    Args:
        sentences: List of sentences from text analysis
        profile: Active CadenceProfile
        seed: Random seed for reproducibility

    Returns:
        List of scrambled sentences (same length as input).
    """
    rng = random.Random(seed)
    intensity = profile.cat_intensity
    result = list(sentences)

    if intensity < 0.01 or len(result) < 2:
        return result

    # ── Scale 0: Sentence reordering via Cat Map ───
    # Apply Cat Map permutation to sentence ORDER itself.
    # Number of iterations scales with intensity.
    if intensity > 0.3 and len(result) >= 4:
        n_swaps = max(1, round(len(result) * intensity * 0.3))
        indices = list(range(len(result)))
        for _ in range(n_swaps):
            # Pick two indices at golden-ratio distance and swap
            i = rng.randint(0, len(result) - 1)
            j = (i + GOLDEN_STEP) % len(result)
            if i != j:
                indices[i], indices[j] = indices[j], indices[i]
        result = [result[indices[k]] for k in range(len(result))]

    # ── Scale 1: Sentence-level clause swapping ────
    pairs = golden_ratio_pairs(len(result))
    n_pairs_to_swap = max(1, round(len(pairs) * intensity))
    selected_pairs = rng.sample(pairs, min(n_pairs_to_swap, len(pairs)))

    for i, j in selected_pairs:
        new_i, new_j = swap_clauses_between(result[i], result[j])
        # If clause swap didn't change anything (single-clause sentences),
        # fall back to full sentence swap at this intensity
        if new_i == result[i] and new_j == result[j] and rng.random() < intensity:
            result[i], result[j] = result[j], result[i]
        else:
            result[i], result[j] = new_i, new_j

    # ── Scale 2: Clause-level permutation ──────────
    for i in range(len(result)):
        if len(result[i].split()) < profile.min_sentence_len:
            continue
        if rng.random() < profile.clause_swap_prob + intensity * 0.2:
            clauses = split_clauses(result[i])
            if len(clauses) > 1:
                mixed = cat_map_clauses(clauses)
                result[i] = ' '.join(mixed)

    # ── Scale 3: Word-level perturbation ───────────
    for i in range(len(result)):
        words = result[i].split()

        # Register shift
        if abs(profile.vocab_register_shift) > 0.01:
            words = apply_register_shift(
                words, profile.vocab_register_shift * intensity, rng
            )
            result[i] = ' '.join(words)

        # Contraction bias
        result[i] = apply_contraction_bias(result[i], profile.contraction_bias, rng)

        # Punctuation chaos
        result[i] = inject_punctuation(
            result[i], profile.punctuation_chaos * intensity, rng
        )

    return result
