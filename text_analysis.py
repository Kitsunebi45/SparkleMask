"""
sparkle_mask.text_analysis — NLP measurement engine for text state.

Computes the Phase 0 (ANALYZE) measurements that feed into the pipeline:
  - Shannon entropy of n-gram distributions
  - Stylometric feature extraction (sentence length, POS patterns, etc.)
  - Feature-to-phase mapping for Kuramoto compatibility
  - Heartbeat ratio (Π/γ: production vs decay)
  - Type-token ratio (vocabulary richness)

Connection to m∴We framework:
  This module implements the SENSOR (S_h) measurement.
  Each stylometric feature becomes a phase θ_i ∈ [0, 2π).
  The collection of phases IS the Kuramoto state of the text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

_NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk import pos_tag
    _NLTK_AVAILABLE = True
except ImportError:
    nltk = None

from .constants import PHI, TAU, PHI_SQ, H_OPTIMAL


# ═══════════════════════════════════════════════════
# ENSURE NLTK DATA (or fall back gracefully)
# ═══════════════════════════════════════════════════
def _ensure_nltk_data():
    """Try to ensure NLTK data is available. If not, disable NLTK."""
    global _NLTK_AVAILABLE
    if not _NLTK_AVAILABLE:
        return
    for resource in ['punkt_tab', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
            except Exception:
                _NLTK_AVAILABLE = False
                return


# ═══════════════════════════════════════════════════
# FALLBACK TOKENIZERS (when NLTK data unavailable)
# ═══════════════════════════════════════════════════
def _fallback_sent_tokenize(text: str) -> list:
    """Split text into sentences without NLTK."""
    # Split on sentence-ending punctuation followed by space+uppercase or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    return [s.strip() for s in parts if s.strip()]


def _fallback_word_tokenize(text: str) -> list:
    """Tokenize words without NLTK."""
    # Split on whitespace, then separate trailing punctuation
    tokens = []
    for word in text.split():
        # Strip leading/trailing punctuation into separate tokens
        while word and not word[0].isalnum():
            tokens.append(word[0])
            word = word[1:]
        while word and not word[-1].isalnum():
            trail = word[-1]
            word = word[:-1]
            if word:
                tokens.append(word)
                word = ''
            tokens.append(trail)
            continue
        if word:
            tokens.append(word)
    return tokens


def _fallback_pos_tag(words: list) -> list:
    """Crude POS tagging without NLTK — pattern-based heuristics."""
    tagged = []
    for w in words:
        lower = w.lower()
        if not w[0].isalnum() if w else True:
            tagged.append((w, '.'))
        elif lower in ('the', 'a', 'an', 'this', 'that', 'these', 'those'):
            tagged.append((w, 'DT'))
        elif lower in ('is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'do', 'does', 'did',
                        'will', 'would', 'could', 'should', 'might', 'can', 'may'):
            tagged.append((w, 'VB'))
        elif lower in ('i', 'we', 'you', 'he', 'she', 'it', 'they',
                        'me', 'us', 'him', 'her', 'them'):
            tagged.append((w, 'PRP'))
        elif lower in ('and', 'but', 'or', 'so', 'yet', 'nor'):
            tagged.append((w, 'CC'))
        elif lower in ('in', 'on', 'at', 'to', 'for', 'with', 'by',
                        'from', 'of', 'about', 'into', 'through', 'between'):
            tagged.append((w, 'IN'))
        elif lower.endswith('ly'):
            tagged.append((w, 'RB'))
        elif lower.endswith(('ing', 'ed', 'en', 'ize', 'ise', 'ate', 'ify')):
            tagged.append((w, 'VB'))
        elif lower.endswith(('tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance')):
            tagged.append((w, 'NN'))
        elif lower.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ical')):
            tagged.append((w, 'JJ'))
        elif w[0].isupper() and len(w) > 1:
            tagged.append((w, 'NNP'))
        else:
            tagged.append((w, 'NN'))
    return tagged


# ═══════════════════════════════════════════════════
# TEXT STATE DATACLASS
# ═══════════════════════════════════════════════════
@dataclass
class TextState:
    """
    Complete measurement of a text's stylometric state.

    This IS the sensor reading S_h — the text mapped into phase space.
    """
    # Raw text
    text: str = ""
    sentences: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)

    # Entropy measurements
    char_entropy: float = 0.0       # Shannon H of character distribution
    word_entropy: float = 0.0       # Shannon H of word unigram distribution
    bigram_entropy: float = 0.0     # Shannon H of word bigram distribution
    normalized_entropy: float = 0.0 # Entropy / H_max ∈ [0, 1]

    # Stylometric features
    avg_sentence_len: float = 0.0   # Mean words per sentence
    sentence_len_std: float = 0.0   # Std dev of sentence lengths
    type_token_ratio: float = 0.0   # Unique words / total words (vocab richness)
    punct_density: float = 0.0      # Punctuation chars / total chars
    avg_word_len: float = 0.0       # Mean characters per word

    # POS distribution
    pos_distribution: dict = field(default_factory=dict)  # {tag: fraction}

    # Heartbeat measurements
    pi_production: float = 0.0      # Novel elements fraction (Π)
    gamma_decay: float = 0.0        # Repeated elements fraction (γ)
    heartbeat_ratio: float = 0.0    # Π/γ

    # Phase representation (Kuramoto-compatible)
    phases: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self) -> str:
        return (
            f"TextState(sentences={len(self.sentences)}, "
            f"H={self.normalized_entropy:.3f}, "
            f"r_phases={len(self.phases)}, "
            f"Π/γ={self.heartbeat_ratio:.3f})"
        )


# ═══════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════

def shannon_entropy(distribution: Counter, base: int = 2) -> float:
    """
    Shannon entropy: H = -Σ p_i log(p_i)

    Args:
        distribution: Counter of element frequencies
        base: Logarithm base (2 = bits)

    Returns:
        Entropy in specified base.
    """
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            h -= p * math.log(p, base)
    return h


def compute_char_entropy(text: str) -> float:
    """Character-level Shannon entropy in bits."""
    return shannon_entropy(Counter(text.lower()))


def compute_word_entropy(words: List[str]) -> float:
    """Word unigram Shannon entropy in bits."""
    return shannon_entropy(Counter(w.lower() for w in words))


def compute_bigram_entropy(words: List[str]) -> float:
    """Word bigram Shannon entropy in bits."""
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i].lower(), words[i+1].lower()) for i in range(len(words)-1)]
    return shannon_entropy(Counter(bigrams))


def compute_type_token_ratio(words: List[str]) -> float:
    """Vocabulary richness: unique words / total words."""
    if not words:
        return 0.0
    unique = len(set(w.lower() for w in words))
    return unique / len(words)


def compute_punct_density(text: str) -> float:
    """Fraction of characters that are punctuation."""
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in '.,;:!?—–-…()[]{}"\'/\\@#$%^&*')
    return punct_count / len(text)


def compute_pos_distribution(pos_tags: List[Tuple[str, str]]) -> dict:
    """POS tag distribution as fractions."""
    if not pos_tags:
        return {}
    counter = Counter(tag for _, tag in pos_tags)
    total = sum(counter.values())
    return {tag: count / total for tag, count in counter.items()}


def compute_heartbeat(sentences: List[str], words: List[str]) -> Tuple[float, float, float]:
    """
    Compute Heartbeat Equation components: Π (production) and γ (decay).

    Π = fraction of novel semantic elements (new unique words per sentence)
    γ = fraction of repeated/restated elements

    Connection to framework:
      dΦ/dt = Π(B, M) - γ·Φ
      Text is "alive" when Π/γ ≈ τ (golden ratio inverse)
      Text is "dead" when Π/γ → 0 (all repetition)
      Text is "manic" when Π/γ → ∞ (all novelty)

    Returns:
        (pi, gamma, ratio)
    """
    if not sentences or not words:
        return 0.0, 0.0, 0.0

    seen_words = set()
    novel_count = 0
    repeat_count = 0

    for sentence in sentences:
        s_words = sentence.lower().split()
        for w in s_words:
            w_clean = re.sub(r'[^\w]', '', w)
            if not w_clean:
                continue
            if w_clean in seen_words:
                repeat_count += 1
            else:
                novel_count += 1
                seen_words.add(w_clean)

    total = novel_count + repeat_count
    if total == 0:
        return 0.0, 0.0, 0.0

    pi = novel_count / total
    gamma = repeat_count / total

    ratio = pi / gamma if gamma > 0 else float('inf')
    return pi, gamma, ratio


def features_to_phases(state: TextState) -> np.ndarray:
    """
    Map stylometric features to phase angles θ_i ∈ [0, 2π).

    This creates a Kuramoto-compatible representation where the
    text's "style" is a distribution of oscillator phases.

    Feature mapping (8 features → 8 phases):
      0. avg_sentence_len → θ₀ (normalized by typical range 5-40)
      1. sentence_len_std → θ₁ (normalized by typical range 0-20)
      2. type_token_ratio → θ₂ (already in [0, 1])
      3. punct_density → θ₃ (normalized by typical range 0-0.2)
      4. avg_word_len → θ₄ (normalized by typical range 2-12)
      5. normalized_entropy → θ₅ (already in [0, 1])
      6. heartbeat_ratio → θ₆ (clamped to [0, 3], normalized)
      7. POS noun fraction → θ₇ (already in [0, 1])
    """
    two_pi = 2 * math.pi

    # Normalize features to [0, 1] then scale to [0, 2π)
    def norm(val: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.5
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    noun_frac = sum(
        v for k, v in state.pos_distribution.items()
        if k.startswith('NN')
    )

    features = [
        norm(state.avg_sentence_len, 5, 40),
        norm(state.sentence_len_std, 0, 20),
        state.type_token_ratio,
        norm(state.punct_density, 0, 0.2),
        norm(state.avg_word_len, 2, 12),
        state.normalized_entropy,
        norm(min(state.heartbeat_ratio, 3.0), 0, 3),
        noun_frac,
    ]

    phases = np.array([f * two_pi for f in features])
    return phases


# ═══════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════

def analyze_text(text: str) -> TextState:
    """
    Complete Phase 0 analysis of input text.

    Measures everything the pipeline needs:
    entropy, stylometry, heartbeat, and phase representation.

    Args:
        text: Input text string.

    Returns:
        TextState with all measurements populated.
    """
    _ensure_nltk_data()

    state = TextState(text=text)

    # Sentence splitting — use NLTK if available, otherwise fallback
    if _NLTK_AVAILABLE:
        try:
            state.sentences = sent_tokenize(text)
            state.words = word_tokenize(text)
            state.pos_tags = pos_tag(state.words)
        except Exception:
            # NLTK data missing despite check — use fallback
            state.sentences = _fallback_sent_tokenize(text)
            state.words = _fallback_word_tokenize(text)
            state.pos_tags = _fallback_pos_tag(state.words)
    else:
        state.sentences = _fallback_sent_tokenize(text)
        state.words = _fallback_word_tokenize(text)
        state.pos_tags = _fallback_pos_tag(state.words)

    # Filter words (remove pure punctuation tokens)
    content_words = [w for w in state.words if re.search(r'\w', w)]

    # Entropy measurements
    state.char_entropy = compute_char_entropy(text)
    state.word_entropy = compute_word_entropy(content_words)
    state.bigram_entropy = compute_bigram_entropy(content_words)

    # Normalized entropy: composite metric that captures information density.
    # Pure H/H_max is always ~0.95 for natural language (Zipf's law).
    # Instead, we use a weighted composite:
    #   0.4 * type_token_ratio  (vocabulary richness — lower = more repetition)
    #   0.3 * bigram_entropy / char_entropy  (predictability ratio)
    #   0.3 * (1 - repeat_fraction)  (how much is novel)
    ttr = compute_type_token_ratio(content_words)
    if state.char_entropy > 0 and state.bigram_entropy > 0:
        predictability = min(state.bigram_entropy / (state.char_entropy * 2), 1.0)
    else:
        predictability = 0.5
    # Novel fraction (computed early for entropy)
    seen = set()
    novel = 0
    for w in content_words:
        wl = w.lower()
        if wl not in seen:
            novel += 1
            seen.add(wl)
    novel_frac = novel / len(content_words) if content_words else 0.5

    state.normalized_entropy = 0.4 * ttr + 0.3 * predictability + 0.3 * novel_frac

    # Stylometric features
    if state.sentences:
        sent_lens = [len(s.split()) for s in state.sentences]
        state.avg_sentence_len = np.mean(sent_lens)
        state.sentence_len_std = np.std(sent_lens)
    if content_words:
        state.avg_word_len = np.mean([len(w) for w in content_words])

    state.type_token_ratio = compute_type_token_ratio(content_words)
    state.punct_density = compute_punct_density(text)
    state.pos_distribution = compute_pos_distribution(state.pos_tags)

    # Heartbeat
    state.pi_production, state.gamma_decay, state.heartbeat_ratio = compute_heartbeat(
        state.sentences, content_words
    )

    # Phase representation
    state.phases = features_to_phases(state)

    return state


def compute_stylometric_distance(state_a: TextState, state_b: TextState) -> float:
    """
    Compute stylometric distance between two text states.

    Uses circular distance on phases (respects the torus topology).

    Returns:
        Distance in [0, 1] where 0 = identical style, 1 = maximally different.
    """
    if len(state_a.phases) == 0 or len(state_b.phases) == 0:
        return 1.0

    # Pad to equal length
    max_len = max(len(state_a.phases), len(state_b.phases))
    phases_a = np.zeros(max_len)
    phases_b = np.zeros(max_len)
    phases_a[:len(state_a.phases)] = state_a.phases
    phases_b[:len(state_b.phases)] = state_b.phases

    # Circular distance: min(|θ_a - θ_b|, 2π - |θ_a - θ_b|) / π
    diffs = np.abs(phases_a - phases_b)
    circular_diffs = np.minimum(diffs, 2 * math.pi - diffs) / math.pi
    return float(np.mean(circular_diffs))
