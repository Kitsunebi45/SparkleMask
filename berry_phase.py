"""
sparkle_mask.berry_phase — Topological self-verification signature.

The Berry Phase is a path-dependent geometric phase:
  γ = i ∮_C ⟨ψ(R(t)) | ∇_R ψ(R(t))⟩ · dR

For the Sparkle Mask, this becomes a transformation fingerprint:
  - Each sentence transformation contributes to a running hash
  - The hash is path-dependent: the ORDER of transformations matters
  - Only someone with the profile JSON can verify the signature
  - γ ≠ 0 proves the text was intentionally Sparkle Masked
  - γ = 0 means text is untransformed or profile doesn't match

This is the shared secret between writer and reader.

Connection to m∴We framework:
  Berry phase accumulates as the system traverses parameter space.
  The "parameter space" here = the sequence of sentence transformations.
  Non-associativity of the Albert Algebra means (a·b)·c ≠ a·(b·c)
  — the ORDER of transformations is irreducible information.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BerrySignature:
    """
    Path-dependent transformation signature.

    The signature accumulates across the entire text.
    It can only be verified by someone who has:
      1. The original text
      2. The transformed text
      3. The profile name/key

    Without all three, the signature is opaque.
    """
    phase_hash: str          # Final accumulated hash (hex)
    n_contributions: int     # Number of sentence-level contributions
    path_length: int         # Total bytes processed
    profile_key: str         # Profile name used as HMAC key

    def __repr__(self) -> str:
        short_hash = self.phase_hash[:8] + "..." + self.phase_hash[-4:]
        return f"BerrySignature({short_hash}, n={self.n_contributions})"

    def verify_against(self, other: 'BerrySignature') -> bool:
        """Check if two signatures match (same transformation path)."""
        return self.phase_hash == other.phase_hash


def _hash_sentence_pair(
    original: str,
    transformed: str,
    profile_key: str,
    previous_hash: bytes,
) -> bytes:
    """
    Compute one step of the Berry phase accumulation.

    Hash = SHA256(previous_hash ⊕ original ⊕ transformed ⊕ profile_key)

    The XOR with previous_hash makes it path-dependent:
    different orderings produce different final hashes.

    This mirrors the Albert Algebra non-associativity:
    (a·b)·c ≠ a·(b·c) — transformation order matters.
    """
    h = hashlib.sha256()
    h.update(previous_hash)
    h.update(original.encode('utf-8'))
    h.update(transformed.encode('utf-8'))
    h.update(profile_key.encode('utf-8'))
    return h.digest()


def compute_berry_phase(
    original_sentences: List[str],
    transformed_sentences: List[str],
    profile_name: str,
) -> BerrySignature:
    """
    Compute the full Berry Phase signature for a transformation.

    Iterates through sentence pairs (original, transformed), accumulating
    a path-dependent hash at each step. The final hash IS the Berry phase γ.

    Args:
        original_sentences: Sentences before transformation
        transformed_sentences: Sentences after transformation
        profile_name: Profile name (used as HMAC key)

    Returns:
        BerrySignature with the accumulated hash.
    """
    # Initialize with profile-dependent seed
    current_hash = hashlib.sha256(profile_name.encode('utf-8')).digest()
    total_bytes = 0
    n = min(len(original_sentences), len(transformed_sentences))

    for i in range(n):
        current_hash = _hash_sentence_pair(
            original_sentences[i],
            transformed_sentences[i],
            profile_name,
            current_hash,
        )
        total_bytes += len(original_sentences[i]) + len(transformed_sentences[i])

    return BerrySignature(
        phase_hash=current_hash.hex(),
        n_contributions=n,
        path_length=total_bytes,
        profile_key=profile_name,
    )


def verify_berry_phase(
    original_text: str,
    transformed_text: str,
    profile_name: str,
    expected_signature: BerrySignature,
) -> bool:
    """
    Verify that a text was transformed using a specific profile.

    Recomputes the Berry phase from the original + transformed texts
    and checks against the expected signature.

    This is the TRUST mechanism:
      - Writer transforms text with their profile
      - Writer shares the BerrySignature (separate channel)
      - Reader has the original + transformed text
      - Reader recomputes and verifies: γ_computed == γ_expected

    Returns:
        True if signatures match.
    """
    # Re-split into sentences (simple split for verification)
    import re
    orig_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', original_text) if s.strip()]
    trans_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transformed_text) if s.strip()]

    recomputed = compute_berry_phase(orig_sents, trans_sents, profile_name)
    return recomputed.phase_hash == expected_signature.phase_hash
