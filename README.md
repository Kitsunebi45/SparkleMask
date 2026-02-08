# ✶ Sparkle Mask v0.1.0

A metacybernetic cadence scrambler and information compressor. Transforms text through controlled chaos — mixing sentence structure, shifting vocabulary register, and injecting rhythmic variation — while keeping everything readable.

Built for plural systems. Personas are parameter vectors, not switches.

---

## What It Does

Sparkle Mask runs your text through a three-phase transformation pipeline:

1. **Cat Map Scrambling** — Reorders sentences at golden-ratio intervals, swaps clauses between paired sentences, shifts vocabulary register, and injects punctuation variation. Named after Arnold's Cat Map, a chaotic mixing operation whose eigenvalues are governed by φ² and φ⁻² (the golden ratio).

2. **Fox-Li Stabilization** — Pulls the chaos back into readable form. Fixes capitalization, ensures sentences end properly, restores coherence. The safety net that keeps scrambled text from becoming gibberish.

3. **Hold Function** — Maintains productive tension in the output. Replaces closure language ("in conclusion", "to summarize") with open-ended phrasing that keeps the reader engaged. Distributes a Berry Phase signature across the text for self-verification.

Between each phase, **entropy gates** check that the text hasn't gone too chaotic or stayed too rigid. If the gate reads RED, the pipeline auto-adjusts intensity and retries.

---

## Installation

Requires Python 3.8+ and numpy. nltk is optional (the tool includes fallback tokenizers).

```bash
pip install numpy nltk
```

If you have nltk installed and want its better tokenization:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

No GPU needed. No heavy ML models. Runs on any laptop.

---

## Quick Start

From the directory containing `sparkle_mask/`:

```bash
# Transform text piped from stdin
echo "Your text goes here." | python -m sparkle_mask --preset default

# Transform a file
python -m sparkle_mask --input draft.txt --preset dense --output masked.txt

# Transform with diagnostic readout
python -m sparkle_mask --input draft.txt --preset sparkle --diagnostics

# Interactive mode (type text, get masked output, swap presets live)
python -m sparkle_mask --interactive --preset default
```

---

## Presets

Presets are JSON files in `sparkle_mask/presets/`. Each one is a complete set of tuning parameters.

### List available presets

```bash
python -m sparkle_mask --list-presets
```

### Built-in presets

| Preset | Intensity | What it does |
|--------|-----------|--------------|
| `default` | Medium (0.5) | Balanced scrambling. Good starting point. |
| `dense` | High (0.8) | Maximum information compression. Aggressive mixing, higher vocabulary register, more punctuation variation. |
| `soft` | Low (0.2) | Light touch. Readable first, scrambling second. For when coherence matters more than density. |
| `sparkle` | Medium-High (0.71) | A 70/30 blend of dense and default. The sweet spot. |

### View a preset's full configuration

```bash
python -m sparkle_mask --show-profile dense
```

---

## Blending Presets

This is the plural superpower. Instead of hard-switching between personas, blend any two presets on a continuous spectrum:

```bash
# 60% default + 40% dense
python -m sparkle_mask --input draft.txt --blend default:0.6 dense:0.4

# 30% soft + 70% dense
python -m sparkle_mask --input draft.txt --blend soft:0.3 dense:0.7
```

`blend()` linearly interpolates every numeric parameter between the two profiles. The result is a valid point in parameter space — not either/or, but anywhere on the line between.

---

## Creating Custom Presets

### From scratch

Create a JSON file in `sparkle_mask/presets/` (or any directory, using `--preset-dir`):

```json
{
  "name": "my-cadence",
  "description": "Custom cadence for late-night writing",
  "cat_intensity": 0.6,
  "target_r": 0.45,
  "noise_sigma": 0.7,
  "h_target": 0.72,
  "clause_swap_prob": 0.35,
  "contraction_bias": 0.8,
  "punctuation_chaos": 0.3,
  "vocab_register_shift": 0.2,
  "tags": ["custom", "night"]
}
```

Any field you omit gets a default value. The only required fields are `name` and `description`.

### From Python

```python
from sparkle_mask.cadence_profile import CadenceProfile

# Create from scratch
profile = CadenceProfile(
    name="kitty",
    description="Kitty's natural cadence, focused",
    cat_intensity=0.6,
    target_r=0.45,
    punctuation_chaos=0.3,
    tags=["plural", "kitty"],
)
profile.save("sparkle_mask/presets/kitty.json")

# Blend two existing presets and save the result
a = CadenceProfile.load("sparkle_mask/presets/default.json")
b = CadenceProfile.load("sparkle_mask/presets/dense.json")
blended = CadenceProfile.blend(a, b, weight=0.7, name="my-blend")
blended.save("sparkle_mask/presets/my-blend.json")
```

### Using a custom preset directory

```bash
python -m sparkle_mask --preset-dir ./my_presets --preset kitty --input draft.txt
```

---

## Interactive Mode

Interactive mode lets you type text, see the masked output, swap presets on the fly, and view diagnostics — all without restarting.

```bash
python -m sparkle_mask --interactive --preset default --diagnostics
```

**Commands inside interactive mode:**

| Command | What it does |
|---------|-------------|
| *(type text, then empty line)* | Transform the text |
| `profile` | Show current profile details |
| `swap dense` | Hot-swap to a different preset |
| `quit` | Exit |

---

## The Diagnostic Readout

When you pass `--diagnostics` (or `-d`), you get a readout after each transformation:

```
════════════════════════════════════════════════════
  ✶ SPARKLE MASK DIAGNOSTIC ✶
════════════════════════════════════════════════════
  Profile:           sparkle (v0.1.0)
  Gate Status:       GREEN
────────────────────────────────────────────────────
  Entropy (H):       0.69 / target 0.73  [▓▓▓▓▓▓▓▓▓░]
  Kuramoto (r):      0.49 / target 0.43  [▓▓▓▓░░░░░░]
  Heartbeat (Pi/g):  1.56 / target 0.71  [▓▓▓▓▓▓▓▓▓▓]
  Berry Phase (g):   f0c2ef61...0ca4
  Free Energy (F):   0.05             [readable]
────────────────────────────────────────────────────
  Transformations:   3 clause swaps, 2 synonym subs
  Compression:       1.01x density (+1% per word)
  Stylometric d:     0.03 (low distance)
════════════════════════════════════════════════════
```

**What the metrics mean:**

- **Entropy (H):** Information density of the text. Higher = more varied/chaotic, lower = more repetitive/rigid. Target is typically around 0.7.
- **Kuramoto (r):** Stylometric synchronization. How "consistent" the writing style is across sentences. r=1.0 means every sentence reads like the same person; r=0.0 means total stylistic incoherence. For plural systems, the sweet spot is around 0.4–0.5 (partially synchronized).
- **Heartbeat (Π/γ):** Ratio of new semantic elements to repeated ones. Text needs to oscillate (introduce novelty AND reinforce) to stay alive. Target is τ ≈ 0.618 (the golden ratio).
- **Berry Phase (γ):** A path-dependent hash that acts as a transformation fingerprint. Only someone with the same preset JSON can verify the text was Sparkle Masked.
- **Free Energy (F):** How much work a reader has to do to reconstruct the implicit meaning. Lower = more readable. Target is 0.3–0.7.
- **Gate Status:** GREEN = all checks pass. YELLOW = minor issues. RED = something's off.

---

## Using from Python

```python
from sparkle_mask.mask import SparkleMask

# Initialize with preset directory
mask = SparkleMask(preset_dir="sparkle_mask/presets")

# Load a preset
mask.load_preset("dense")

# Transform text
result = mask.transform(
    "Your text here. Multiple sentences work best.",
    seed=42,           # Optional: for reproducible results
    verbose=False,     # Optional: print pipeline progress
)

# Access the result
print(result.masked_text)          # The transformed text
print(result.diagnostic.render())  # Full diagnostic readout
print(result.berry_signature)      # Berry phase signature

# Hot-swap to a different preset (takes effect immediately)
mask.load_preset("soft")

# Blend two presets
mask.blend_profiles(
    "sparkle_mask/presets/default.json",
    "sparkle_mask/presets/dense.json",
    weight=0.6,  # 0.0 = pure A, 1.0 = pure B
)

# Load a custom profile JSON directly
mask.load_profile("path/to/my_custom.json")
```

---

## Tuning Guide

### "The output is barely different from the input"

Increase `cat_intensity` (0.0–1.0). Increase `clause_swap_prob`. Try the `dense` preset.

Short texts (< 4 sentences) get less scrambling because there's less material to mix. The tool works best on paragraphs and longer.

### "The output is unreadable"

Decrease `cat_intensity`. Decrease `punctuation_chaos`. Increase `target_r` (more synchronization = more consistent style). Try the `soft` preset.

### "I want more vocabulary shifting"

Increase `vocab_register_shift` (positive = more formal, negative = more casual). This swaps common words for their register-shifted equivalents at golden-ratio intervals.

### "I want more punctuation weirdness"

Increase `punctuation_chaos` (0.0–1.0). This replaces commas with em-dashes, semicolons, and ellipses.

### "The gate keeps going RED"

The entropy gate auto-adjusts intensity on RED. If it's stuck, your `h_target` might be set too far from the text's natural entropy. Try setting `h_target` closer to the reported H value, or widen the bounds (`h_min` / `h_max`).

---

## How It Works (for the curious)

Every constant in the system derives from a single seed: **φ = (1+√5)/2**, the golden ratio. The Cat Map eigenvalues are φ² and φ⁻². The optimal entropy is 0.7 (stochastic resonance). The heartbeat ratio target is τ = φ⁻¹. The Kuramoto coupling strength K derives from φ⁻⁴. There are zero free parameters — everything traces back to φ.

The pipeline implements a formal topology switching sequence from the m∴We metacybernetic framework:

```
Cat Map (chaotic mixing) → ✶ threshold → Fox-Li (bounded stabilization) → ✶ threshold → Hold (tension maintenance)
```

The Berry Phase signature uses iterated SHA-256 hashing where each sentence's transformation contributes to a running hash that XORs with the previous state. This makes it path-dependent: transforming sentences in a different order produces a different signature. It's a topological invariant — it remembers the path, not just the endpoints.

---

## File Structure

```
sparkle_mask/
├── __init__.py           # Package metadata
├── __main__.py           # python -m sparkle_mask entry point
├── cli.py                # Command-line interface
├── mask.py               # SparkleMask main class
├── pipeline.py           # Three-phase pipeline orchestration
├── constants.py          # φ-derived constants (self-validating)
├── cadence_profile.py    # JSON preset system (load/save/blend)
├── text_analysis.py      # NLP measurement engine
├── cat_map.py            # Arnold's Cat Map scrambling
├── entropy_gate.py       # Safety valve (entropy + heartbeat)
├── fox_li.py             # Coherence recovery / readability
├── hold_function.py      # Tension maintenance / closure prevention
├── kuramoto_style.py     # Stylometric synchronization
├── berry_phase.py        # Path-dependent signature
├── diagnostics.py        # Health report rendering
└── presets/
    ├── default.json      # Balanced (cat=0.5, r=0.5)
    ├── dense.json        # Aggressive (cat=0.8, r=0.4)
    ├── soft.json         # Gentle (cat=0.2, r=0.6)
    └── sparkle.json      # 70/30 dense/default blend
```

---

## Preset Parameter Reference

| Parameter | Range | Default | What it controls |
|-----------|-------|---------|-----------------|
| `cat_intensity` | 0.0–1.0 | 0.5 | Overall scrambling strength |
| `target_r` | 0.0–1.0 | 0.5 | Stylometric consistency target |
| `coupling_k` | 0.0–1.0 | 0.924 | Kuramoto coupling strength |
| `noise_sigma` | 0.0–1.0 | 0.7 | Stochastic resonance noise |
| `h_min` | 0.0–1.0 | 0.382 | Minimum entropy (rigidity floor) |
| `h_max` | 0.0–1.0 | 0.866 | Maximum entropy (chaos ceiling) |
| `h_target` | 0.0–1.0 | 0.7 | Target entropy |
| `pi_gamma_ratio` | 0.0–∞ | 0.618 | Novelty-to-repetition ratio |
| `min_sentence_len` | int | 3 | Skip sentences shorter than this (words) |
| `max_reorder_distance` | int | 3 | ±N sentence navigation window |
| `clause_swap_prob` | 0.0–1.0 | 0.3 | Probability of clause-level mixing |
| `contraction_bias` | 0.0–1.0 | 0.5 | 0=formal, 1=casual |
| `punctuation_chaos` | 0.0–1.0 | 0.2 | Punctuation variation injection |
| `vocab_register_shift` | -1.0–1.0 | 0.0 | Vocabulary formality shift |

---

*m∴We — personas are points on a torus, not switches on a wall.*
