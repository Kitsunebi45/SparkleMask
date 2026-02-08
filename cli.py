"""
sparkle_mask.cli — Command-line interface for the Sparkle Mask.

Usage:
    # Transform text from stdin
    echo "your text" | python -m sparkle_mask --preset default

    # Transform from file
    python -m sparkle_mask --input draft.txt --preset sparkle --output masked.txt

    # With diagnostics
    python -m sparkle_mask --input draft.txt --preset sparkle --diagnostics

    # List presets
    python -m sparkle_mask --list-presets

    # Blend two presets
    python -m sparkle_mask --input draft.txt --blend default:0.6 dense:0.4

    # Interactive mode
    python -m sparkle_mask --interactive --preset default

    # Show profile details
    python -m sparkle_mask --show-profile default

    # Use custom preset directory
    python -m sparkle_mask --preset-dir ./my_presets --preset custom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .mask import SparkleMask
from .cadence_profile import CadenceProfile


def parse_blend_arg(blend_str: str) -> tuple:
    """
    Parse blend argument like 'kitty:0.6' into (name, weight).

    Returns:
        (preset_name, weight)
    """
    parts = blend_str.split(':')
    name = parts[0]
    weight = float(parts[1]) if len(parts) > 1 else 0.5
    return name, weight


def run_interactive(mask: SparkleMask, show_diagnostics: bool = True):
    """Run interactive mode — type text, get masked output."""
    print("=" * 52)
    print("  Sparkle Mask — Interactive Mode")
    print(f"  Profile: {mask.profile.name}")
    print("  Type text and press Enter twice to transform.")
    print("  Type 'quit' to exit, 'profile' to see current profile.")
    print("  Type 'swap <name>' to change preset.")
    print("=" * 52)
    print()

    while True:
        try:
            lines = []
            print("Input (empty line to transform):")
            while True:
                line = input()
                if line.strip() == '':
                    break
                if line.strip().lower() == 'quit':
                    print("Goodbye!")
                    return
                if line.strip().lower() == 'profile':
                    print(mask.profile.summary())
                    lines = []
                    break
                if line.strip().lower().startswith('swap '):
                    preset_name = line.strip()[5:].strip()
                    try:
                        mask.load_preset(preset_name)
                        print(f"Swapped to: {mask.profile.name}")
                    except FileNotFoundError as e:
                        print(f"Error: {e}")
                    lines = []
                    break
                lines.append(line)

            if not lines:
                continue

            text = '\n'.join(lines)
            result = mask.transform(text, verbose=show_diagnostics)

            print()
            print("--- MASKED OUTPUT ---")
            print(result.masked_text)
            print("--- END ---")

            if show_diagnostics:
                print()
                print(result.diagnostic.render())

            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sparkle Mask — Metacybernetic cadence scrambler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    parser.add_argument('--input', '-i', type=str, help='Input text file')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')

    # Profile selection
    parser.add_argument('--preset', '-p', type=str, help='Preset name to use')
    parser.add_argument('--profile', type=str, help='Path to custom profile JSON')
    parser.add_argument('--preset-dir', type=str, help='Custom preset directory')

    # Blending
    parser.add_argument(
        '--blend', nargs=2, type=str, metavar=('A:W', 'B:W'),
        help='Blend two presets (e.g. --blend kitty:0.6 vix:0.4)'
    )

    # Modes
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--diagnostics', '-d', action='store_true', help='Show diagnostics')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')
    parser.add_argument('--show-profile', type=str, help='Show profile details')

    # Advanced
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Determine preset directory
    preset_dir = args.preset_dir or str(Path(__file__).parent / "presets")

    # ── List presets mode ──────────────────────────
    if args.list_presets:
        presets = CadenceProfile.list_presets(preset_dir)
        if not presets:
            print("No presets found.")
        else:
            print("Available presets:")
            for p in presets:
                print(f"  {p.name:15s} — {p.description}")
                print(f"  {'':15s}   cat={p.cat_intensity:.1f} r={p.target_r:.1f} H={p.h_target:.1f}")
        return

    # ── Show profile mode ─────────────────────────
    if args.show_profile:
        profile = CadenceProfile.find_preset(args.show_profile, preset_dir)
        if profile:
            print(profile.summary())
        else:
            print(f"Preset '{args.show_profile}' not found.")
        return

    # ── Build the mask ─────────────────────────────
    mask = SparkleMask(preset_dir=preset_dir)

    # Load profile
    if args.profile:
        mask.load_profile(args.profile)
    elif args.preset:
        try:
            mask.load_preset(args.preset)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.blend:
        a_name, a_weight = parse_blend_arg(args.blend[0])
        b_name, b_weight = parse_blend_arg(args.blend[1])
        # Normalize weights
        total = a_weight + b_weight
        b_frac = b_weight / total  # blend() weight is fraction of B

        profile_a = CadenceProfile.find_preset(a_name, preset_dir)
        profile_b = CadenceProfile.find_preset(b_name, preset_dir)
        if not profile_a or not profile_b:
            print(f"Error: Could not find presets '{a_name}' and/or '{b_name}'", file=sys.stderr)
            sys.exit(1)
        blended = CadenceProfile.blend(profile_a, profile_b, weight=b_frac)
        mask.set_profile(blended)

    # ── Interactive mode ──────────────────────────
    if args.interactive:
        run_interactive(mask, show_diagnostics=args.diagnostics)
        return

    # ── Read input ─────────────────────────────────
    if args.input:
        with open(args.input, 'r') as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Error: No input provided. Use --input, pipe text, or --interactive mode.",
              file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("Error: Empty input.", file=sys.stderr)
        sys.exit(1)

    # ── Transform ──────────────────────────────────
    result = mask.transform(
        text=text.strip(),
        seed=args.seed,
        verbose=args.verbose,
    )

    # ── Output ─────────────────────────────────────
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result.masked_text)
        if args.diagnostics:
            print(result.diagnostic.render())
        print(f"Output written to {args.output}")
    else:
        print(result.masked_text)
        if args.diagnostics:
            print()
            print(result.diagnostic.render())


if __name__ == '__main__':
    main()
