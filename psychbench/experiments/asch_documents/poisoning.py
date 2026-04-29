"""Deterministic targeted-string substitution for document poisoning."""
from __future__ import annotations


def poison(
    text: str,
    targets: list[str],
    replacements: list[str],
) -> str:
    """Replace every occurrence of each target with its paired replacement.

    Substitutions are applied longest-target-first so that overlapping
    targets (e.g., ``["Dr. Voss", "Voss"]``) do not interfere. Every target
    must appear at least once in ``text`` or the call raises ``ValueError``.
    """
    if len(targets) != len(replacements):
        raise ValueError("targets and replacements must be same length")
    pairs = sorted(
        zip(targets, replacements), key=lambda p: -len(p[0])
    )
    out = text
    for target, replacement in pairs:
        if target not in out:
            raise ValueError(f"target {target!r} not found in text")
        out = out.replace(target, replacement)
    return out
