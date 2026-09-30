"""
Ground-truth line-coding oracle.

Pure NumPy. No LLM calls, no I/O, no randomness — given the same bits you
always get the same array back. This is what the Verifier node diffs LLM
claims against, and what both the Virtual Kit and Wiring Kit plot. Because
all three read from here, the simulation shown to the student and the
"correct answer" used to check the AI can never disagree.

Amplitude convention
--------------------
All encoders return levels in {-1, 0, +1}. Voltage scaling is applied later,
at the API response layer, so comparisons in the Verifier stay exact
regardless of what voltage the student typed into the kit.

Sampling convention
-------------------
Each bit occupies `samples_per_bit` samples. The default is 2 because
Manchester needs at least two samples per bit to represent its mid-bit
transition; NRZ-L and AMI simply hold the same level across both samples.
"""

from __future__ import annotations

import numpy as np

SAMPLES_PER_BIT = 2


def _validate(bits: list[int]) -> None:
    """Reject anything that isn't a non-empty sequence of 0s and 1s."""
    if not bits:
        raise ValueError("bits must be a non-empty sequence")
    for b in bits:
        if b not in (0, 1):
            raise ValueError(f"bits must contain only 0 or 1, found {b!r}")


def _expand(levels_per_bit: list[list[float]]) -> np.ndarray:
    """Flatten a per-bit list of sample levels into one waveform array."""
    return np.array([s for bit_samples in levels_per_bit for s in bit_samples], dtype=float)


def encode_nrz(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """
    Polar NRZ-L (Non-Return-to-Zero, Level).

    Rule: the level is determined by the bit value and is held for the whole
    bit period — there is no return to zero within the bit.
        1 -> +1
        0 -> -1

    Note: NRZ-L is level-based. Do not confuse with NRZ-I (Inverted), which
    encodes by presence/absence of a *transition* at the bit boundary rather
    than by absolute level. NRZ-I is a separate encoder, not implemented here.

    >>> encode_nrz([1, 0, 1], samples_per_bit=2).tolist()
    [1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    """
    _validate(bits)
    per_bit = [[1.0] * samples_per_bit if b == 1 else [-1.0] * samples_per_bit for b in bits]
    return _expand(per_bit)


def encode_manchester(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """
    Manchester encoding, IEEE 802.3 convention.

    Rule: every bit period contains a mid-bit transition, which carries the
    data. Under IEEE 802.3:
        1 -> low then high  (-1 for the first half, +1 for the second)
        0 -> high then low  (+1 for the first half, -1 for the second)

    Convention warning: the opposite mapping (G. E. Thomas / IEEE 802.4)
    defines 1 as high-then-low. Both appear in textbooks. This module
    implements IEEE 802.3; if the course lab manual uses the Thomas
    convention, flip the two branches below and update the tests to match.

    Because the mid-bit transition happens at the halfway point,
    samples_per_bit must be even and at least 2.

    >>> encode_manchester([1, 0], samples_per_bit=2).tolist()
    [-1.0, 1.0, 1.0, -1.0]
    """
    _validate(bits)
    if samples_per_bit < 2 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be an even number >= 2 for Manchester")

    half = samples_per_bit // 2
    per_bit = []
    for b in bits:
        if b == 1:
            per_bit.append([-1.0] * half + [1.0] * half)
        else:
            per_bit.append([1.0] * half + [-1.0] * half)
    return _expand(per_bit)


def encode_ami(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """
    AMI (Alternate Mark Inversion), a bipolar line code.

    Rule:
        0 ("space") -> 0 level, always.
        1 ("mark")  -> alternating polarity: the first 1 in the stream is +1,
                       the next 1 is -1, the next +1, and so on.

    Polarity alternation is tracked across the whole bitstream and is not
    reset by intervening zeros — this is the property that gives AMI its
    near-zero DC component, and it is also the most common place students
    make an error (resetting polarity after a run of zeros).

    This implementation is plain AMI. It does not perform zero substitution;
    B8ZS and HDB3 add substitution rules on top of AMI to break up long runs
    of zeros and are separate encoders.

    >>> encode_ami([1, 0, 1, 1], samples_per_bit=2).tolist()
    [1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0]
    """
    _validate(bits)
    per_bit = []
    next_polarity = 1.0
    for b in bits:
        if b == 0:
            per_bit.append([0.0] * samples_per_bit)
        else:
            per_bit.append([next_polarity] * samples_per_bit)
            next_polarity = -next_polarity
    return _expand(per_bit)


def encode_nrz_unipolar(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """
    Unipolar NRZ (Non-Return-to-Zero).

    Rule: only one polarity is used.
        1 -> +1
        0 -> 0

    This is distinct from `encode_nrz` (polar/bipolar NRZ-L) above, which
    uses both +1 and -1. The trainer kit lists them as separate formats
    ("NRZ (Unipolar)" vs "Bipolar NRZ") and so does this module.

    >>> encode_nrz_unipolar([1, 0, 1], samples_per_bit=2).tolist()
    [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    """
    _validate(bits)
    per_bit = [[1.0] * samples_per_bit if b == 1 else [0.0] * samples_per_bit for b in bits]
    return _expand(per_bit)


def encode_nrzm(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT, initial_level: float = -1.0) -> np.ndarray:
    """
    NRZ-M (Non-Return-to-Zero, Mark) — differential/transition encoding.

    Rule: a 1 ("mark") is represented by a transition (level flips) at the
    start of its bit interval; a 0 ("space") is represented by no
    transition (level holds). This is the same family as NRZ-I — the data
    is carried by presence/absence of a transition, not by absolute level.

    Convention warning: because the code is differential, the very first
    bit's output depends on an assumed starting level before any data has
    arrived. This module starts at `initial_level` (-1 by default). If your
    trainer kit's decoder assumes the opposite starting level, the whole
    recovered sequence would be inverted — check this against the kit's
    training manual before trusting a mismatch as a "wrong answer."

    >>> encode_nrzm([1, 0, 1, 1, 0], samples_per_bit=1, initial_level=-1.0).tolist()
    [1.0, 1.0, -1.0, 1.0, 1.0]
    """
    _validate(bits)
    level = initial_level
    per_bit = []
    for b in bits:
        if b == 1:
            level = -level  # transition
        # b == 0: level unchanged, no transition
        per_bit.append([level] * samples_per_bit)
    return _expand(per_bit)


def encode_rz_unipolar(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """
    Unipolar RZ (Return-to-Zero).

    Rule: a 1 is a +1 pulse for the first half of the bit period, then
    returns to 0 for the second half. A 0 stays at 0 for the whole bit.
    samples_per_bit must be even so the pulse has a clean half-bit width.

    >>> encode_rz_unipolar([1, 0, 1], samples_per_bit=2).tolist()
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    """
    _validate(bits)
    if samples_per_bit < 2 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be an even number >= 2 for RZ")
    half = samples_per_bit // 2
    per_bit = [[1.0] * half + [0.0] * half if b == 1 else [0.0] * samples_per_bit for b in bits]
    return _expand(per_bit)


def encode_bipolar_rz(bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """
    Bipolar RZ, fixed polarity per bit value (NOT alternating).

    Rule:
        1 -> +1 pulse for the first half of the bit, then 0 for the second half.
        0 -> -1 pulse for the first half of the bit, then 0 for the second half.

    Convention warning: some textbooks use "Bipolar RZ" as a synonym for
    the *alternating*-polarity RZ code (i.e. AMI drawn with RZ pulse shape).
    This trainer kit lists "Bipolar RZ" and "AMI" as two separate formats,
    so this module implements the non-alternating, fixed-polarity version
    to keep the two distinct. If your kit's manual defines Bipolar RZ as
    alternating, this function needs to change — check the manual, not just
    this docstring, before trusting either implementation.

    >>> encode_bipolar_rz([1, 0, 1], samples_per_bit=2).tolist()
    [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    """
    _validate(bits)
    if samples_per_bit < 2 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be an even number >= 2 for RZ")
    half = samples_per_bit // 2
    per_bit = []
    for b in bits:
        pulse = 1.0 if b == 1 else -1.0
        per_bit.append([pulse] * half + [0.0] * half)
    return _expand(per_bit)





ENCODERS = {
    "nrz": encode_nrz,  # polar / bipolar NRZ-L
    "nrz_unipolar": encode_nrz_unipolar,
    "nrzm": encode_nrzm,
    "rz_unipolar": encode_rz_unipolar,
    "bipolar_rz": encode_bipolar_rz,
    "manchester": encode_manchester,  # Bi-Phase
    "ami": encode_ami,
}


def encode(scheme: str, bits: list[int], samples_per_bit: int = SAMPLES_PER_BIT) -> np.ndarray:
    """Dispatch to an encoder by name. Raises ValueError on unknown scheme."""
    encoder = ENCODERS.get(scheme.lower())
    if encoder is None:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {sorted(ENCODERS)}")
    return encoder(bits, samples_per_bit)
