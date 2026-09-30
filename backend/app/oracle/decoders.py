"""
Decoding oracle — the software equivalent of the trainer kit's
"Comparator, Bit Clock Recovery, Bit Clock Decoder" section.

Each decoder takes a waveform (as produced by the matching encoder in
encoders.py) plus the known samples_per_bit, and recovers the original
bitstream. `samples_per_bit` stands in for what the kit's Bit Clock
Recovery module extracts from the signal itself — true blind clock
recovery (inferring bit timing with no prior knowledge) is a harder
problem and out of scope here; this oracle assumes correct bit-sync,
same as the physical kit does once its clock recovery has locked on.

Every decoder is deliberately the mathematical inverse of its encoder in
encoders.py — that pairing is what lets a round-trip test
(decode(encode(bits)) == bits) serve as the correctness guarantee, the
same role the AMI-polarity and DSSS full-chain tests play elsewhere.
"""

from __future__ import annotations

import numpy as np


def _validate_waveform(levels: np.ndarray, samples_per_bit: int) -> int:
    """Check the waveform length is a whole number of bits; return bit count."""
    if len(levels) == 0:
        raise ValueError("levels must be non-empty")
    if len(levels) % samples_per_bit != 0:
        raise ValueError(
            f"waveform length {len(levels)} is not a multiple of "
            f"samples_per_bit={samples_per_bit}"
        )
    return len(levels) // samples_per_bit


def decode_nrz(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """Inverse of encode_nrz. Threshold each bit's mean level at 0."""
    n_bits = _validate_waveform(levels, samples_per_bit)
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    return [1 if chunk.mean() > 0 else 0 for chunk in chunks]


def decode_nrz_unipolar(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """Inverse of encode_nrz_unipolar. Threshold at the midpoint, 0.5."""
    n_bits = _validate_waveform(levels, samples_per_bit)
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    return [1 if chunk.mean() > 0.5 else 0 for chunk in chunks]


def decode_nrzm(levels: np.ndarray, samples_per_bit: int, initial_level: float = -1.0) -> list[int]:
    """
    Inverse of encode_nrzm. Differential decode: a transition between
    consecutive bit levels means 1, no transition means 0.

    Critical: `initial_level` must match what the encoder used, or every
    single recovered bit comes out wrong (not just the first one) — since
    each bit's decoded value depends on comparing to the previous level,
    a wrong starting reference cascades through the whole sequence. This
    is a genuinely realistic trainer-kit fault: a decoder that's lost sync
    with the encoder's assumed starting state.
    """
    n_bits = _validate_waveform(levels, samples_per_bit)
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    bit_levels = [chunk.mean() for chunk in chunks]

    bits = []
    prev = initial_level
    for lvl in bit_levels:
        bits.append(1 if np.sign(lvl) != np.sign(prev) else 0)
        prev = lvl
    return bits


def decode_rz_unipolar(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """Inverse of encode_rz_unipolar. Only the first half-bit carries the pulse."""
    n_bits = _validate_waveform(levels, samples_per_bit)
    if samples_per_bit < 2 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be an even number >= 2 for RZ")
    half = samples_per_bit // 2
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    return [1 if chunk[:half].mean() > 0.5 else 0 for chunk in chunks]


def decode_bipolar_rz(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """Inverse of encode_bipolar_rz. Fixed polarity: positive first-half pulse = 1."""
    n_bits = _validate_waveform(levels, samples_per_bit)
    if samples_per_bit < 2 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be an even number >= 2 for RZ")
    half = samples_per_bit // 2
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    return [1 if chunk[:half].mean() > 0 else 0 for chunk in chunks]


def decode_manchester(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """
    Inverse of encode_manchester (IEEE 802.3 convention).

    1 = low-then-high, so the second half's level is greater than the
    first half's. Must match whichever Manchester convention the encoder
    used — see the convention warning in encoders.encode_manchester.
    """
    n_bits = _validate_waveform(levels, samples_per_bit)
    if samples_per_bit < 2 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be an even number >= 2 for Manchester")
    half = samples_per_bit // 2
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    bits = []
    for chunk in chunks:
        first_half, second_half = chunk[:half].mean(), chunk[half:].mean()
        bits.append(1 if second_half > first_half else 0)
    return bits


def decode_ami(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """Inverse of encode_ami. Any non-zero level (either polarity) is a 1."""
    n_bits = _validate_waveform(levels, samples_per_bit)
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    return [1 if abs(chunk.mean()) > 1e-9 else 0 for chunk in chunks]


DECODERS = {
    "nrz": decode_nrz,
    "nrz_unipolar": decode_nrz_unipolar,
    "nrzm": decode_nrzm,
    "rz_unipolar": decode_rz_unipolar,
    "bipolar_rz": decode_bipolar_rz,
    "manchester": decode_manchester,
    "ami": decode_ami,
}


def decode(scheme: str, levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """Dispatch to a decoder by name. Raises ValueError on unknown scheme."""
    decoder = DECODERS.get(scheme.lower())
    if decoder is None:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {sorted(DECODERS)}")
    return decoder(levels, samples_per_bit)


def check_ami_violation(levels: np.ndarray, samples_per_bit: int) -> list[int]:
    """
    Bonus diagnostic, not part of basic decoding: flags AMI Code Violations
    (CV) — two consecutive marks (1s) with the SAME polarity instead of
    alternating. In real systems a CV usually signals a transmission error;
    for this project it's the kind of check a future Diagnosis Agent could
    reuse to flag a specific, nameable fault instead of just "wrong."

    Returns the bit-index of each violation (empty list = clean AMI).
    """
    n_bits = _validate_waveform(levels, samples_per_bit)
    chunks = np.array(levels).reshape(n_bits, samples_per_bit)
    bit_levels = [chunk.mean() for chunk in chunks]

    violations = []
    last_mark_polarity = None
    for i, lvl in enumerate(bit_levels):
        if abs(lvl) > 1e-9:
            polarity = np.sign(lvl)
            if last_mark_polarity is not None and polarity == last_mark_polarity:
                violations.append(i)
            last_mark_polarity = polarity
    return violations
