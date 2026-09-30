"""
Ground-truth oracle for the DSSS Modulator / Coherent Detector trainer kit.

Same role as app.oracle.encoders: pure NumPy, deterministic, no LLM calls.
Given a bitstream + spreading code + carrier parameters, this computes the
exact signal at each stage of the chain — the same computation the physical
trainer kit performs with its IC-based modules (74266 spreading mixer, 4052
bipolar converter, 1496 balanced modulator/detector, 741/311 LPF-comparator).

Chain modeled
-------------
NRZ data -> spread (XOR-equivalent, bipolar domain) -> modulate onto carrier
         -> [channel] -> coherent detect (multiply by same carrier)
         -> low-pass filter (chip-averaging) -> threshold -> recovered bits

This intentionally does NOT model component-level analog behaviour (no
SPICE, no op-amp non-idealities) — it models the same functional blocks the
trainer kit exposes as patchable test points. That matches what a student
actually interacts with: block inputs/outputs, not transistor-level circuits.
"""

from __future__ import annotations

import numpy as np


def _validate_bits(bits: list[int]) -> None:
    if not bits:
        raise ValueError("bits must be a non-empty sequence")
    for b in bits:
        if b not in (0, 1):
            raise ValueError(f"bits must contain only 0 or 1, found {b!r}")


def generate_pn_code(chip_rate: int, seed: int = 0) -> np.ndarray:
    """
    Deterministic pseudo-noise spreading code, one system-wide code of length
    `chip_rate` (chips per bit). Fixed `seed` so the code is reproducible —
    both transmitter (spreader) and receiver (detector) must use the same
    code, exactly like the trainer kit's shared spreading-code generator.
    """
    if chip_rate < 1:
        raise ValueError("chip_rate must be >= 1")
    rng = np.random.default_rng(seed)
    return rng.choice([1, -1], size=chip_rate).astype(float)


def spread(bits: list[int], pn_code: np.ndarray) -> np.ndarray:
    """
    Spreading Code Mixer + Uni-to-Bipolar Converter stage.

    Converts bits to bipolar NRZ (1 -> +1, 0 -> -1), then spreads each bit
    across `len(pn_code)` chips by multiplying with the PN code — bipolar
    multiplication is equivalent to XOR spreading in the unipolar domain.
    """
    _validate_bits(bits)
    nrz = np.where(np.array(bits) == 1, 1.0, -1.0)
    return np.concatenate([b * pn_code for b in nrz])


def modulate(spread_signal: np.ndarray, fs: float, fc: float, samples_per_chip: int = 10) -> np.ndarray:
    """
    DSSS Modulator stage (balanced modulator, IC 1496 equivalent).

    Upsamples the chip sequence and multiplies onto a carrier of frequency
    `fc`, sampled at `fs`. This is the RF-carrier mixing stage.
    """
    upsampled = np.repeat(spread_signal, samples_per_chip)
    t = np.arange(len(upsampled)) / fs
    carrier = np.sin(2 * np.pi * fc * t)
    return upsampled * carrier


def coherent_detect(
    tx_signal: np.ndarray, fs: float, fc: float, pn_code: np.ndarray,
    n_bits: int, samples_per_chip: int = 10,
) -> np.ndarray:
    """
    Coherent Detector + Low Pass Filter + Comparator stage.

    Multiplies the received signal by a *locally generated, phase-aligned*
    copy of the same carrier (the "coherent" part — this is what makes DSSS
    detection sensitive to carrier sync in the real kit), averages each
    chip's samples (the low-pass filter), despreads against the same PN
    code, then thresholds at zero (the comparator) to recover bits.
    """
    t = np.arange(len(tx_signal)) / fs
    local_carrier = np.sin(2 * np.pi * fc * t)
    demodulated = tx_signal * local_carrier

    chip_rate = len(pn_code)
    n_chips = n_bits * chip_rate
    recovered_chips = demodulated.reshape(n_chips, samples_per_chip).mean(axis=1)

    despread_energy = recovered_chips.reshape(n_bits, chip_rate) @ pn_code
    return (despread_energy > 0).astype(int)


def run_dsss_chain(
    bits: list[int], chip_rate: int = 4, fs: float = 2000.0, fc: float = 500.0,
    samples_per_chip: int = 10, pn_seed: int = 0,
) -> dict:
    """
    Runs the full chain and returns every intermediate stage — this is what
    the Wiring Kit plots at each test point, and what the Verifier diffs the
    LLM's claims against, stage by stage (not just the final recovered bits).
    """
    _validate_bits(bits)
    pn_code = generate_pn_code(chip_rate, seed=pn_seed)
    spread_signal = spread(bits, pn_code)
    tx_signal = modulate(spread_signal, fs, fc, samples_per_chip)
    recovered_bits = coherent_detect(tx_signal, fs, fc, pn_code, len(bits), samples_per_chip)

    return {
        "bits": bits,
        "pn_code": pn_code.tolist(),
        "spread_signal": spread_signal.tolist(),
        "tx_signal": tx_signal.tolist(),
        "recovered_bits": recovered_bits.tolist(),
        "bits_match": recovered_bits.tolist() == list(bits),
    }
