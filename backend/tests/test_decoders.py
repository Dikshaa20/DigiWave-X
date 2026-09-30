"""
Tests for the decoding oracle.

The core guarantee tested here: decode(encode(bits)) == bits, for all 7
formats. This is the same role the AMI-polarity test and the DSSS
full-chain test play — a property that has to hold for the oracle to be
trustworthy, not just a check that the code does whatever it happens to do.

This file also demonstrates the thing worth being explicit about in the
report: every intermediate value (chunk means, per-bit levels, violation
indices) is a plain Python/NumPy value a LangGraph node could inspect
directly — nothing here is hidden inside an opaque simulator.
"""

import numpy as np
import pytest

from app.oracle.decoders import (
    check_ami_violation,
    decode,
    decode_ami,
    decode_bipolar_rz,
    decode_manchester,
    decode_nrz,
    decode_nrz_unipolar,
    decode_nrzm,
    decode_rz_unipolar,
)
from app.oracle.encoders import (
    encode_ami,
    encode_bipolar_rz,
    encode_manchester,
    encode_nrz,
    encode_nrz_unipolar,
    encode_nrzm,
    encode_rz_unipolar,
)

BITS = [1, 0, 1, 1, 0]
SPB = 4  # samples_per_bit; even, so RZ/Manchester's half-bit split is exact


# --------------------------------------------------------------------------
# Round-trip correctness — the property that actually matters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [
    [1], [0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 1],
])
def test_nrz_round_trip(bits):
    assert decode_nrz(encode_nrz(bits, SPB), SPB) == bits


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [1, 0, 1, 0, 1]])
def test_nrz_unipolar_round_trip(bits):
    assert decode_nrz_unipolar(encode_nrz_unipolar(bits, SPB), SPB) == bits


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]])
def test_nrzm_round_trip(bits):
    """NRZ-M round trip only holds if decoder's initial_level matches encoder's."""
    waveform = encode_nrzm(bits, SPB, initial_level=-1.0)
    assert decode_nrzm(waveform, SPB, initial_level=-1.0) == bits


def test_nrzm_mismatched_initial_level_breaks_round_trip():
    """
    Demonstrates the failure mode documented in decode_nrzm's docstring:
    a decoder out of sync with the encoder's assumed starting state gets
    every bit wrong, not just the first one.
    """
    waveform = encode_nrzm(BITS, SPB, initial_level=-1.0)
    wrong = decode_nrzm(waveform, SPB, initial_level=1.0)
    assert wrong != BITS


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [1, 0, 1, 0, 1]])
def test_rz_unipolar_round_trip(bits):
    assert decode_rz_unipolar(encode_rz_unipolar(bits, SPB), SPB) == bits


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [1, 0, 1, 0, 1]])
def test_bipolar_rz_round_trip(bits):
    assert decode_bipolar_rz(encode_bipolar_rz(bits, SPB), SPB) == bits


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [1, 0, 1, 0, 1]])
def test_manchester_round_trip(bits):
    assert decode_manchester(encode_manchester(bits, SPB), SPB) == bits


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1]])
def test_ami_round_trip(bits):
    assert decode_ami(encode_ami(bits, SPB), SPB) == bits


@pytest.mark.parametrize("scheme,enc,dec", [
    ("nrz", encode_nrz, decode_nrz),
    ("nrz_unipolar", encode_nrz_unipolar, decode_nrz_unipolar),
    ("rz_unipolar", encode_rz_unipolar, decode_rz_unipolar),
    ("bipolar_rz", encode_bipolar_rz, decode_bipolar_rz),
    ("manchester", encode_manchester, decode_manchester),
    ("ami", encode_ami, decode_ami),
])
def test_dispatch_round_trip(scheme, enc, dec):
    waveform = enc(BITS, SPB)
    assert decode(scheme, waveform, SPB) == dec(waveform, SPB) == BITS


# --------------------------------------------------------------------------
# Fault detection — what a Diagnosis Agent would actually call
# --------------------------------------------------------------------------

def test_ami_violation_detects_repeated_polarity():
    """
    Manually construct a waveform with a CV (code violation): two
    consecutive marks with the same polarity, which valid AMI never
    produces. [+1, 0, +1] is invalid AMI — the second mark should have
    been -1.
    """
    bad_waveform = np.array([1, 1, 0, 0, 1, 1])  # samples_per_bit=2, bits: 1,0,1
    violations = check_ami_violation(bad_waveform, samples_per_bit=2)
    assert violations == [2], "the second mark (bit index 2) should be flagged"


def test_ami_violation_clean_signal_has_none():
    clean_waveform = encode_ami(BITS, SPB)
    assert check_ami_violation(clean_waveform, SPB) == []


def test_ami_violation_returns_inspectable_indices_not_a_black_box():
    """
    The point being demonstrated: the violation check returns plain bit
    indices a LangGraph node can read and act on directly (e.g. "bit 2 has
    an AMI code violation"), not an opaque pass/fail from a simulator.
    """
    bad_waveform = np.array([1, 1, 0, 0, 1, 1])
    result = check_ami_violation(bad_waveform, samples_per_bit=2)
    assert isinstance(result, list)
    assert all(isinstance(i, (int, np.integer)) for i in result)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def test_decode_rejects_length_not_multiple_of_samples_per_bit():
    with pytest.raises(ValueError, match="not a multiple"):
        decode_nrz(np.array([1, 1, 1]), samples_per_bit=2)


def test_decode_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        decode_nrz(np.array([]), samples_per_bit=2)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError, match="unknown scheme"):
        decode("hdb3", np.array([1, 1]), samples_per_bit=2)
