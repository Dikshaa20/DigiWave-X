"""Tests for the DSSS oracle chain."""

import numpy as np
import pytest

from app.oracle.dsss import coherent_detect, generate_pn_code, modulate, run_dsss_chain, spread

BITS = [1, 0, 1, 1, 0]


def test_pn_code_deterministic():
    """Same seed must give the same code — tx and rx have to agree."""
    a = generate_pn_code(4, seed=0)
    b = generate_pn_code(4, seed=0)
    assert a.tolist() == b.tolist()


def test_pn_code_is_bipolar():
    code = generate_pn_code(8, seed=1)
    assert set(code.tolist()) <= {1.0, -1.0}


def test_spread_length_scales_with_chip_rate():
    pn = generate_pn_code(4, seed=0)
    result = spread(BITS, pn)
    assert len(result) == len(BITS) * 4


def test_spread_bipolar_encoding():
    """A single chip PN code of [1] should just be bipolar NRZ."""
    pn = np.array([1.0])
    result = spread([1, 0, 1], pn)
    assert result.tolist() == [1.0, -1.0, 1.0]


def test_full_chain_recovers_original_bits():
    """
    The end-to-end property that actually matters: what goes in comes back
    out, exactly, when there's no noise in the channel. This is the
    equivalent of the AMI-polarity test for line coding — the core
    correctness guarantee of the whole oracle.
    """
    result = run_dsss_chain(BITS, chip_rate=4, fs=2000, fc=500, samples_per_chip=10)
    assert result["recovered_bits"] == BITS
    assert result["bits_match"] is True


@pytest.mark.parametrize("bits", [[1], [0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1]])
def test_chain_recovers_various_bit_patterns(bits):
    result = run_dsss_chain(bits, chip_rate=4, fs=2000, fc=500, samples_per_chip=10)
    assert result["recovered_bits"] == bits


def test_wrong_pn_code_fails_to_despread():
    """
    Sanity check on the detector: a receiver using an UNSYNCED spreading
    code should fail to recover the original bits. This is what your
    Diagnosis Agent would flag as a "PN code mismatch" fault later.

    A genuinely random mismatched code isn't reliable for this test — for
    short bit patterns it can coincidentally still despread correctly by
    chance, which would make this test flaky (it did, on first run: seed 99
    happened to recover BITS correctly anyway). An inverted code
    (`pn_rx = -pn_tx`) is used instead: this deterministically flips the
    sign of every despread symbol, so the recovered bits are guaranteed to
    be the exact bitwise complement of the original — not a matter of luck.
    """
    pn_tx = generate_pn_code(4, seed=0)
    pn_rx = -pn_tx
    spread_signal = spread(BITS, pn_tx)
    tx_signal = modulate(spread_signal, fs=2000, fc=500, samples_per_chip=10)
    recovered = coherent_detect(tx_signal, fs=2000, fc=500, pn_code=pn_rx, n_bits=len(BITS), samples_per_chip=10)
    assert recovered.tolist() == [1 - b for b in BITS], "inverted code should recover the exact bitwise complement"


def test_rejects_empty_bits():
    with pytest.raises(ValueError, match="non-empty"):
        run_dsss_chain([])


def test_rejects_non_binary_bits():
    with pytest.raises(ValueError, match="only 0 or 1"):
        run_dsss_chain([1, 2, 0])
