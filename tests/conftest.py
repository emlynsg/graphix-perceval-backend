"""Configuration file for testing backend interface between Graphix and Perceval.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

import pytest
from graphix.pattern import Pattern
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

SEED = 25


@pytest.fixture
def fx_bg() -> PCG64:
    """Fixture for bit generator.

    Returns
    -------
        Bit generator

    """
    return PCG64(SEED)


@pytest.fixture
def fx_rng(fx_bg: PCG64) -> Generator:
    """Fixture for generator.

    Returns
    -------
        Generator

    """
    return Generator(fx_bg)


@pytest.fixture
def hadamardpattern() -> Pattern:
    """Fixture for Hadamard pattern.

    Returns
    -------
        Hadamard pattern

    """
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern
