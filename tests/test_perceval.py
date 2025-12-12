"""Tests for backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

import numpy as np
import perceval as pcvl
import pytest
from graphix.sim import DensityMatrix, Statevec
from graphix.transpiler import Circuit
from perceval import Source
from veriphix.client import Client, Secrets
from veriphix.verifying import TrappifiedSchemeParameters

from graphix_perceval_backend import PercevalBackend


def convert_single_to_statevec(psvec: pcvl.StateVector) -> Statevec:
    """Convert a Perceval StateVector to a Graphix Statevec.

    Parameters
    ----------
    psvec : pcvl.StateVector
        Perceval StateVector to convert.

    Returns
    -------
    Statevec
        Graphix Statevec.

    """
    basic_states = [0.0 + 0.0j] * psvec.m
    for basic_state, amplitude in psvec:
        basic_states[basic_state.photon2mode(0)] = amplitude
    return Statevec(data=basic_states, nqubit=1)


class TestPercevalBackend:
    """Basic tests for the PercevalBackend class."""

    @staticmethod
    def test_with_veriphix() -> None:
        """Basic test for the PercevalBackend class with Veriphix."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        secrets = Secrets(r=True, a=True, theta=True)
        d = 10
        t = 10
        w = 1
        trap_scheme_param = TrappifiedSchemeParameters(d, t, w)
        client = Client(pattern=pattern, secrets=secrets, parameters=trap_scheme_param)
        protocol_runs = client.sample_canvas()
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        outcomes = client.delegate_canvas(protocol_runs, backend)  # pyright: ignore[reportArgumentType]
        result = client.analyze_outcomes(protocol_runs, outcomes)
        assert result[2].nr_failed_test_rounds == 0
        assert result[2].computation_outcomes_count["0"] == d

    @staticmethod
    def test_basic_vs_svec() -> None:
        """Test running simulation with PercevalBackend."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        percy = convert_single_to_statevec(pattern.simulate_pattern(backend))
        svec = circ.simulate_statevector().statevec
        assert np.abs(np.dot(percy.flatten().conjugate(), svec.flatten())) == pytest.approx(1)

    @staticmethod
    def test_basic_vs_dm() -> None:
        """Test running simulation with PercevalBackend."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern1 = pattern.copy()
        pattern.standardize()
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        percy = DensityMatrix(convert_single_to_statevec(pattern.simulate_pattern(backend)))
        dm: DensityMatrix = pattern1.simulate_pattern("densitymatrix")  # type: ignore  # noqa: PGH003
        assert np.allclose(dm.rho, percy.rho)

    @staticmethod
    def test_basic_diff_emission() -> None:
        """Test running simulation with PercevalBackend."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source1 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        source2 = Source(emission_probability=0.95, multiphoton_component=0, indistinguishability=1)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = convert_single_to_statevec(pattern.simulate_pattern(backend1))
        percy2 = convert_single_to_statevec(pattern.simulate_pattern(backend2))
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)
        #  TODO: Figure out how to define this test properly to reduce number of captured qubits.  # noqa: FIX002, TD002, TD003

    @staticmethod
    def test_basic_diff_multiphoton() -> None:
        """Test running simulation with PercevalBackend."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source1 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        source2 = Source(emission_probability=1, multiphoton_component=0.1, indistinguishability=1)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = convert_single_to_statevec(pattern.simulate_pattern(backend1))
        percy2 = convert_single_to_statevec(pattern.simulate_pattern(backend2))
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)
        #  TODO: Figure out how to define this test properly to reduce number of captured qubits.  # noqa: FIX002, TD002, TD003

    @staticmethod
    def test_basic_diff_indisting() -> None:
        """Test running simulation with PercevalBackend."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source1 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        source2 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=0.9)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = convert_single_to_statevec(pattern.simulate_pattern(backend1))
        percy2 = convert_single_to_statevec(pattern.simulate_pattern(backend2))
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)
        #  TODO: Figure out how to define this test properly to reduce number of captured qubits.  # noqa: FIX002, TD002, TD003
