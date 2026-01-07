"""Tests for backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

# ruff: noqa
# ruff: noqa: PGH004

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import perceval as pcvl
import pytest
from graphix.sim import DensityMatrix, Statevec
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates, PlanarState
from graphix.transpiler import Circuit
from numpy.random import Generator
from perceval import Source
from graphix.fundamentals import Plane
from graphix.pauli import Pauli
from graphix.measurements import Measurement
from graphix.clifford import Clifford
from veriphix.client import Client, Secrets
from veriphix.verifying import TrappifiedSchemeParameters

from graphix_perceval_backend import (
    PercevalBackend,
    graphix_planar_state_to_perceval_statevec,
    graphix_state_to_perceval_statevec,
    perceval_statevector_to_graphix_statevec,
)


class TestConversionFunctions:
    """Test state conversion between Graphix and Perceval."""

    @staticmethod
    def test_plus_state():
        """Test |+⟩ state conversion."""
        graphix_plus = BasicStates.PLUS.get_statevector()
        perceval_state = graphix_state_to_perceval_statevec(graphix_plus)
        back_to_graphix = perceval_statevector_to_graphix_statevec(perceval_state)
        # Check that back_to_graphix is a Statevec and compare data
        assert np.allclose(graphix_plus, back_to_graphix.psi.flatten())

    @staticmethod
    def test_zero_state():
        """Test |0⟩ state conversion."""
        graphix_zero = BasicStates.ZERO.get_statevector()
        perceval_state = graphix_state_to_perceval_statevec(graphix_zero)
        back_to_graphix = perceval_statevector_to_graphix_statevec(perceval_state)
        assert np.allclose(graphix_zero, back_to_graphix.psi.flatten())

    @staticmethod
    def test_one_state():
        """Test |1⟩ state conversion."""
        graphix_one = BasicStates.ONE.get_statevector()
        perceval_state = graphix_state_to_perceval_statevec(graphix_one)
        back_to_graphix = perceval_statevector_to_graphix_statevec(perceval_state)
        assert np.allclose(graphix_one, back_to_graphix.psi.flatten())

    @staticmethod
    def test_bell_state():
        """Test Bell state conversion."""
        # |00> + |11>
        bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        perceval_state = graphix_state_to_perceval_statevec(bell)
        # Verify nphoton = 2
        # Use basic_state.n which returns number of photons
        for basic_state, _ in perceval_state:
            assert basic_state.n == 2

        # Convert back
        back_to_graphix = perceval_statevector_to_graphix_statevec(perceval_state)
        assert np.allclose(bell, back_to_graphix.psi.flatten())


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
        percy = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend))
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
        percy = DensityMatrix(perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend)))
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
        source2 = Source(emission_probability=0.99, multiphoton_component=0, indistinguishability=1)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1))
        percy2 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2))
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)
        #  TODO: Figure out how to define this test to reduce number of captured qubits.  # noqa: FIX002, TD002, TD003

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
        percy1 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1))
        percy2 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2))
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)
        #  TODO: Figure out how to define this test to reduce number of captured qubits.  # noqa: FIX002, TD002, TD003

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
        percy1 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1))
        percy2 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2))
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)
        #  TODO: Figure out how to define this test to reduce number of captured qubits.  # noqa: FIX002, TD002, TD003

    # @pytest.mark.parametrize("state", [BasicStates.PLUS, BasicStates.ZERO, BasicStates.ONE
    #                                    , BasicStates.PLUS_I, BasicStates.MINUS_I])
    def test_init_success(self, hadamardpattern, fx_rng: Generator) -> None:  # noqa: ANN001
        """Test successful initialization of backend with nodes."""
        # Test with plus state (default)
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        backend.add_nodes(hadamardpattern.input_nodes)

        # Verify backend was initialized correctly
        assert backend.nqubit == 1
        assert len(list(backend.node_index)) == 1
        assert 0 in backend.node_index

        # Verify state is not empty
        assert backend.state is not None
        assert backend.state.m == 2  # 2 modes for 1 qubit (path encoding)

    #     # minus state
    #     backend = StatevectorBackend()
    #     backend.add_nodes(hadamardpattern.input_nodes, data=BasicStates.MINUS)
    #     vec = Statevec(nqubit=1, data=BasicStates.MINUS)
    #     assert np.allclose(vec.psi, backend.state.psi)
    #     assert len(backend.state.dims()) == 1

    #     # random planar state
    #     rand_angle = fx_rng.random() * 2 * np.pi
    #     rand_plane = fx_rng.choice(np.array(Plane))
    #     state = PlanarState(rand_plane, rand_angle)
    #     backend = StatevectorBackend()
    #     backend.add_nodes(hadamardpattern.input_nodes, data=state)
    #     vec = Statevec(nqubit=1, data=state)
    #     assert np.allclose(vec.psi, backend.state.psi)
    #     # assert backend.state.nqubit == 1
    #     assert len(backend.state.dims()) == 1

    #     # data input and Statevec input

    def test_init_fail(self, hadamardpattern, fx_rng: Generator) -> None:
        """Test that initialization fails with incorrect number of states."""
        rand_angle = fx_rng.random(2) * 2 * np.pi
        rand_plane = fx_rng.choice([Plane.XY, Plane.YZ, Plane.XZ], 2)

        state = PlanarState(rand_plane[0], rand_angle[0])
        state2 = PlanarState(rand_plane[1], rand_angle[1])
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        # add_nodes should raise ValueError if list length doesn't match
        with pytest.raises(ValueError, match="Length mismatch"):
            PercevalBackend(source).add_nodes(hadamardpattern.input_nodes, data=[state, state2])

    def test_clifford(self) -> None:
        """Test single-qubit Clifford gate application."""
        for clifford in Clifford:
            state = BasicStates.PLUS
            # Reference Statevector result
            vec = Statevec(nqubit=1, data=state)
            vec.evolve_single(clifford.matrix, 0)

            # PercevalBackend result
            source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
            backend = PercevalBackend(source)
            backend.add_nodes(nodes=[0], data=state)
            backend.apply_clifford(node=0, clifford=clifford)

            percy_vec = perceval_statevector_to_graphix_statevec(backend.state)

            # Check overlap
            assert np.abs(np.dot(percy_vec.psi.flatten().conjugate(), vec.psi.flatten())) == pytest.approx(1)

    def test_deterministic_measure_one(self, fx_rng: Generator):
        """Test deterministic measurement result."""
        # plus state & zero state (default), but with tossed coins
        for _ in range(5):
            source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
            backend = PercevalBackend(source)
            coins = [fx_rng.choice([0, 1]), fx_rng.choice([0, 1])]
            expected_result = sum(coins) % 2
            states = [
                Pauli.X.eigenstate(coins[0]),
                Pauli.Z.eigenstate(coins[1]),
            ]
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            backend.entangle_nodes(edge=(nodes[0], nodes[1]))
            measurement = Measurement(plane=Plane.XY, angle=0)
            node_to_measure = backend.node_index[0]
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == expected_result

    def test_deterministic_measure(self):
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
        for _ in range(3):
            # plus state (default)
            source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
            backend = PercevalBackend(source)
            n_neighbors = 4
            states = [Pauli.X.eigenstate()] + [Pauli.Z.eigenstate() for i in range(n_neighbors)]
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            for i in range(1, n_neighbors + 1):
                backend.entangle_nodes(edge=(nodes[0], i))
            measurement = Measurement(plane=Plane.XY, angle=0)
            node_to_measure = backend.node_index[0]
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == 0
            assert list(backend.node_index) == list(range(1, n_neighbors + 1))

    def test_deterministic_measure_many(self):
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
        for _ in range(3):
            # plus state (default)
            source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
            backend = PercevalBackend(source)
            n_traps = 2
            n_neighbors = 2
            n_whatever = 2
            traps = [Pauli.X.eigenstate() for _ in range(n_traps)]
            dummies = [Pauli.Z.eigenstate() for _ in range(n_neighbors)]
            others = [Pauli.Z.eigenstate() for _ in range(n_whatever)]
            states = traps + dummies + others
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            for dummy in nodes[n_traps : n_traps + n_neighbors]:
                for trap in nodes[:n_traps]:
                    backend.entangle_nodes(edge=(trap, dummy))
                for other in nodes[n_traps + n_neighbors :]:
                    backend.entangle_nodes(edge=(other, dummy))

            # Same measurement for all traps
            measurement = Measurement(plane=Plane.XY, angle=0)

            for trap in nodes[:n_traps]:
                node_to_measure = trap
                result = backend.measure(node=node_to_measure, measurement=measurement)
                assert result == 0

            assert list(backend.node_index) == list(range(n_traps, n_neighbors + n_traps + n_whatever))

    def test_deterministic_measure_with_coin(self, fx_rng: Generator):
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.

        We add coin toss to that.
        """
        for _ in range(3):
            # plus state (default)
            source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
            backend = PercevalBackend(source)
            n_neighbors = 4
            coins = [fx_rng.choice([0, 1])] + [fx_rng.choice([0, 1]) for _ in range(n_neighbors)]
            expected_result = sum(coins) % 2
            states = [Pauli.X.eigenstate(coins[0])] + [Pauli.Z.eigenstate(coins[i + 1]) for i in range(n_neighbors)]
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            for i in range(1, n_neighbors + 1):
                backend.entangle_nodes(edge=(nodes[0], i))
            measurement = Measurement(plane=Plane.XY, angle=0)
            node_to_measure = backend.node_index[0]
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == expected_result
            assert list(backend.node_index) == list(range(1, n_neighbors + 1))
