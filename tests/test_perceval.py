"""Tests for backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

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
from veriphix.client import Client, Secrets
from veriphix.verifying import TrappifiedSchemeParameters

from graphix_perceval_backend import (
    PercevalBackend,
    graphix_state_to_perceval_statevec,
    perceval_single_qubit_statevector_to_graphix_statevec,
)


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
        percy = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend))
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
        percy = DensityMatrix(perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend)))
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
        percy1 = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1))
        percy2 = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2))
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
        percy1 = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1))
        percy2 = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2))
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
        percy1 = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1))
        percy2 = perceval_single_qubit_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2))
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

    # def test_init_fail(self, hadamardpattern, fx_rng: Generator) -> None:
    #     rand_angle = fx_rng.random(2) * 2 * np.pi
    #     rand_plane = fx_rng.choice(np.array(Plane), 2)

    #     state = PlanarState(rand_plane[0], rand_angle[0])
    #     state2 = PlanarState(rand_plane[1], rand_angle[1])
    #     with pytest.raises(ValueError):
    #         StatevectorBackend().add_nodes(hadamardpattern.input_nodes, data=[state, state2])

    # def test_clifford(self) -> None:
    #     for clifford in Clifford:
    #         state = BasicStates.PLUS
    #         vec = Statevec(nqubit=1, data=state)
    #         backend = StatevectorBackend()
    #         backend.add_nodes(nodes=[0], data=state)
    #         # Applies clifford gate "Z"
    #         vec.evolve_single(clifford.matrix, 0)
    #         backend.apply_clifford(node=0, clifford=clifford)
    #         np.testing.assert_allclose(vec.psi, backend.state.psi)

    # def test_deterministic_measure_one(self, fx_rng: Generator):
    #     # plus state & zero state (default), but with tossed coins
    #     for _ in range(10):
    #         backend = StatevectorBackend()
    #         coins = [fx_rng.choice([0, 1]), fx_rng.choice([0, 1])]
    #         expected_result = sum(coins) % 2
    #         states = [
    #             Pauli.X.eigenstate(coins[0]),
    #             Pauli.Z.eigenstate(coins[1]),
    #         ]
    #         nodes = range(len(states))
    #         backend.add_nodes(nodes=nodes, data=states)

    #         backend.entangle_nodes(edge=(nodes[0], nodes[1]))
    #         measurement = Measurement(0, Plane.XY)
    #         node_to_measure = backend.node_index[0]
    #         result = backend.measure(node=node_to_measure, measurement=measurement)
    #         assert result == expected_result

    # def test_deterministic_measure(self):
    #     """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
    #     for _ in range(10):
    #         # plus state (default)
    #         backend = StatevectorBackend()
    #         n_neighbors = 10
    #         states = [Pauli.X.eigenstate()] + [Pauli.Z.eigenstate() for i in range(n_neighbors)]
    #         nodes = range(len(states))
    #         backend.add_nodes(nodes=nodes, data=states)

    #         for i in range(1, n_neighbors + 1):
    #             backend.entangle_nodes(edge=(nodes[0], i))
    #         measurement = Measurement(0, Plane.XY)
    #         node_to_measure = backend.node_index[0]
    #         result = backend.measure(node=node_to_measure, measurement=measurement)
    #         assert result == 0
    #         assert list(backend.node_index) == list(range(1, n_neighbors + 1))

    # def test_deterministic_measure_many(self):
    #     """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
    #     for _ in range(10):
    #         # plus state (default)
    #         backend = StatevectorBackend()
    #         n_traps = 5
    #         n_neighbors = 5
    #         n_whatever = 5
    #         traps = [Pauli.X.eigenstate() for _ in range(n_traps)]
    #         dummies = [Pauli.Z.eigenstate() for _ in range(n_neighbors)]
    #         others = [Pauli.I.eigenstate() for _ in range(n_whatever)]
    #         states = traps + dummies + others
    #         nodes = range(len(states))
    #         backend.add_nodes(nodes=nodes, data=states)

    #         for dummy in nodes[n_traps : n_traps + n_neighbors]:
    #             for trap in nodes[:n_traps]:
    #                 backend.entangle_nodes(edge=(trap, dummy))
    #             for other in nodes[n_traps + n_neighbors :]:
    #                 backend.entangle_nodes(edge=(other, dummy))

    #         # Same measurement for all traps
    #         measurement = Measurement(0, Plane.XY)

    #         for trap in nodes[:n_traps]:
    #             node_to_measure = trap
    #             result = backend.measure(node=node_to_measure, measurement=measurement)
    #             assert result == 0

    #         assert list(backend.node_index) == list(range(n_traps, n_neighbors + n_traps + n_whatever))

    # def test_deterministic_measure_with_coin(self, fx_rng: Generator):
    #     """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.

    #     We add coin toss to that.
    #     """
    #     for _ in range(10):
    #         # plus state (default)
    #         backend = StatevectorBackend()
    #         n_neighbors = 10
    #         coins = [fx_rng.choice([0, 1])] + [fx_rng.choice([0, 1]) for _ in range(n_neighbors)]
    #         expected_result = sum(coins) % 2
    #         states = [Pauli.X.eigenstate(coins[0])] + [Pauli.Z.eigenstate(coins[i + 1]) for i in range(n_neighbors)]
    #         nodes = range(len(states))
    #         backend.add_nodes(nodes=nodes, data=states)

    #         for i in range(1, n_neighbors + 1):
    #             backend.entangle_nodes(edge=(nodes[0], i))
    #         measurement = Measurement(0, Plane.XY)
    #         node_to_measure = backend.node_index[0]
    #         result = backend.measure(node=node_to_measure, measurement=measurement)
    #         assert result == expected_result
    #         assert list(backend.node_index) == list(range(1, n_neighbors + 1))
