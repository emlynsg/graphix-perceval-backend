"""Tests for backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2026, QAT team (ENS-PSL, Inria, CNRS).
"""

# ruff: noqa
# ruff: noqa: PGH004

import graphix.pauli
import graphix.fundamentals

# from graphix.fundamentals import ANGLE_PI
from math import pi

ANGLE_PI = pi

import numpy as np
import pytest
from graphix.sim.statevec import StatevectorBackend, Statevec
from graphix.sim.density_matrix import DensityMatrix
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
    graphix_state_to_perceval_statevec,
    perceval_statevector_to_graphix_statevec,
)


# # Monkeypatch for graphix.pauli.IXYZ which has been replaced in recent graphix master
# # and is now a TypeAlias (Union) which cannot be instantiated.
# # veriphix (dev dependency) still expects a callable Enum-like object.
# # TODO: Update Veriphix to use current form of IXYZ, and remove this monkeypatch
# class IXYZ_Meta(type):
#     def __getattr__(cls, name):
#         # Fallback to IXYZ_VALUES for enum-like access (IXYZ.X, IXYZ.Y, IXYZ.Z)
#         if hasattr(graphix.fundamentals, "IXYZ_VALUES"):
#             # I=0, X=1, Y=2, Z=3 in IXYZ_VALUES based on previous context
#             # But let's check standard names
#             if name == "I":
#                 return graphix.fundamentals.IXYZ_VALUES[0]
#             if name == "X":
#                 return graphix.fundamentals.IXYZ_VALUES[1]
#             if name == "Y":
#                 return graphix.fundamentals.IXYZ_VALUES[2]
#             if name == "Z":
#                 return graphix.fundamentals.IXYZ_VALUES[3]

#         if hasattr(graphix.fundamentals, name):
#             return getattr(graphix.fundamentals, name)
#         # Fallback for older graphix where they are members of IXYZ Enum
#         if hasattr(graphix.fundamentals, "IXYZ") and hasattr(graphix.fundamentals.IXYZ, name):
#             return getattr(graphix.fundamentals.IXYZ, name)
#         raise AttributeError(name)

#     def __getitem__(cls, key):
#         return getattr(cls, key)

#     def __iter__(cls):
#         if hasattr(graphix.fundamentals, "IXYZ_VALUES"):
#             return iter(graphix.fundamentals.IXYZ_VALUES)
#         return iter(graphix.fundamentals.IXYZ)

#     def __call__(cls, arg):
#         if hasattr(graphix.fundamentals, "IXYZ_VALUES"):
#             try:
#                 # Fix: Handle 1-based indexing from older Enum behavior (I=1 -> index 0)
#                 index = arg - 1 if isinstance(arg, int) and arg > 0 else arg
#                 return graphix.fundamentals.IXYZ_VALUES[index]
#             except (IndexError, TypeError):
#                 pass
#         if hasattr(graphix.fundamentals, "IXYZ") and callable(graphix.fundamentals.IXYZ):
#             try:
#                 return graphix.fundamentals.IXYZ(arg)
#             except TypeError:
#                 pass
#         if hasattr(graphix.fundamentals, "IXYZ_VALUES"):
#             return graphix.fundamentals.IXYZ_VALUES[0]
#         return arg

#     def __instancecheck__(cls, instance):
#         if hasattr(graphix.fundamentals, "Axis") and isinstance(instance, graphix.fundamentals.Axis):
#             return True
#         if hasattr(graphix.fundamentals, "IXYZ"):
#             if isinstance(graphix.fundamentals.IXYZ, type):
#                 return isinstance(instance, graphix.fundamentals.IXYZ)
#             pass
#         if hasattr(graphix.fundamentals, "I") and instance is graphix.fundamentals.I:
#             return True
#         return False


# class IXYZ_Shim(metaclass=IXYZ_Meta):
#     pass


# graphix.pauli.IXYZ = IXYZ_Shim


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


# class TestVeriphixProblems:
#     @staticmethod
#     def test_without_theta_and_secret():
#         circ = Circuit(1)
#         circ.h(0)
#         # circ.h(0)
#         pattern = circ.transpile().pattern
#         pattern.standardize()

#         print("Original pattern:")
#         for cmd in pattern:
#             print(f"  {cmd}")

#         secrets = Secrets(r=False, a=False, theta=False)
#         d, t, w = 3, 0, 0
#         trap_scheme_param = TrappifiedSchemeParameters(d, t, w)
#         client = Client(pattern=pattern, secrets=secrets, parameters=trap_scheme_param)

#         # Run directly with pattern.simulate_pattern for comparison
#         print("\n--- Direct simulation (3 runs) ---")
#         source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
#         for i in range(3):
#             backend = PercevalBackend(source)
#             state = pattern.simulate_pattern(backend)
#             vec = perceval_statevector_to_graphix_statevec(state)
#             comparison = np.array([1, 0])  # |0>
#             assert np.abs(np.dot(vec.psi.flatten().conjugate(), comparison)) == pytest.approx(1)
#         # TODO: Something wrong with the shape of the converted statevec, coming from something wrong with the Perceval statevec itself

#         # Now run through veriphix
#         print("\n--- Veriphix execution ---")
#         backend = PercevalBackend(source)
#         protocol_runs = client.sample_canvas()

#         print(f"\nNumber of protocol runs: {len(protocol_runs)}")
#         print(f"First run type: {type(protocol_runs[0])}")

#         print("\nInitial pattern in Veriphix:")
#         for cmd in client.initial_pattern:
#             print(f"  {cmd}")
#         print("\nClean pattern in Veriphix:")
#         for cmd in client.clean_pattern:
#             print(f"  {cmd}")

#         print(f"\nByproduct DB: {client.byproduct_db}")
#         print(f"Output nodes: {client.output_nodes}")
#         print(f"Input nodes: {client.input_nodes}")

#         outcomes = client.delegate_canvas(protocol_runs, backend)
#         result = client.analyze_outcomes(protocol_runs, outcomes)

#         #  Debug output
#         print(f"Computation outcomes: {result[2].computation_outcomes_count}")  # Should be d
#         print(f"Test round failures: {result[2].nr_failed_test_rounds}")  # Should be 0
#         print(f"Total computation rounds: {d}")

#         assert result[2].nr_failed_test_rounds == 0
#         assert result[2].computation_outcomes_count["0"] == d

#     @staticmethod
#     def test_Veriphix_statevec():
#         """Same as Veriphix test in the main test file, but using Statevector."""
#         circ = Circuit(1)
#         circ.h(0)
#         circ.h(0)
#         pattern = circ.transpile().pattern
#         pattern.standardize()
#         backend = StatevectorBackend()
#         state = pattern.simulate_pattern(backend)
#         vec = perceval_statevector_to_graphix_statevec(state)
#         print("Original pattern:")
#         for cmd in pattern:
#             print(f"  {cmd}")

#         secrets = Secrets(r=True, a=True, theta=True)
#         d = 10
#         t = 10
#         w = 1
#         trap_scheme_param = TrappifiedSchemeParameters(d, t, w)
#         client = Client(pattern=pattern, secrets=secrets, parameters=trap_scheme_param)
#         protocol_runs = client.sample_canvas()

#         print(f"\nNumber of protocol runs: {len(protocol_runs)}")
#         print(f"First run type: {type(protocol_runs[0])}")

#         print("\nInitial pattern in Veriphix:")
#         for cmd in client.initial_pattern:
#             print(f"  {cmd}")
#         print("\nClean pattern in Veriphix:")
#         for cmd in client.clean_pattern:
#             print(f"  {cmd}")

#         print(f"\nByproduct DB: {client.byproduct_db}")
#         print(f"Output nodes: {client.output_nodes}")
#         print(f"Input nodes: {client.input_nodes}")

#         outcomes = client.delegate_canvas(protocol_runs, StatevectorBackend)  # pyright: ignore[reportArgumentType]
#         result = client.analyze_outcomes(protocol_runs, outcomes)

#         #  Debug output
#         print(f"Computation outcomes: {result[2].computation_outcomes_count}")  # Should be d
#         print(f"Test round failures: {result[2].nr_failed_test_rounds}")  # Should be 0
#         print(f"Total computation rounds: {d}")

#         assert result[2].nr_failed_test_rounds == 0
#         assert result[2].computation_outcomes_count["0"] == d


class TestPercevalBackend:
    """Basic tests for the PercevalBackend class."""

    @staticmethod
    def test_h_deterministic() -> None:
        """Verify H gives deterministic "0" outcome."""
        circ = Circuit(1)
        # circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()

        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)

        # Run 20 times, should always get same result
        results = []
        for _ in range(20):
            backend = PercevalBackend(source)
            state = pattern.simulate_pattern(backend).state
            # Convert to computational basis outcome
            vec = perceval_statevector_to_graphix_statevec(state)
            comparison = np.array([1, 0])  # |0>
            assert np.abs(np.dot(vec.psi.flatten().conjugate(), comparison)) == pytest.approx(1)

    @staticmethod
    def test_check_veriphix() -> None:
        """Verify PercevalBackend integration with the Veriphix trappified verification scheme."""
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

        outcomes = client.delegate_canvas(protocol_runs, StatevectorBackend)  # pyright: ignore[reportArgumentType]
        result = client.analyze_outcomes(protocol_runs, outcomes)

        assert result[2].nr_failed_test_rounds == 0
        assert result[2].computation_outcomes_count["0"] == d

    @staticmethod
    def test_with_veriphix() -> None:
        """Verify PercevalBackend integration with the Veriphix trappified verification scheme."""
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
        backend = lambda: PercevalBackend(source)
        outcomes = client.delegate_canvas(protocol_runs, backend)  # pyright: ignore[reportArgumentType]
        result = client.analyze_outcomes(protocol_runs, outcomes)

        assert result[2].nr_failed_test_rounds == 0
        assert result[2].computation_outcomes_count["0"] == d

    @staticmethod
    def test_basic_vs_svec() -> None:
        """Verify that PercevalBackend results match Graphix StatevectorBackend for a simple circuit."""
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        percy = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend).state)
        svec = circ.simulate_statevector().statevec
        assert np.abs(np.dot(percy.flatten().conjugate(), svec.flatten())) == pytest.approx(1)

    @staticmethod
    def test_basic_vs_dm() -> None:
        """Verify that PercevalBackend results match Graphix DensityMatrix backend for a simple circuit."""
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        percy = DensityMatrix(perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend).state))
        dm: DensityMatrix = pattern.simulate_pattern("densitymatrix")  # type: ignore  # noqa: PGH003
        assert np.allclose(dm.rho, percy.rho)

    @staticmethod
    def test_basic_diff_emission() -> None:
        """Verify that varying the source emission probability does not break the simulation logic."""
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source1 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        source2 = Source(emission_probability=0.99, multiphoton_component=0, indistinguishability=1)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1).state)
        percy2 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2).state)
        assert np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten())) == pytest.approx(1)

    @staticmethod
    def test_basic_diff_multiphoton() -> None:
        """Verify that varying the source multiphoton component does not break the simulation logic."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source1 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        source2 = Source(emission_probability=1, multiphoton_component=0.02, indistinguishability=1)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1).state)
        percy2 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2).state)
        assert np.abs(percy2.psi.flatten().conjugate(), percy2.psi.flatten()) != pytest.approx(1)
        # Multiphoton component should theoretically introduce noise.
        fidelity = np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten()))
        assert fidelity > 0.6

    @staticmethod
    def test_basic_diff_indisting() -> None:
        """Verify that varying the photon indistinguishability does not break the simulation logic."""
        # client computation pattern definition
        circ = Circuit(1)
        circ.h(0)
        circ.h(0)
        pattern = circ.transpile().pattern
        pattern.standardize()
        source1 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        source2 = Source(emission_probability=1, multiphoton_component=0, indistinguishability=0.95)
        backend1 = PercevalBackend(source1)
        backend2 = PercevalBackend(source2)
        percy1 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend1).state)
        percy2 = perceval_statevector_to_graphix_statevec(pattern.simulate_pattern(backend2).state)

        # Indistinguishability < 1 introduces noise/mixed states, so fidelity should drop
        fidelity = np.abs(np.dot(percy1.flatten().conjugate(), percy2.flatten()))
        assert fidelity > 0.8

    @pytest.mark.parametrize(
        "state",
        [
            BasicStates.PLUS,
            BasicStates.MINUS,
            BasicStates.ZERO,
            BasicStates.ONE,
            BasicStates.PLUS_I,
            BasicStates.MINUS_I,
        ],
    )
    def test_init_success(self, hadamardpattern, state: PlanarState) -> None:
        """Verify successful initialization of PercevalBackend with various basic states."""
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        backend = PercevalBackend(source)
        backend.add_nodes(hadamardpattern.input_nodes, data=state)

        # Verify backend was initialized correctly
        assert backend.nqubit == 1
        assert len(list(backend.node_index)) == 1
        assert 0 in backend.node_index

        # Verify state is not empty
        assert backend.state is not None
        assert backend.state.m == 2  # 2 modes for 1 qubit

        # Verify state content
        percy_vec = perceval_statevector_to_graphix_statevec(backend.state)
        target_psi = Statevec(nqubit=1, data=state).psi
        assert np.allclose(target_psi.flatten(), percy_vec.psi.flatten())

    def test_init_planar(self, hadamardpattern, fx_rng: Generator) -> None:
        """Verify initialization of PercevalBackend using PlanarState inputs."""
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)

        # random planar state
        rand_angle = fx_rng.random() * 2 * np.pi
        rand_plane = fx_rng.choice(np.array(Plane))
        state = PlanarState(rand_plane, rand_angle)
        backend = PercevalBackend(source)
        backend.add_nodes(hadamardpattern.input_nodes, data=state)
        percy_vec = perceval_statevector_to_graphix_statevec(backend.state)
        target_psi = state.get_statevector()
        assert np.allclose(target_psi.flatten(), percy_vec.psi.flatten())

    def test_init_svec(self, hadamardpattern) -> None:
        """Verify initialization of PercevalBackend using Statevec inputs."""
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)

        # Statevec input
        vec_input = Statevec(nqubit=1, data=BasicStates.PLUS)
        backend = PercevalBackend(source)
        backend.add_nodes(hadamardpattern.input_nodes, data=vec_input)
        percy_vec = perceval_statevector_to_graphix_statevec(backend.state)
        assert np.allclose(vec_input.psi.flatten(), percy_vec.psi.flatten())

    def test_init_fail(self, hadamardpattern, fx_rng: Generator) -> None:
        """Verify that initialization fails correctly when the number of input states does not match the nodes."""
        rand_angle = fx_rng.random(2) * 2 * ANGLE_PI
        rand_plane = fx_rng.choice(np.array(Plane), 2)

        state = PlanarState(rand_plane[0], rand_angle[0])
        state2 = PlanarState(rand_plane[1], rand_angle[1])
        source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
        with pytest.raises(ValueError, match="Length mismatch"):
            PercevalBackend(source).add_nodes(hadamardpattern.input_nodes, data=[state, state2])

    def test_clifford(self) -> None:
        """Verify the correct application of all single-qubit Clifford gates on PercevalBackend."""
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
            assert np.abs(np.dot(percy_vec.psi.flatten().conjugate(), vec.psi.flatten())) == pytest.approx(1)

    def test_deterministic_measure_one(self, fx_rng: Generator):
        """Verify deterministic measurement outcomes for a 2-qubit entangled system."""
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
        """Verify deterministic measurement for a star-shaped graph with 5 qubits."""
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
        """Verify deterministic measurement outcomes in a more complex system."""
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
