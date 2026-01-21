"""Backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

# ruff: noqa: ANN001

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import perceval as pcvl
import perceval.components as comp
from graphix.command import CommandKind
from graphix.fundamentals import Plane
from graphix.sim.base_backend import Backend, NodeIndex
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates, PlanarState
from perceval.components import catalog

PRECISION = 1e-15

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from graphix.clifford import Clifford
    from graphix.measurements import Measurement
    from numpy.random import Generator


class PercevalState:
    """Quantum state management using Perceval.

    This class wraps pcvl.StateVector and provides methods for evolution and measurement,
    mimicking the Graphix Statevec interface while using Perceval's simulation capabilities.
    """

    def __init__(self, source: pcvl.Source, state: pcvl.StateVector | None = None) -> None:
        """Initialize PercevalState.

        Parameters
        ----------
        source : pcvl.Source
            Photon source configuration.
        state : pcvl.StateVector | None
            Initial state (default: empty BasicState).

        """
        self.source = source
        self.pcvl_state = state if state is not None else pcvl.BasicState()
        self._sim = pcvl.simulators.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        self._sim.set_min_detected_photons_filter(0)
        self._sim.keep_heralds(False)  # noqa: FBT003

    @property
    def nqubit(self) -> int:
        """Return the number of qubits (modes // 2)."""
        return self.pcvl_state.m // 2

    @property
    def m(self) -> int:
        """Return the number of optical modes."""
        return self.pcvl_state.m

    def __iter__(self) -> Iterator:
        """Iterate over the underlying Perceval StateVector.

        Returns
        -------
        iterator
            Iterator over the Perceval StateVector.

        """
        return iter(self.pcvl_state)

    def evolve(self, circuit: pcvl.Circuit, heralds: dict | None = None) -> None:
        """Evolve the state using a Perceval circuit.

        Parameters
        ----------
        circuit : pcvl.Circuit
            The circuit to apply.
        heralds : dict | None
            Optional heralds configuration.

        """
        self._sim.set_circuit(circuit)
        if heralds:
            self._sim.set_heralds(heralds)
            self.pcvl_state = self._sim.evolve(self.pcvl_state)
            self._sim.clear_heralds()
        else:
            self.pcvl_state = self._sim.evolve(self.pcvl_state)

    def measure(self, modes: list[int], postselect: pcvl.PostSelect | None = None) -> pcvl.BasicState:
        """Measure the specified modes and update the state to the conditional outcome.

        Parameters
        ----------
        modes : list[int]
            Indices of modes to measure.
        postselect : pcvl.PostSelect | None
            Optional post-selection condition.

        Returns
        -------
        pcvl.BasicState
            The sampled measurement outcome.

        """
        all_possible_meas_outcomes = self.pcvl_state.measure(modes)
        outcome_dist = {outcome: res[0] for outcome, res in all_possible_meas_outcomes.items()}
        outcomes = pcvl.BSDistribution(outcome_dist)

        if postselect:
            ps_outcomes = pcvl.utils.postselect.post_select_distribution(outcomes, postselect)[0]
            sampled_outcome = ps_outcomes.sample(1)[0]
        else:
            sampled_outcome = outcomes.sample(1)[0]

        self.pcvl_state = all_possible_meas_outcomes[sampled_outcome][1]
        return sampled_outcome

    def add_qubit(self, psi: npt.NDArray) -> None:
        """Add a new path-encoded qubit prepared in the given statevector psi.

        Parameters
        ----------
        psi : npt.NDArray
            2-element statevector.

        """
        zero_mixed_state = self.source.generate_distribution(pcvl.BasicState([1, 0]))
        init_qubit = zero_mixed_state.sample(1)[0]

        alpha = psi[0]
        beta = psi[1]
        if np.abs(beta) > PRECISION:
            gamma = np.abs(beta)
            delta = -np.conjugate(alpha) * gamma / np.conjugate(beta)
            matrix = pcvl.Matrix(np.asarray([[alpha, gamma], [beta, delta]]))
            init_circ = pcvl.Circuit(2)
            init_circ.add(0, comp.Unitary(U=matrix))
            self._sim.set_circuit(init_circ)
            init_qubit = self._sim.evolve(init_qubit)
        self.pcvl_state *= init_qubit

    def entangle(self, index_0: int, index_1: int) -> None:
        """Apply a heralded CZ gate between two qubits.

        Parameters
        ----------
        index_0 : int
            Logical index of first qubit.
        index_1 : int
            Logical index of second qubit.

        """
        ctrl = min(index_0, index_1)
        target = max(index_0, index_1)
        cz_input_modes = [2 * ctrl, 2 * ctrl + 1, 2 * target, 2 * target + 1]

        ent_proc = pcvl.Processor("SLOS", 2 * self.nqubit)
        ent_proc.add(cz_input_modes, catalog["heralded cz"].build_processor())
        ent_circ = ent_proc.linear_circuit()

        heralds = dict.fromkeys(list(range(2 * self.nqubit, ent_circ.m)), 1)
        herald_state_dist = self.source.generate_distribution(pcvl.BasicState([1, 1]))
        sampled_herald_state = herald_state_dist.sample(1)[0]

        self.pcvl_state *= sampled_herald_state
        self.evolve(ent_circ, heralds=heralds)

    def measure_qubit(self, index: int, circuit: pcvl.Circuit) -> int:
        """Measure a qubit after applying a measurement basis circuit.

        Parameters
        ----------
        index : int
            Logical index of the qubit to measure.
        circuit : pcvl.Circuit
            Circuit representing the basis change.

        Returns
        -------
        int
            Measurement outcome (0 or 1).

        """
        self.evolve(circuit)
        ps = pcvl.PostSelect("([0] > 0 & [1] == 0) | ([0] == 0 & [1] > 0)")
        sampled_outcome = self.measure([2 * index, 2 * index + 1], postselect=ps)

        # logical |0> is |1, 0> (photon in mode 0) -> result 0
        # logical |1> is |0, 1> (photon in mode 1) -> result 1
        return 0 if sampled_outcome[0] == 1 else 1

    def copy(self) -> PercevalState:
        """Return a deep copy of the PercevalState.

        Returns
        -------
        PercevalState
            Deep copy of the PercevalState.

        """
        new_state = deepcopy(self.pcvl_state)
        new_source = deepcopy(self.source)
        return PercevalState(new_source, new_state)

    def to_graphix_statevec(self) -> Statevec:
        """Convert to a Graphix Statevec object.

        Returns
        -------
        Statevec
            Graphix Statevec object.

        """
        return perceval_statevector_to_graphix_statevec(self.pcvl_state)


class PercevalBackend(Backend):
    """Backend interface between Graphix and Quandela's Perceval package for pattern simulation."""

    def __init__(self, source: pcvl.Source, state: PercevalState | pcvl.StateVector | None = None) -> None:
        """Initialize PercevalBackend.

        Parameters
        ----------
        source : pcvl.Source
            Photon source configuration.
        state : PercevalState | pcvl.StateVector | None
            Initial state (default: empty).

        """
        if isinstance(state, PercevalState):
            self._state = state
        elif isinstance(state, pcvl.StateVector):
            self._state = PercevalState(source, state)
        else:
            self._state = PercevalState(source)
        self.node_index = NodeIndex()

    def __call__(self) -> PercevalBackend:
        """Return a copy of the PercevalBackend object."""  # noqa: DOC201
        return self

    @property
    def source(self) -> pcvl.Source:
        """Return the photon source of the backend."""
        return self._state.source

    @property
    def state(self) -> PercevalState:
        """Return the internal PercevalState wrapper."""
        return self._state

    @property
    def nqubit(self) -> int:
        """Return the number of qubits in the system."""
        return int(self.state.m / 2)

    def copy(self) -> PercevalBackend:
        """Return a copy of the PercevalBackend object.

        Returns
        -------
        PercevalBackend
            Copy of the PercevalBackend object

        """
        new_backend = PercevalBackend(self.source, self.state.copy())
        new_node_index = NodeIndex()
        new_node_index.__dict = self.node_index.__dict.copy()  # noqa: SLF001
        new_node_index.__list = self.node_index.__list.copy()  # noqa: SLF001
        new_backend.node_index = new_node_index
        return new_backend

    def add_nodes(self, nodes: Iterable[int], data=BasicStates.PLUS) -> None:  # type: ignore  # noqa: PGH003
        """Add nodes to the perceval system.

        Parameters
        ----------
        nodes : Iterable[int]
            List of nodes to add
        data : Data
            State to initialize the nodes with. Can be:
            - Single State: broadcast to all nodes
            - List of States: one per node (must match length)
            - Statevec: must be single qubit

        Raises
        ------
        ValueError
            If the input state is not a single qubit state, or if list length doesn't match nodes

        """
        nodes_list = list(nodes)

        # Handle list of states - add nodes one at a time
        if isinstance(data, list):
            if len(data) != len(nodes_list):
                msg = f"Length mismatch: {len(data)} states for {len(nodes_list)} nodes"
                raise ValueError(msg)
            # Add nodes one at a time with individual states
            for node, state in zip(nodes_list, data, strict=True):
                self.add_nodes([node], state)
            return

        # Single state for all nodes
        if isinstance(data, Statevec):
            psi = data.psi
            if data.nqubit != 1:
                msg = "input state must be a single qubit state"
                raise ValueError(msg)
        else:
            psi = data.get_statevector()

        for _ in nodes_list:
            self.state.add_qubit(psi)
        self.node_index.extend(nodes_list)

    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """Entangle two nodes using a heralded CZ gate.

        Parameters
        ----------
        edge : tuple[int, int]
            Nodes to entangle.

        Raises
        ------
        ValueError
            If nodes in the edge are not in the current node_index.

        """
        if edge[0] not in self.node_index or edge[1] not in self.node_index:
            msg = f"Nodes {edge} not in current node_index"
            raise ValueError(msg)

        # get optical modes corresponding to edge qubits
        index_0 = self.node_index.index(edge[0])
        index_1 = self.node_index.index(edge[1])
        self.state.entangle(index_0, index_1)

    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Literal[0, 1]:  # noqa: ARG002
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.
        measurement : Measurement
            Measurement to perform.
        rng : Generator, optional
            Random number generator to use for sampling. Defaults to None.

        Returns
        -------
        Literal[0, 1]
            Measurement outcome.

        Raises
        ------
        ValueError
            If the node is not in the current node_index.

        """
        if node not in self.node_index:
            msg = f"Node {node} not in current node_index"
            raise ValueError(msg)

        index = self.node_index.index(node)

        meas_circ = pcvl.Circuit(2 * self.nqubit)
        match measurement.plane:
            # YZ and XZ not properly tested, only used XY plane measurements
            case Plane.XY:
                # rotation around Z axis by -angle
                meas_circ.add(2 * index + 1, comp.PS(-measurement.angle))
                # transformation from X basis to Z basis
                meas_circ.add(2 * index, comp.BS.H())

            case Plane.YZ:
                # rotation around X axis by -angle
                meas_circ.add(2 * index, comp.BS.H())
                meas_circ.add(2 * index + 1, comp.PS(-measurement.angle))
                meas_circ.add(2 * index, comp.BS.H())
                # transformation from Y basis to Z basis
                meas_circ.add(2 * index + 1, comp.PS(-np.pi / 2))
                meas_circ.add(2 * index, comp.BS.H())

            case Plane.XZ:
                # rotation around Y axis by -angle
                meas_circ.add(2 * index + 1, comp.PS(-np.pi / 2))
                meas_circ.add(2 * index, comp.BS.H())
                meas_circ.add(2 * index + 1, comp.PS(-measurement.angle))
                meas_circ.add(2 * index, comp.BS.H())
                # transformation from X basis to Z basis
                meas_circ.add(2 * index + 1, comp.PS(np.pi / 2))
                meas_circ.add(2 * index, comp.BS.H())

        # applies operation on measured qubit before performing a Z basis measurement
        result = self.state.measure_qubit(index, meas_circ)
        self.node_index.remove(node)
        return result

    def correct_byproduct(self, cmd, measure_method) -> None:  # type: ignore[no-untyped-def]
        """Apply byproduct correction.

        Corrects for the X or Z byproduct operators by applying the X or Z gate
        depending on previous measurement outcomes.

        Parameters
        ----------
        cmd : Command
            Byproduct correction command.
        measure_method : MeasureMethod
            Measurement method providing results to check correction condition.

        """
        if np.mod(sum(measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
            index = self.node_index.index(cmd.node)
            correct_circ = pcvl.Circuit(2 * self.nqubit)

            if cmd.kind == CommandKind.X:
                correct_circ.add(2 * index, comp.PERM([1, 0]))
            elif cmd.kind == CommandKind.Z:
                correct_circ.add(2 * index + 1, comp.PS(np.pi))

            self.state.evolve(correct_circ)

    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate.

        Parameters
        ----------
        node : int
            Node label of the qubit to apply the gate to.
        clifford : Clifford
            Clifford gate to apply.

        Raises
        ------
        ValueError
            If the node is not in the current node_index.

        """
        if node not in self.node_index:
            msg = f"Node {node} not in current node_index"
            raise ValueError(msg)

        index = self.node_index.index(node)

        # use unitary defining the clifford to initialise the perceval circuit
        clifford_circ = pcvl.Circuit(2 * self.nqubit).add(2 * index, comp.Unitary(U=pcvl.Matrix(clifford.matrix)))

        self.state.evolve(clifford_circ)

    def sort_qubits(self, output_nodes: Iterable[int]) -> None:
        """Sort the qubit order in internal statevector.

        Parameters
        ----------
        output_nodes : Iterable[int]
            Desired order of output nodes.

        Raises
        ------
        ValueError
            If any of the output_nodes are not in the current node_index.

        """
        for node in output_nodes:
            if node not in self.node_index:
                msg = f"Output node {node} not in current nodes"
                raise ValueError(msg)

        if self.nqubit > 0:
            perm_circ = pcvl.Circuit(2 * self.nqubit)

            for i, ind in enumerate(output_nodes):
                if self.node_index.index(ind) != i:
                    move_from = self.node_index.index(ind)
                    self.node_index.swap(i, move_from)

                    low = min(i, move_from)
                    high = max(i, move_from)
                    perm_circ.add(
                        2 * low,
                        comp.PERM([2 * (high - low), 2 * (high - low) + 1, *list(range(2, 2 * (high - low))), 0, 1]),
                    )

            self.state.evolve(perm_circ)

    def finalize(self, output_nodes: Iterable[int]) -> None:
        """Finalize the simulation by sorting the output qubits.

        Parameters
        ----------
        output_nodes : Iterable[int]
            Desired order of output nodes.

        """
        self.sort_qubits(output_nodes)


def perceval_statevector_to_graphix_statevec(psvec: pcvl.StateVector) -> Statevec:
    """Convert a multi-qubit Perceval StateVector to a Graphix Statevec.

    Uses path encoding (2 modes per qubit).

    Parameters
    ----------
    psvec : pcvl.StateVector
        Perceval StateVector to convert.

    Returns
    -------
    Statevec
        Graphix Statevec.

    Raises
    ------
    ValueError
        If the number of modes is not even.

    """
    n_qubit = psvec.m // 2
    if psvec.m % 2 != 0:
        msg = f"Expected even number of modes for path encoding, got {psvec.m}"
        raise ValueError(msg)

    data = np.zeros(2**n_qubit, dtype=np.complex128)
    for basic_state, amplitude in psvec:
        # Extract photon modes for each qubit
        # Path encoding uses one photon per mode pair (2i, 2i+1)
        # The basis index is calculated as the sum of (bit_i * 2^(n-1-i))
        index = 0
        for i in range(n_qubit):
            # Check if photon is in mode 2i+1 (logical |1>)
            if basic_state[2 * i + 1] == 1:
                index += 2 ** (n_qubit - 1 - i)
        data[index] = amplitude
    return Statevec(data=data, nqubit=n_qubit)


def graphix_planar_state_to_perceval_statevec(planar_state: PlanarState) -> pcvl.StateVector:
    """Convert a Graphix PlanarState to a Perceval StateVector.

    Parameters
    ----------
    planar_state : PlanarState
        Graphix PlanarState to convert.

    Returns
    -------
    pcvl.StateVector
        Perceval StateVector (2-mode path encoding).

    """
    statevector = planar_state.get_statevector()
    alpha = statevector[0]
    beta = statevector[1]
    return alpha * pcvl.StateVector([1, 0]) + beta * pcvl.StateVector([0, 1])


def graphix_state_to_perceval_statevec(statevec: npt.NDArray) -> pcvl.StateVector:
    """Convert a Graphix statevector to a Perceval StateVector.

    Uses path encoding (2 modes per qubit).

    Parameters
    ----------
    statevec : npt.NDArray
        Graphix statevector to convert.

    Returns
    -------
    pcvl.StateVector
        Perceval StateVector (2n modes).

    Raises
    ------
    ValueError
        If the length of statevec is not a power of 2.

    """
    n_qubit = int(np.log2(len(statevec)))
    if 2**n_qubit != len(statevec):
        msg = f"Statevec length {len(statevec)} is not a power of 2"
        raise ValueError(msg)

    result = None
    for index, amplitude in enumerate(statevec):
        if np.abs(amplitude) < PRECISION:
            continue

        # Basis state binary representation
        # Graphix uses lexicographical order: index 1 with 2 qubits is |01>
        # Correspondingly, qubit 0 is index 0 and qubit 1 is index 1
        fock_state = [0] * (2 * n_qubit)
        for i in range(n_qubit):
            # Extract i-th bit from the left
            bit_val = (index >> (n_qubit - 1 - i)) & 1
            if bit_val == 0:
                fock_state[2 * i] = 1  # Logic 0 maps to mode 2i
            else:
                fock_state[2 * i + 1] = 1  # Logic 1 maps to mode 2i+1

        state_vec = pcvl.StateVector(fock_state)
        # Convert numpy amplitude to complex to avoid type incompatibility
        term = complex(amplitude) * state_vec
        if result is None:
            result = term
        else:
            result += term

    return result if result is not None else pcvl.StateVector()
