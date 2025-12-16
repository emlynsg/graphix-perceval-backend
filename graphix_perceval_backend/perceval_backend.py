"""Backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.clifford import Clifford
    from graphix.measurements import Measurement
    from numpy.random import Generator


class PercevalBackend(Backend):
    """Backend interface between Graphix and Quandela's Perceval package for pattern simulation."""

    # possible upgrades:
    # - fix issue with perceval_state (see init)
    # - choice of PNR or threshold detectors (currently only threshold is implemented)
    # - other state generation strategies: with fusions, with RUS gates (would probably required standardised pattern)
    # - add option to track and return success probability (adapt add_nodes, entangle_nodes, measure to work with mixed states)  # noqa: E501
    # - add a mode which does not perform the simulation but only constructs the circuit (with the proper feed-forward), to run on QPU  # noqa: E501

    def __init__(self, source: pcvl.Source, perceval_state: pcvl.StateVector | None = None) -> None:  # noqa: D107
        self._source = source
        if perceval_state is None:
            self._perceval_state = pcvl.BasicState()
        else:
            self._perceval_state = perceval_state

        self.node_index = NodeIndex()
        self._sim = pcvl.simulators.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        self._sim.set_min_detected_photons_filter(0)
        self._sim.keep_heralds(False)  # noqa: FBT003

        # Ideally we want the state below to be the perceval state,
        # but this requires creating a new class that inherits from pcvl.StateVector and State
        # In that case, need to replace all calls to self._perceval_state by self.state
        # super().__init__(DensityMatrix(nqubit = 0), pr_calc = True, rng = None)  # noqa: ERA001

    def __call__(self) -> PercevalBackend:
        """Return a copy of the PercevalBackend object."""  # noqa: DOC201
        return self

    @property
    def source(self) -> pcvl.Source:  # noqa: D102
        return self._source

    # changing how the state works would remove the need to redefine this
    @property
    def state(self) -> pcvl.StateVector:  # noqa: D102
        return self._perceval_state

    @property
    def nqubit(self) -> int:  # noqa: D102
        return int(self.state.m / 2)

    def copy(self) -> PercevalBackend:
        return PercevalBackend(self._source, self._perceval_state)

    def add_nodes(self, nodes: Iterable[int], data=BasicStates.PLUS) -> None:  # type: ignore  # noqa: PGH003
        """Add nodes to the perceval system.

        TODO: implement add_nodes for mixed states

        Parameters
        ----------
            nodes: list of nodes to add
            data: state to initialize the nodes with

        Raises
        ------
            ValueError: if the input state is not a single qubit state

        """
        # path-encoded |0> state
        zero_mixed_state = self._source.generate_distribution(pcvl.BasicState([1, 0]))
        # Here we explicitely choose not to deal with mixed states by sampling a single input state from the distribution above.
        # This means we cannot compute the fraction of successful runs and may run into post-selection problems (see measure function).
        # This is done because perceval does not (for now) handle measurements on SVDistribution (mixed states).
        # Those cannot be translated to DensityMatrix objects (which can be measured) since DensityMatrix does not handle distinguishable photons (yet).
        # One solution would be to manually implement measurements on SVDistributions.
        init_qubit = zero_mixed_state.sample(1)[0]
        # recover amplitudes of input state
        if isinstance(data, Statevec):
            statevector = data.psi
            if data.nqubit != 1:
                msg = "input state must be a single qubit state"
                raise ValueError(msg)
        else:
            statevector = data.get_statevector()
        alpha = statevector[0]
        beta = statevector[1]
        if np.abs(beta) != 0:  # if beta = 0, the input is already |0>, no need to do anything
            # construct unitary matrix taking |0> to the state psi
            gamma = np.abs(beta)
            delta = -np.conjugate(alpha) * gamma / np.conjugate(beta)
            matrix = pcvl.Matrix(np.asarray([[alpha, gamma], [beta, delta]]))
            init_circ = pcvl.Circuit(2)
            init_circ.add(0, comp.Unitary(U=matrix))
            self._sim.set_circuit(init_circ)
            init_qubit = self._sim.evolve(init_qubit)
        self._perceval_state *= init_qubit
        self.node_index.extend(nodes)

    def entangle_nodes(self, edge: tuple[int, int]) -> None:  # noqa: D102
        # get optical modes corresponding to edge qubits
        index_0 = self.node_index.index(edge[0])
        index_1 = self.node_index.index(edge[1])
        ctrl = min(index_0, index_1)
        target = max(index_0, index_1)
        cz_input_modes = [2 * ctrl, 2 * ctrl + 1, 2 * target, 2 * target + 1]

        # We construct the circuit via the Processor class since this class applies
        # the correct permutation before and after to place CZ at correct modes
        # (while the Circuit class does not and returns an error if the modes are not contiguous)
        ent_proc = pcvl.Processor("SLOS", 2 * self.nqubit)
        ent_proc.add(cz_input_modes, catalog["heralded cz"].build_processor())
        ent_circ = ent_proc.linear_circuit()
        self._sim.set_circuit(ent_circ)

        # the first 2n modes store the state, the last modes are heralds (1 photon in 2 modes for each CZ gate)
        heralds = dict.fromkeys(list(range(2 * self.nqubit, ent_circ.m)), 1)
        self._sim.set_heralds(heralds)
        herald_state = self._source.generate_distribution(pcvl.BasicState([1, 1]))
        # Here we again explicitely choose not to deal with mixed states
        sampled_herald_state = herald_state.sample(1)[0]

        self._perceval_state = self._sim.evolve(self._perceval_state * sampled_herald_state)

        self._sim.clear_heralds()

    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Literal[0, 1]:
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.
        measurement : Measurement
            Measurement to perform.
        rng : Generator, optional
            Random number generator to use for sampling, by default None

        Returns
        -------
        int

        """
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
                meas_circ.add(2 * index + 1, comp.PS(np.pi / 2))
                # transformation from X basis to Z basis
                meas_circ.add(2 * index, comp.BS.H())

        # applies operation on measured qubit before performing a Z basis measurement
        self._sim.set_circuit(meas_circ)
        self._perceval_state = self._sim.evolve(self._perceval_state)

        # the measure function returns a dictionary where the keys are the possible measurement outcomes (as pcvl.BasicState s)
        # and the values are the results (the first is the probability of obtaining that outcome, the second is the remaining state, after collapse)
        # in order to sample from these outcomes, we construct a pcvl.BSDistribution
        all_possible_meas_outcomes = self._perceval_state.measure([2 * index, 2 * index + 1])
        outcome_dist = {}
        for outcome, res in all_possible_meas_outcomes.items():
            outcome_dist[outcome] = res[0]
        outcomes = pcvl.BSDistribution(outcome_dist)

        # we then post-select the distribution above on having a qubit encoding and sample from it
        # the post-selection may fail (because we don't simulate the full distribution, see comment in add_nodes)
        # need to decide how to catch that and what to do with it
        # one possibility would be to retry the full computation and count the number of retries
        # if we're simulating the full distribution instead, we can at this stage recover the success probability
        ps = pcvl.PostSelect("([0] > 0 & [1] == 0) | ([0] == 0 & [1] > 0)")
        ps_outcomes = pcvl.utils.postselect.post_select_distribution(outcomes, ps)[0]
        meas_result = ps_outcomes.sample(1)[0]
        result = meas_result[0] == 0

        # we then set the state to the reduced state that corresponds to the sampled outcome
        self._perceval_state = all_possible_meas_outcomes[meas_result][1]
        self.node_index.remove(node)
        return result

    def correct_byproduct(self, cmd, measure_method) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""
        if np.mod(sum(measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
            index = self.node_index.index(cmd.node)
            correct_circ = pcvl.Circuit(2 * self.nqubit)

            if cmd.kind == CommandKind.X:
                correct_circ.add(2 * index, comp.PERM([1, 0]))
            elif cmd.kind == CommandKind.Z:
                correct_circ.add(2 * index + 1, comp.PS(np.pi))

            self._sim.set_circuit(correct_circ)
            self._perceval_state = self._sim.evolve(self._perceval_state)

    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate, specified by vop index specified in graphix.clifford.CLIFFORD."""
        index = self.node_index.index(node)

        # use unitary defining the clifford to initialise the perceval circuit
        clifford_circ = pcvl.Circuit(2 * self.nqubit).add(2 * index, comp.Unitary(U=pcvl.Matrix(clifford.matrix)))

        self._sim.set_circuit(clifford_circ)
        self._perceval_state = self._sim.evolve(self._perceval_state)

    def sort_qubits(self, output_nodes: Iterable[int]) -> None:
        """Sort the qubit order in internal statevector."""
        # not tested, checked code only on classical outputs
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

            self._sim.set_circuit(perm_circ)
            self._perceval_state = self._sim.evolve(self._perceval_state)

    def finalize(self, output_nodes: Iterable[int]) -> None:
        """To be run at the end of pattern simulation."""
        self.sort_qubits(output_nodes)


def perceval_single_qubit_statevector_to_graphix_statevec(psvec: pcvl.StateVector) -> Statevec:
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


def graphix_planar_state_to_perceval_statevec(planar_state: PlanarState) -> pcvl.StateVector:
    """Convert a Graphix PlanarState to a Perceval StateVector.

    Parameters
    ----------
    planar_state : PlanarState
        Graphix PlanarState to convert.

    Returns
    -------
    pcvl.StateVector
        Perceval StateVector.

    """
    return pcvl.StateVector(planar_state.get_statevector())


def graphix_state_to_perceval_statevec(statevec: npt.NDArray) -> pcvl.StateVector:
    """Convert a Graphix Statevec to a Perceval StateVector.

    Parameters
    ----------
    statevec : Statevec
        Graphix Statevec to convert.

    Returns
    -------
    pcvl.StateVector
        Perceval StateVector.

    """
    n_qubit = len(statevec)
    fock_states = []
    amplitudes = []
    for index, amplitude in enumerate(statevec):
        fock_state = [0] * n_qubit
        fock_state[index] = 1
        amplitudes.append(amplitude)
        fock_states.append(fock_state)
    return sum(amplitudes[i] * pcvl.StateVector(fock_states) for i in range(len(amplitudes)))
