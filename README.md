# graphix-perceval-backend
Simulation backend interface between Graphix and Quandela's Perceval package.

## Implementation Notes

- **State Representation**: Ideally, `PercevalBackend` should inherit from Graphix's `Backend` while using a state class that inherits from both `pcvl.StateVector` and Graphix's `State`. Currently, it manages `_perceval_state` internally.
- **Mixed States**: We currently choose not to deal with mixed states by sampling single input states from distributions. This means we cannot compute the fraction of successful runs and may run into post-selection issues. This is because Perceval does not currently handle measurements on `SVDistribution` (mixed states).
- **Density Matrix Limitations**: `SVDistributions` cannot be directly made into `DensityMatrix` objects for measurement because Perceval's `DensityMatrix` doesn't handle distinguishable photons yet.
- **Entanglement**: CZ gates are applied via the `Processor` class to ensure correct mode permutations, as the `Circuit` class requires contiguous modes.
- **Post-selection**: Post-selection in the `measure` function may fail because we don't simulate the full distribution. A future improvement could implement retries or full distribution simulation to recover success probabilities.

## Future Improvements / TODOs

- **State Representation**: Refactor to use a dedicated state class inheriting from `pcvl.StateVector`.
- **Measurement Selection**: Implement choice of PNR or threshold detectors (currently only threshold implemented).
- **State Generation**: Add alternative state generation strategies, such as using fusions or RUS gates.
- **Mixed State Support**: Implement `add_nodes` and measurements for mixed states to track success probabilities.
- **QPU Mode**: Add a mode that constructs the linear optical circuit (with feed-forward) instead of simulating, for execution on a QPU.
- **Sorting**: Thoroughly test `sort_qubits` (currently only checked on classical outputs).
- **Measurement Planes**: Properly test YZ and XZ plane measurements (currently only XY plane is extensively tested).
