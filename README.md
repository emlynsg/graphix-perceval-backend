# graphix-perceval-backend
Simulation backend interface between Graphix and Perceval.

## Implementation Notes

- **State Representation**: Ideally, `PercevalBackend` should inherit from Graphix's `Backend` while using a state class that inherits from both `pcvl.StateVector` and Graphix's `State`. Currently, it manages `_perceval_state` internally.
- **Mixed States**: We currently choose not to deal with mixed states by sampling single input states from distributions. This means we cannot compute the fraction of successful runs and may run into post-selection issues. This is because Perceval does not currently handle measurements on `SVDistribution` (mixed states).
- **Density Matrix Limitations**: `SVDistributions` cannot be directly made into `DensityMatrix` objects for measurement because Perceval's `DensityMatrix` doesn't handle distinguishable photons yet.
- **Entanglement**: CZ gates are applied via the `Processor` class to ensure correct mode permutations, as the `Circuit` class requires contiguous modes.
- **Post-selection**: Post-selection in the `measure` function may fail because we don't simulate the full distribution. A future improvement could implement retries or full distribution simulation to recover success probabilities.

## Future Improvements / TODOs

- Update Veriphix with Graphix main and remove monkeypatch.
- Consider how to manage multiple photons in one mode in conversion functions.
- Implement choice of detectors.
- Add alternative state generation strategies.
- Implement `add_nodes` and measurements for mixed states to track success probabilities.
- Add a mode that constructs the linear optical circuit (with feed-forward) instead of simulating, for execution on a QPU.
- Thoroughly test `sort_qubits` (currently only checked on classical outputs).
- Properly test YZ and XZ plane measurements (currently only XY plane is tested).
