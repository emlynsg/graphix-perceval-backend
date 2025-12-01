"""Tests for backend interface between Graphix and Quandela's Perceval package for pattern simulation.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

import perceval as pcvl

from graphix_perceval_backend.perceval_backend import PercevalBackend

def test_perceval_backend():
    source = pcvl.Source(emission_probability = 1, 
                multiphoton_component = 0, 
                indistinguishability = 1)
    backend = PercevalBackend(source=source, perceval_state=None)
    assert backend.nqubit == 0
