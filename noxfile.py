"""Run tests with nox.

Copyright (C) 2026, QAT team (ENS-PSL, Inria, CNRS).
"""

import nox
from nox import Session


def install_pytest(session: Session) -> None:
    """Install pytest when requirements-dev.txt is not installed."""
    session.install("pytest")


def run_pytest(session: Session) -> None:
    """Run pytest."""
    args = ["pytest"]
    session.run(*args)


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install(".")
    install_pytest(session)
    run_pytest(session)


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests_dev(session: Session) -> None:
    """Run the test suite with dev dependencies."""
    session.install(".[dev]")
    # We cannot run `pytest --doctest-modules` here, since some tests
    # involve optional dependencies, like pyzx.
    run_pytest(session)
