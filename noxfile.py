import nox

@nox.session(python=["3.12"])
def tests(session):
    session.install("pytest")
    session.install("-r", "requirements.txt")
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("pytest", "tests/test_perceval.py")

@nox.session(python=["3.12"])
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", ".")

@nox.session(python=["3.12"])
def mypy(session):
    session.install("mypy")
    session.install("-r", "requirements.txt")
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("mypy", "graphix_perceval_backend", "tests")
