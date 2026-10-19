# Contributing

Thank you for looking at ACT. This is a research codebase, and its main product is a
reproducible record, so a change is judged by whether its claims still recompute.

## Setup

```bash
uv sync --extra dev
```

## Checks

Run these before you open a change. They are fast and do not need a GPU.

```bash
PYTHONPATH=src python -m pytest -q                       # the test suite (parallel by default)
PYTHONPATH=src python -m compileall -q src/act tests     # syntax check
PYTHONPATH=src python scripts/verify_record.py           # every documented number vs its ledger
PYTHONPATH=src python scripts/check_record_freshness.py --strict-rank   # committed records regenerate
PYTHONPATH=src python scripts/build_notebooks.py --check # notebooks match their builder
```

`scripts/check_fresh_clone.py` runs the same checks against a clean clone of `HEAD`. Use
it after adding a test that reads run output, since untracked local output is visible in
your working tree but not to a reader of the repository.

## Conventions

- **A number is a claim only if it recomputes.** State where it came from and let
  `verify_record.py` re-derive it.
- **A committed record must regenerate byte-for-byte from its own documented command.**
  If the runbook and the record disagree, decide which one is wrong.
- **Pre-register before spending compute.** The rule and the acceptance criteria belong in
  the arm's `NOTES.md` before the run, and are edited afterwards only to record what
  happened.
- **Do not pool measurements that differ in precision or configuration.**
- **Heavy jobs are launched by scripts under `scripts/`.** Read a script's `--help` and
  docstring before running it; several research instruments have defaults that spend
  GPU hours.
- **Documents go under `docs/`.** Only `README.md` and `AGENTS.md` belong at the root,
  and every standing document must be named in the README's Documentation section.

## Reporting problems

Open an issue describing the command you ran, its full output, and the commit you ran it
at. For a claimed number that does not match its artifact, include the line of the
document and the ledger it reads from.

## Licence

This repository is All rights reserved (see `LICENSE`). Contributions are accepted on the
condition that you have the right to submit them and that the copyright holder may
incorporate them.
