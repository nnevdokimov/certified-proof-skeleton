# Certified Proof Skeleton

This repository contains the main native experiment for certified cumulative
proof growth. The method keeps a reusable library and a bounded search interface,
retains certificates for inactive support, and measures how the resulting
structure supports further growth.

The current native method combines cumulative construction, a learned half-gate
where its syntax supports that model, deferred replay, reversible premise pruning,
a positive-marginal-gain proof codec, and a fixed structural rule repertoire.
It distinguishes permission to search with a statement from the certificate
support needed to justify it. The registered comparison did not establish an
advantage for source-adaptive rule selection; that selector does not enter the
recommended configuration.

## Start here

- [Code, dependencies, tests, and reproduction boundaries](docs/ARTICLE1_REPRODUCIBILITY.md)
- [Compact checked results and provenance](release/article1/README.md)

## Code map

| Path | Role |
| --- | --- |
| `scripts/run_native_holdout.py` | Frozen prospective protocol and one-shot evaluation workflow |
| `scripts/audit_native_holdout.py` | Independent certificate and permission checks |
| `scripts/metamath_native_skeleton.py` | Shared library I/O, certificate export, and verification helpers |
| `src/metamath_native/` | Typed terms, explicit axiom frames, certificates, and native growth |
| `hpc/article1_skeleton/` | Cumulative constructor, deferred queue, marginal codec, structural rules, and their dependencies |
| `tests/` | Self-contained checks and separately marked integration tests |
| `release/article1/` | Compact checked results and provenance |

Only the main native workflow forms the supported experiment interface. Shared
modules remain as its dependencies, not as separate experiment launchers. All
five registered comparison arms remain: `core`, `fixed`, `random`, `adaptive`,
and `fixed_all`. This focused release excludes shell launchers, earlier study
drivers, graph-analysis tooling, manuscript files, and editorial tools.

## Install and test

Use Python 3.11 or later. The default branch is `main`.

```bash
git clone https://github.com/nnevdokimov/certified-proof-skeleton.git
cd certified-proof-skeleton
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The default test configuration selects only self-contained checks. Integration
tests require the pinned inputs listed in the reproduction guide and may generate
bounded proof states or call the independent Metamath verifier. Run them
explicitly only after supplying those inputs:

```bash
python -m pytest -q -m integration
```

A source-only checkout cannot reproduce the complete historical experiment:
the full raw archive, trained gate, inherited library, and platform-specific
verifier require separate provision. Compact results do not replace those inputs.

Do not run `prepare`, `execute`, or `evaluate` over a completed frozen run.
Use a new directory and a new protocol for a new experiment. Installation and
the default self-contained tests do not launch a scientific run or submit
cluster jobs.

## Repository boundaries

Generated outputs, raw trajectories, local agent worktrees, caches, private
research notes, manuscript files, and submission materials remain outside Git.
This repository starts from a clean source snapshot on `main`. The original
working copy, earlier Git history, and frozen experimental records remain in the
local research archive. Historical presentations, research diaries, and obsolete
plans do not form part of this repository.

This release does not add a licence grant. Third-party inputs and verifier
sources retain their own licence and attribution requirements.
