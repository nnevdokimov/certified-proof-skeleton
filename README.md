# Certified Proof Skeleton

A goal-free constructor starts from three axioms of classical propositional
logic and selects proved schemata for reuse. Its primary finite skeleton
contains **10 anchors** whose certificates share **27 inference steps**.

## Reproduce the propositional skeleton

The release includes the constructor, frozen banks, growth traces for both
generators, and external proof graphs. To regenerate all 16 banks and check
the published proof scores, run this command from the repository root.

```sh
python3 release/propositional/verify.py --reselect
```

- [Algorithm, primary skeleton, experimental conditions, and a fresh growth example](release/propositional/README.md)
- [Frozen schemata, certificates, and proof-level results](release/propositional/skeleton-data.json)
- [Online and terminal anchor selection](hpc/article1_skeleton/fragment_policy.py)

## Earlier native continuation study

The previous study examines how active premises support subsequent growth on
a broader native basis. Its frozen protocol uses the historical compression
selector and a fixed structural rule repertoire.

- [Native workflow and reproduction guide](docs/ARTICLE1_REPRODUCIBILITY.md)
- [Audited continuation and network results](release/article1/README.md)
- [Archived manuscript and supporting appendices](release/article1/manuscript/README.md)

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

The historical comparison includes five arms: `core`, `fixed`, `random`,
`adaptive`, and `fixed_all`.

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

Replaying the earlier native continuation experiment requires additional inputs:
the full raw archive, trained gate, inherited library, and platform-specific
verifier require separate provision. Compact results do not replace those inputs.

Do not run `prepare`, `execute`, or `evaluate` over a completed frozen run.
Use a new directory and a new protocol for a new experiment. Installation and
the default self-contained tests do not launch a scientific run or submit
cluster jobs.

## Repository boundaries

The propositional release includes the frozen input traces for its replay.
Local research notes, caches, and submission-system materials stay outside Git.
This repository starts from a clean source snapshot on `main`. The original
working copy, earlier Git history, and frozen experimental records remain in the
local research archive. Historical presentations, research diaries, and obsolete
plans do not form part of this repository.

This release does not add a licence grant. Third-party inputs and verifier
sources retain their own licence and attribution requirements.
