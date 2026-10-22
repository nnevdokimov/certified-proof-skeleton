# Article 1: implementation and reproducibility

This focused repository publishes the main native experiment, its dependencies,
tests, and compact evidence in [`release/article1/`](../release/article1/).
It excludes shell launchers, earlier study drivers, graph-analysis tooling,
the manuscript, and editorial history. Compact evidence records completed
results but omits full trajectories, frozen inputs, and verifier executables.
A source checkout supports code inspection and self-contained tests, not complete
end-to-end reproduction.

## Current method

The cumulative constructor extends a certified library while controlling which
statements may participate in search. It combines the following mechanisms:

1. A typed Metamath-native kernel checks substitutions, distinct-variable
   restrictions, modus ponens, generalisation, and stored certificates. The basis
   extractor reads an explicit list of axioms and definitions from a pinned
   `set.mm`; it does not import human `$p` theorems as generation premises.
2. The established learned-half gate ranks candidate premise pairs on its
   supported propositional syntax. A bounded deferred queue returns skipped
   opportunities to later search. Native predicate/set-theoretic syntax uses
   declared intrinsic fallback features, not an assumed transfer of the
   propositional predictor.
3. Reversible retirement controls the active frontier. Inactive certificate
   support can justify a stored proof but does not automatically acquire rights
   to act as a premise or a subsumer. Explicit activity and queue limits remain
   computational controls; they do not set the final compression dictionary size.
4. Shared proof fragments compress certificates only when their exact marginal
   saving remains positive after the current dictionary and overlap costs.
   Structural synthesis compiles reusable implication rules from bounded
   modus-ponens programmes and checks their certificates.
5. The recommended configuration uses the fixed structural repertoire. The
   registered comparison also retains `core`, `random`, `adaptive`, and
   `fixed_all` as controls. Its result, `no_confirmed_adaptive_superiority`, does
   not establish equivalence or optimality and does not justify substituting an
   adaptive winner for the fixed configuration.

The quality criterion measures subsequent certified growth and new coverage of
an initially uncovered target remainder, not merely the number of formulas at
construction time. Target observations do not guide the frozen search workers.

## Code map

| Component | Location |
|---|---|
| Typed terms, native basis, certificates, indexing, and online growth | `src/metamath_native/` |
| Adapter that reuses the accumulated mechanisms | `hpc/article1_skeleton/native_cumulative_policy.py` |
| Learned gate, delayed opportunities, and shared policy code | `hpc/article1_skeleton/skeleton_gate.py`, `deferred_gate.py`, and `adaptive_policy.py` |
| Proof-fragment costs and marginal compression | `hpc/article1_skeleton/fragment_policy.py` and `compression_overlay.py` |
| Structural rules and repertoire controls | `hpc/article1_skeleton/native_rule_synthesis.py` and `native_holdout.py` |
| Neutral continuations and coverage observers | `hpc/article1_skeleton/native_quality_evaluation.py` |
| Shared library I/O, certificate export, and verifier helpers | `scripts/metamath_native_skeleton.py` |
| Frozen protocol, one-shot evaluation, and pipeline | `scripts/run_native_holdout.py` |
| Separate replay auditor | `scripts/audit_native_holdout.py` |
| Kernel, protocol, queue, codec, and continuation checks | `tests/` |

Only the main native workflow forms the supported experiment interface.
`metamath_native_skeleton.py` supplies shared helpers; its earlier `build`
subcommand alone does not reproduce the cumulative method. The remaining shared
modules, including the required `core`, `graph`, and `hilbert` code, support this
workflow rather than offer separate studies. All five registered arms remain
available: `core`, `fixed`, `random`, `adaptive`, and `fixed_all`.

## Environment and quick checks

The recorded confirmation environment used CPython 3.13.3 on macOS/arm64.
[`requirements.txt`](../requirements.txt) contains NumPy, SciPy, NetworkX, and
pytest. SciPy supports the native engine's existing numerical functions;
NetworkX supports shared graph dependencies. The independent Metamath verifier
requires separate provision. Dependency ranges do not reconstruct an exact frozen
environment; CPU-limited trajectories may differ across hardware and versions.

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The default pytest configuration excludes the `integration` marker. Its checks
use small synthetic fixtures and require neither `set.mm`, a trained model,
external trajectories, nor a Metamath executable. They cover typed operations,
queue permissions, exact marginal savings, parsers, corpus exclusions, seals,
the blind-opening guard, timing credit, and paired inference. They do not submit
jobs, run long search experiments, or certify the published trajectories.

Data-dependent integration tests require the historical inputs below. They
exercise native certificates, imported rules, accumulated growth, and neutral
continuations; some perform bounded generation or invoke the independent verifier.
Select them explicitly:

```sh
.venv/bin/python -m pytest -q -m integration
```

These tests do not repeat the registered confirmation or its full replay audit.
The release validation record states which checks the maintainer actually ran;
instructions alone do not establish a passing result.

## Inputs absent from a source-only checkout

Paths below describe the existing drivers' defaults, relative to the repository
root. The cleanup preserves the original local files but does not add the full
`output/` tree to Git. Check the compact release inventory for any supplied copy;
do not silently replace a missing historical input with a newer file.

| Input | Historical relative path | SHA-256 |
|---|---|---|
| Pinned native-basis source | `output/article1_hpc_runs/skeleton-utility-final-20261006-v1/bundle/inputs/set.mm` | `90de6bac87d549023fc83fb698a1baacfed3f99b71438722171674f16777c881` |
| Certified starting library | `output/metamath_native_study_20261020_v2/working_core_deep/library.json.gz` | `9e08079f0bc489e2bb439dbf005e489ef179dd38cc450fc4efaa184ba57e808c` |
| Frozen propositional gate | `output/article1_skeleton_gate_pilot_20261016_v1/training/prop_200.model.json` | `97d91ac29f25ab6d9821c0393ddb9f0e252d3562c1e1a673db0b17e0359c3f22` |
| Historical independent verifier binary | `output/metamath_native_skeleton_20261020_v1/metamath` | `451562cfaa76c4b7a593473f2f52fe16313b328303b1dbc75934b00e4fe50d54` |
| POI archive | `output/poi_holdout_inputs_20261021_v1/poi_tptp.tgz` | `24f0b10dd85864dca014c76da470429b40ce967cfd3d7efcab1f7855e2798c13` |

The POI provenance record names Christoph Wernhard's
`http://cs.christophwernhard.com/cdtools/exp-thgen/downloads/poi_tptp.tgz`.
The recorded archive hash, not the continuing availability of that URL, identifies
the study input. The historical verifier targets macOS/arm64; another platform
requires a compatible independent Metamath executable and a separate provenance
record. A different binary must not replace the executable inside a sealed run.

Further indispensable material includes:

- Development profiles under
  `output/metamath_lemma_transactions_quality_20261020_v1/`, which determine
  the fixed extended repertoire. A directory name does not serve as a checksum;
  retain its original per-file manifests.
- The complete run
  `output/metamath_holdout_confirmation_20261021_v3/`: its frozen `code/`,
  `inputs/`, `evaluation/`, all 16 construction receipts, trajectory states,
  and certificate/event records. Summary JSON alone cannot replay these records.
- To reproduce the separately reported network diagnostics: inputs under
  `output/metamath_native_study_20261020_v2/` and the original graph-analysis
  packages, including analysis code, the degree-tail executable, and frozen
  graph arrays. Their original `PROTOCOL.json` files link graph and executable
  hashes. This focused code release omits that analysis tooling; compact network
  profiles do not replace the arrays or null-model samples.

The final native basis differs from the earlier short propositional diagnostic.
Do not substitute that diagnostic's primitive prefix or an older `set.mm` solely
because both files share a Metamath origin.

## Recorded confirmation and scope of replay

The completed run uses 16 source seeds, `2026102201` through `2026102216`, five
methods, two continuators, and 963 held-out targets. The 160 continuations form
16 statistical source blocks, not 160 independent theories. Its immutable seal
and protocol identify the reported experiment:

| Record inside `output/metamath_holdout_confirmation_20261021_v3/` | SHA-256 |
|---|---|
| `SEAL.json` | `eb73763e8a0730126498918edcd9bd58876497c1f150e01b3ddfda2288c26080` |
| `PROTOCOL.json` | `9ea3337b1604b3549c3e84b56d94e9e5bd26eb0779d603013cea528134c3e01d` |
| `EVALUATION_RECEIPT.json` | `e357a65de0f1cfe58b31e7de94c79d375946b6b47c2ca1f2a92963edda568d18` |
| `PIPELINE_COMPLETE.json` | `97fff406c5e1e9ab2afb095756b91c39b675fc4b8b080ee01183c8792e471540` |
| `independent_review/INDEPENDENT_REVIEW.json` | `29acbee7b2e5a7fec480985d8d8011123ee0da4e631656f858ed9bfe793f0931` |

The frozen pipeline checks construction receipts before opening targets once.
Its independent review replays 4,174,478 admission events and 2,846 witnesses
for rederived hidden support. These counts aggregate events, not distinct
mathematical theorems. The recorded comparison does not confirm adaptive
superiority; retain the complete registered comparisons alongside the working
fixed configuration.

Current source and the frozen snapshot serve different purposes. In particular,
`run_native_holdout.py::snapshot_code` copies Python files from selected source
directories. The original snapshot also contained utilities that this focused
release excludes. That snapshot and its original seal remain unchanged locally.
A fresh snapshot of this cleaned checkout will have different hashes. It cannot
recreate the original seal, and it must not masquerade as that registered run.

Do not rerun `execute` or `evaluate` in the completed directory. A future
experiment needs a new output directory and protocol. A replay audit of archived
results requires the complete matching frozen package and a separate output
directory; the compact Git release alone cannot provide that audit.

## Network interpretation

Network records distinguish the Metamath-compatible fragment, the native
certificate-supported skeleton, the complete finite H3 first-proof projection,
and the H3 all-inference projection. The last two share a finite theorem set but
different arcs. None denotes the infinite unrestricted closure.

The final-method representative uses `fixed/initial` for source `2026102201`:
675 derived anchors, 29 roots, and 137 additional support nodes, giving 841
vertices and 1,573 directed arcs. The registered adaptive illustration and the
profiles across sources remain separate from this representative. H3 depth and
human Metamath proof-step granularity do not match native skeleton depth.
Read clustering, paths, modularity, degree tails, and controls with their recorded
projection conventions; a small-world or power-law label does not by itself
establish computational utility.

Consult [`release/article1/README.md`](../release/article1/README.md) for the
published compact files and their checksums. Full raw-data archival and exact
environment reconstruction remain distinct from this source-and-results release.
