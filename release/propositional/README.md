# A finite semantic skeleton for propositional proofs

From the three classical Hilbert axiom schemata, the constructor grows checked
consequences by condensed detachment without seeing an external target. At seed
101, its first-proof archive contains 17,202 distinct derived statements. The
selection rule retains ten anchor schemata. Their shared certificate contains
27 distinct derived steps. The axiom nodes have identifiers 0, 1, and 2.

## How the constructor chooses anchors

During growth, the engine counts which generated proofs contain a candidate as
a proper intermediate conclusion and which inferences use it as an immediate
premise. It retains evidence from the accumulated archive. When it freezes a
bank, it also checks whether a candidate schema can supply an already-used
multistep premise through one consistent substitution. Repeated variables must
match, types must agree, and a schema with distinct-variable restrictions
receives credit only for directly recorded uses.

Each selection credits newly covered proofs and distinct consumers on the
frontier of formula size and reuse. Ties favour additional proof coverage,
then shorter proof height, then earlier archive order. Selection stops when no
candidate earns a vote. In the primary run the identity schema has zero direct
consumers by node ID. A premise it can supply occurs in 39 distinct inferences,
which gives it fifth place. The identity appears in every native bank at both
growth ceilings.

- [Growth engine and the transfer mode](../../hpc/article1_skeleton/native_cumulative_policy.py)
- [Coverage, premise matching, and the stopping rule](../../hpc/article1_skeleton/fragment_policy.py)
- [Common archive reader and terminal selector for native and SGCD traces](bank.py)

## Primary skeleton

The node identifiers index `banks.ours101.proof_dag` in the data file.
Implication associates to the right and variables are local to each row.

| Rank | Node | Schema |
| ---: | ---: | --- |
| 1 | 6 | (A → B → C → D) → A → (B → C) → B → D |
| 2 | 107 | A → B → C → A |
| 3 | 203 | A → B → B |
| 4 | 8 | ((A → B → C) → ((A → B) → A → C) → D) → (A → B → C) → D |
| 5 | 167 | A → A |
| 6 | 32 | A → B → C → D → C |
| 7 | 295 | (A → ¬B → ¬C) → A → C → B |
| 8 | 65 | (A → (B → A) → C) → A → C |
| 9 | 40 | (A → A → B) → A → B |
| 10 | 45 | A → B → (¬C → ¬D) → D → C |

[All banks, ordered parent pairs, and scores for each external proof](skeleton-data.json)

## Check the results

Run from the repository root with Python 3.11 or later. The checker uses the
Python standard library and the repository's independent propositional kernel.

```sh
python3 release/propositional/verify.py
python3 release/propositional/verify.py --reselect
```

The first command checks the trace digests and the derivations from the three
axioms. It replays the external proof graphs, checks their target types and
structural hashes, identifies the anchors used, and recomputes the proof
lengths. It checks the full banks, matched-length prefixes, and results with
the identity removed. The second command also rebuilds all 16 banks from the
included growth traces and checks their ordered selection records and shared
certificates against the frozen data.

To grow a **new** bank on a fixed attempt budget, install the dependencies from
the root README and run this example with a new output directory.

```sh
python3 release/propositional/grow.py /tmp/propositional-example --seed 101 --attempts 500
```

The example uses the tested constructor, the three axioms, and the same learned
gate weights. It writes a growth trace and its selected bank. If the budget
yields only axioms, it writes the trace alone. The original experiment used
CPU-time cutoffs. The included traces preserve the exact prefixes that each
timed run reached. The files `basis.json` and `gate.json` supply the growth
inputs. The gate file contains the frozen scoring weights and fallback weights.

## External comparison

Native and SGCD runs started cold with the same axioms, condensed detachment,
one pinned core, a 1 GiB memory limit, and a growth ceiling of 90 CPU-seconds.
A second series used 15 CPU-seconds. The terminal selector had a separate
90-CPU-second ceiling per archive and used the same rule on both generators.
Three deterministic SGCD repeats and five native seeds at each ceiling give
the 16 banks. SGCD finished growth in 32.26–37.38 CPU-seconds in the
90-second condition.

| External proofs | Original steps | Saved by native seed 101 | Saved by full SGCD | Native wins / ties / losses across five seeds |
| --- | ---: | ---: | ---: | ---: |
| Aviary, all 41 distinct declared types | 454 | 104 | 146 | 3 / 0 / 2 |
| Aviary, 30 previously unseen proof structures | 412 | 90 | 128 | 3 / 0 / 2 |
| Aviary, four previously unseen statements | 76 | 9 | 15 | 1 / 2 / 2 |
| Published SGCD, 475 new proofs of familiar statements | 10,898 | 1,180 | 1,308 | 1 / 1 / 3 |

The Aviary rows overlap. Each proof keeps its final inference. A shared
intermediate step counts once within that proof. For net savings subtract
the cost of the bank's shared certificate once per panel. For the 30-proof
Aviary panel the primary ten anchors save 63 net steps. SGCD's full 22-anchor
bank saves 100 net steps. Both banks shorten all 30 proofs. SGCD's first ten
anchors also shorten all 30 and save 115 steps before the certificate charge.
The table measures compression of supplied proofs. A proof-search speedup
would require a separate timed search.

The archived data identify the external source distributions by SHA-256.
The Aviary proofs come from typed combinators via bracket abstraction with
eta-contraction. The second corpus draws on seven published SGCD runs.
[Aviary 0.4.0](https://hackage-content.haskell.org/package/data-aviary-0.4.0) · [Wernhard, IJCAR 2026](https://doi.org/10.1007/978-3-032-32589-1_3) · [Wernhard's experiment archive](http://cs.christophwernhard.com/cdtools/exp-thgen/)
