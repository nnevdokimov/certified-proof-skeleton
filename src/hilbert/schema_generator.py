"""
Schema-level Hilbert proof graph generator.

This generator can run in:
  - propositional / intuitionistic / modal_k schema modes
  - zfc schema mode (first-order formulas + ZFC axioms + Q1/Q2 + Gen)

In schema mode, PropVars are metavariables and MP uses unification.

Supports checkpointing: save state after each depth, resume later.
"""

from __future__ import annotations

import gzip
import gc
import math
import multiprocessing as _mp
import os
import logging
import pickle
import queue
import re
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.axioms import ZFCAxioms
from core.formula import (
    And,
    BigAnd,
    BigOr,
    Box,
    Equality,
    Exists,
    ExistsPred,
    ForAll,
    ForAllPred,
    Formula,
    Iff,
    Implies,
    Membership,
    Not,
    Or,
    PredApp,
    PredicateVar,
    PropVar,
    Term,
    make_variable,
)
from core.primitives import to_primitive_form
from core.schema import (
    canonicalize_formula,
    canonicalize_with_hash,
    collect_vars,
    collect_vars_ordered,
    detect_implicit_dvar_constraints,
    match as schema_match,
    normalize_schema,
    schema_hash,
    schema_hash_normalized,
    schema_mp,
)
from graph.proof_graph import ProofGraph, ProofNode
from graph.subsumption_index import SubsumptionIndex
from hilbert.side_conditions import is_free_for, is_not_free_in

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hilbert.utility import AdaptiveBeam, UtilityRanker

logger = logging.getLogger(__name__)

PROVENANCE_MODES = {"full", "first"}
CHECKPOINT_MODES = {"sync", "async", "fork_sync"}


class CheckpointChildError(RuntimeError):
    """A forked checkpoint writer failed or was terminated by a signal."""


def _checkpoint_mode_from_environment() -> Optional[str]:
    """Return an explicitly requested checkpoint mode, if any.

    ``ZFC_CHECKPOINT_MODE`` is the canonical switch.  The two boolean names
    remain supported so existing async launchers keep working and a wrapper can
    opt into fork-synchronous writes without changing generator construction.
    Conflicting switches are rejected rather than silently choosing a mode.
    """
    requested: List[Tuple[str, str]] = []
    explicit = os.environ.get("ZFC_CHECKPOINT_MODE", "").strip().lower()
    if explicit:
        requested.append(("ZFC_CHECKPOINT_MODE", explicit))
    if os.environ.get("ZFC_FORK_SYNC_CHECKPOINT", "").strip() == "1":
        requested.append(("ZFC_FORK_SYNC_CHECKPOINT", "fork_sync"))
    if os.environ.get("ZFC_ASYNC_CHECKPOINT", "").strip() == "1":
        requested.append(("ZFC_ASYNC_CHECKPOINT", "async"))
    modes = {mode for _name, mode in requested}
    unknown = modes - CHECKPOINT_MODES
    if unknown:
        raise ValueError(
            f"unknown checkpoint mode(s) {sorted(unknown)}; "
            f"expected one of {sorted(CHECKPOINT_MODES)}"
        )
    if len(modes) > 1:
        rendered = ", ".join(f"{name}={mode}" for name, mode in requested)
        raise ValueError(f"conflicting checkpoint mode switches: {rendered}")
    return next(iter(modes), None)

# Subsumption modes for the schema → theorem-space quotient. See
# docs/SUBSUMPTION_CORRECTNESS.md for the soundness argument.
SUBSUMPTION_MODES = {"off", "forward", "full"}


def provenance_metadata(mode: str) -> Dict[str, str]:
    """Metadata describing online proof provenance completeness."""
    if mode == "full":
        return {
            "provenance_mode": "full",
            "rediscovery_status": "online",
            "projection_edges_semantics": "full_online_rediscovery",
        }
    if mode == "first":
        return {
            "provenance_mode": "first",
            "rediscovery_status": "not_reconstructed",
            "projection_edges_semantics": "first_derivation_lower_bound",
        }
    raise ValueError(
        f"Unknown provenance_mode={mode!r}; expected one of {sorted(PROVENANCE_MODES)}"
    )


@dataclass(slots=True)
class SchemaEntry:
    """A schema in the graph with metadata."""

    id: int
    formula: Formula
    schema_hash: str
    depth: int
    rule: str
    premise_ids: List[int]


@dataclass(slots=True)
class MPCandidate:
    """Cached MP metadata for an implication entry."""

    entry: SchemaEntry
    major_vars: Set[str]
    antecedent_tag: int
    antecedent_is_wildcard: bool
    dvar_constraints: Tuple[Tuple[str, str], ...] = ()
    # Memoized "detachment" for a vacuous-antecedent wildcard major
    # ``P → ψ`` with ``P ∉ ψ``: MP with ANY minor yields the SAME canonical ψ
    # (verified: σ={P↦minor}, σ(ψ)=ψ). Caching ``(canon(ψ), hash)`` lets the MP
    # pass skip the per-minor schema_mp + canonicalize — 97–99% of d≥2 wildcard
    # work — while emitting the identical result stream (same graph).
    detached_result: Optional[Tuple["Formula", str]] = None


@dataclass(frozen=True, slots=True)
class SubsumedRecord:
    """Derivation evidence for a schema that was dropped by subsumption.

    Stored on the surviving subsumer's id. Together with the subsumer's
    canonical formula, ``sigma`` reconstructs the original (dropped) ψ:
    ``ψ = canonical(apply_subst(dict(sigma), subsumer.formula))``.

    See docs/SUBSUMPTION_CORRECTNESS.md Lemma C — sigma is the only datum
    not derivable from the subsumer alone.
    """

    parents: Tuple[int, ...]
    rule: str
    sigma: Tuple[Tuple[str, "Formula"], ...]
    depth: int


_PROPVAR_TOP_TAG = id(PropVar)


# ── Module-level workers for parallel MP pass ────────────────────────────
# Set per-worker via Pool initializer. Workers read these globals to avoid
# re-pickling large candidate lists for every chunk. Fork start method
# (default on Linux/macOS) makes initializer-passed data essentially free
# via copy-on-write.
_WORKER_ALL_CANDIDATES: Optional[List["MPCandidate"]] = None
_WORKER_WILDCARD_CANDIDATES: Optional[List["MPCandidate"]] = None
_WORKER_CANDIDATES_BY_TAG: Optional[Dict[int, List["MPCandidate"]]] = None
_WORKER_USE_PATTERN_UNIFY: bool = False  # Phase A I4: pattern-unification flag


def _mp_worker_init(all_candidates, wildcard_candidates, candidates_by_tag,
                    use_pattern_unify=False):
    global _WORKER_ALL_CANDIDATES, _WORKER_WILDCARD_CANDIDATES, _WORKER_CANDIDATES_BY_TAG
    global _WORKER_USE_PATTERN_UNIFY
    _WORKER_ALL_CANDIDATES = all_candidates
    _WORKER_WILDCARD_CANDIDATES = wildcard_candidates
    _WORKER_CANDIDATES_BY_TAG = candidates_by_tag
    _WORKER_USE_PATTERN_UNIFY = use_pattern_unify


def _mp_worker_result_batches(
    minor_chunk,
    result_batch_size: int,
) -> Iterator[Tuple[List[Tuple[int, int, Formula, str]], int]]:
    """Yield ``(unique_results_batch, raw_success_count_in_batch)``.

    Worker pre-dedupes locally: results whose schema_hash was already seen
    in this worker session are dropped from the unique-results batch. The
    raw_success_count tallies *all* successful MP applications since the
    last yield (including duplicates), so the master can keep the
    ``mp_successes`` statistic identical to a no-pre-dedup run.

    Each result item is ``(minor_id, major_id, canonical_formula, schema_hash)``.
    Master still does authoritative dedup against its own ``_entries``.
    """
    results: List[Tuple[int, int, Formula, str]] = []
    raw_success_count = 0
    # Worker pre-dedup is keyed by (conclusion_hash, sorted_premise_pair).
    # We MUST NOT collapse different premise sets that yield the same
    # conclusion — those are distinct (rule, premises, conclusion) hyperedges
    # the master needs to record for full provenance. Only collapse exact
    # syntactic duplicates (same minor & major, same canonical result).
    seen_local: Set[Tuple[str, int, int]] = set()
    minor_tag_propvar = _PROPVAR_TOP_TAG
    all_cands = _WORKER_ALL_CANDIDATES
    wildcard_cands = _WORKER_WILDCARD_CANDIDATES
    by_tag = _WORKER_CANDIDATES_BY_TAG

    if all_cands is None or wildcard_cands is None or by_tag is None:
        raise RuntimeError("MP worker was not initialized")

    for minor_id, minor_formula in minor_chunk:
        minor_tag = id(type(minor_formula))
        if minor_tag == minor_tag_propvar:
            buckets = (all_cands,)
        else:
            tagged = by_tag.get(minor_tag)
            if tagged:
                buckets = (wildcard_cands, tagged)
            else:
                buckets = (wildcard_cands,)

        for bucket in buckets:
            for major in bucket:
                det = major.detached_result
                if det is not None:
                    # Vacuous-antecedent wildcard major: MP always succeeds and
                    # yields the precomputed canonical ψ for any minor. Skip the
                    # schema_mp + canonicalize entirely.
                    canonical, h = det
                else:
                    result = schema_mp(
                        minor_formula,
                        major.entry.formula,
                        major_vars=major.major_vars,
                        antecedent_tag=major.antecedent_tag,
                        antecedent_is_wildcard=major.antecedent_is_wildcard,
                        inputs_canonical=True,
                        dvar_constraints=list(major.dvar_constraints),
                        canonicalize_output=False,
                        use_pattern_unify=_WORKER_USE_PATTERN_UNIFY,
                    )
                    if result is None:
                        continue
                    # canonicalize_with_hash does both canonicalization and hash
                    # in one AST traversal — schema_mp already skipped its own.
                    canonical, h = canonicalize_with_hash(result)
                raw_success_count += 1
                # Pre-dedup only exact (hash, minor_id, major_id) triples —
                # different premise pairs yielding the same hash are distinct
                # hyperedges and must reach master.
                dedup_key = (h, minor_id, major.entry.id)
                if dedup_key in seen_local:
                    continue
                seen_local.add(dedup_key)
                results.append((minor_id, major.entry.id, canonical, h))
                if len(results) >= result_batch_size:
                    yield results, raw_success_count
                    results = []
                    raw_success_count = 0

    if results or raw_success_count:
        yield results, raw_success_count


def _reset_child_signal_handlers() -> None:
    """Restore default SIGTERM/SIGUSR1 in forked children.

    run_experiment.py installs a graceful-shutdown handler on the MAIN process.
    Forked pool workers / MP processes / the async-checkpoint child INHERIT it,
    so on normal teardown (pool close) the child receives SIGTERM and runs the
    handler — spuriously printing "[signal] SIGTERM", saving a checkpoint from
    its copy-on-write generator snapshot (racing the master's save), and raising
    SystemExit inside atexit. Resetting to SIG_DFL makes children die quietly;
    only the main process handles preemption. Safe to call in any process.
    """
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except (ValueError, OSError, AttributeError):
        pass  # not in main thread of child / unsupported — best effort
    try:
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    except (ValueError, OSError, AttributeError):
        pass


def _mp_worker_stream_chunks(
    task_queue,
    result_queue,
    all_candidates,
    wildcard_candidates,
    candidates_by_tag,
    result_batch_size: int,
    use_pattern_unify: bool = False,
) -> None:
    """Worker target: stream bounded MP result batches back to the master."""
    _reset_child_signal_handlers()
    _mp_worker_init(all_candidates, wildcard_candidates, candidates_by_tag,
                    use_pattern_unify)
    try:
        while True:
            item = task_queue.get()
            if item is None:
                break

            chunk_index, minor_chunk = item
            for batch, raw_count in _mp_worker_result_batches(minor_chunk, result_batch_size):
                result_queue.put(("results", chunk_index, (batch, raw_count)))
            result_queue.put(("chunk_done", chunk_index, None))
    except BaseException as exc:  # pragma: no cover - exercised via master failure path
        result_queue.put(("error", None, (repr(exc), traceback.format_exc())))
    finally:
        result_queue.put(("worker_done", None, None))


def _q12_worker_wrapper(args):
    """Pool.imap_unordered entry point: unpack (chunk, terms, exists_rules)."""
    chunk, terms, exists_rules = args
    return _q12_worker_process(chunk, terms, exists_rules)


def _q12_worker_process(
    chunk: List[Tuple[int, Formula]],
    terms: List[Term],
    exists_rules: bool = False,
) -> List[Tuple[Formula, int, str]]:
    """Generate Q1/Q2 (and, for intuit/minimal, ∃-intro/∃-elim) instances for a
    chunk of (entry_id, formula) pairs.

    Returns a list of (implication_formula, premise_id, rule_name) triples.
    Master applies _add_schema sequentially to enforce single-writer dedup.
    Mirrors the serial path in ``_apply_quantifier_rules``.
    """
    from core.formula import ForAll, Exists
    results: List[Tuple[Formula, int, str]] = []
    for entry_id, formula in chunk:
        if type(formula) is ForAll:
            # Q1-inst: ∀x.φ(x) → φ(t)
            for term in terms:
                if not is_free_for(term, formula.variable, formula.body):
                    continue
                substituted = normalize_schema(
                    formula.body.substitute(formula.variable, term)
                )
                q1_impl = normalize_schema(Implies(formula, substituted))
                results.append((q1_impl, entry_id, "Q1-inst"))

            # Q2-inst: ∀x.(φ→ψ) → (φ → ∀x.ψ)  (when x ∉ FV(φ))
            if type(formula.body) is Implies:
                alpha = formula.body.antecedent
                beta = formula.body.consequent
                var = formula.variable
                if is_not_free_in(var, alpha) and not _contains_propvar(alpha):
                    consequent = normalize_schema(Implies(alpha, ForAll(var, beta)))
                    q2_impl = normalize_schema(Implies(formula, consequent))
                    results.append((q2_impl, entry_id, "Q2-inst"))

                # ∃-elim: ∀x.(φ→ψ) → (∃x.φ → ψ)  (when x ∉ FV(ψ))
                if exists_rules and is_not_free_in(var, beta) \
                        and not _contains_propvar(beta):
                    consequent = normalize_schema(Implies(Exists(var, alpha), beta))
                    ee_impl = normalize_schema(Implies(formula, consequent))
                    results.append((ee_impl, entry_id, "Exists-Elim"))

        # ∃-intro: φ[t/x] → ∃x.φ
        if exists_rules and type(formula) is Exists:
            for term in terms:
                if not is_free_for(term, formula.variable, formula.body):
                    continue
                substituted = normalize_schema(
                    formula.body.substitute(formula.variable, term)
                )
                ei_impl = normalize_schema(Implies(substituted, formula))
                results.append((ei_impl, entry_id, "Exists-Intro"))
    return results


def _make_axiom_schemas(logic: str) -> List[Tuple[Formula, str]]:
    """Build base Hilbert schema formulas for a given logic or propositional_base."""
    x0 = PropVar("x0")
    x1 = PropVar("x1")
    x2 = PropVar("x2")

    schemas: List[Tuple[Formula, str]] = []

    # H1-H4: shared core for all H-variants
    def _add_h1_h4() -> None:
        # TEST (ZFC_DROP_WEAKENING): drop H1 = K = weakening (the paradigm 'paradox of
        # material implication', SEP relevance logic). Leaves B,C,W (H2,H3,H4) = the
        # relevant implicational fragment. Tests whether removing weakening intrinsically
        # cuts the trivial-tautology bloat while keeping relevant theorems.
        if not os.environ.get("ZFC_DROP_WEAKENING"):
            schemas.append((Implies(x0, Implies(x1, x0)), "H1"))
        schemas.append((
            Implies(
                Implies(x0, Implies(x0, x1)),
                Implies(x0, x1),
            ),
            "H2",
        ))
        schemas.append((
            Implies(
                Implies(x0, Implies(x1, x2)),
                Implies(x1, Implies(x0, x2)),
            ),
            "H3",
        ))
        schemas.append((
            Implies(
                Implies(x1, x2),
                Implies(Implies(x0, x1), Implies(x0, x2)),
            ),
            "H4",
        ))

    # Minimal logic (H1-H4): no ex falso, no tertium non datur
    if logic == "h16_minimal":
        _add_h1_h4()
        return schemas

    # Intuitionistic logic (H1-H5): no tertium non datur
    if logic == "h16_intuit":
        _add_h1_h4()
        schemas.append((Implies(Not(x0), Implies(x0, x1)), "H5"))
        return schemas

    # Classical implication-negation base from H1..H6.
    # Used both for pure propositional mode and first-order (zfc/h16) mode.
    if logic in ("propositional", "zfc", "h16"):
        _add_h1_h4()
        schemas.append((Implies(Not(x0), Implies(x0, x1)), "H5"))
        schemas.append((
            Implies(
                Implies(x0, x1),
                Implies(Implies(Not(x0), x1), x1),
            ),
            "H6",
        ))
        if logic in ("propositional", "h16"):
            return schemas

    # Metamath set.mm propositional basis: ax-1, ax-2, ax-3.
    # ``metamath_prop`` is the pure-propositional fragment (no ax-4..13, no df-*).
    if logic in ("metamath", "metamath_prop"):
        # ax-1 (K-combinator): φ → (ψ → φ)
        schemas.append((Implies(x0, Implies(x1, x0)), "ax-1"))
        # ax-2 (S/Frege): (φ → (ψ → χ)) → ((φ → ψ) → (φ → χ))
        schemas.append((
            Implies(
                Implies(x0, Implies(x1, x2)),
                Implies(Implies(x0, x1), Implies(x0, x2)),
            ),
            "ax-2",
        ))
        # ax-3 (contraposition): (¬φ → ¬ψ) → (ψ → φ)
        schemas.append((
            Implies(
                Implies(Not(x0), Not(x1)),
                Implies(x1, x0),
            ),
            "ax-3",
        ))

    # Classical families outside the H1-H6 basis.
    _modal_bases = ("modal_k", "modal_t", "modal_s4", "modal_s5", "modal_gl")
    if logic in _modal_bases:
        schemas.append((Implies(x0, Implies(x1, x0)), "A1"))
        schemas.append(
            (
                Implies(
                    Implies(x0, Implies(x1, x2)),
                    Implies(Implies(x0, x1), Implies(x0, x2)),
                ),
                "A2",
            )
        )
        schemas.append(
            (
                Implies(
                    Implies(Not(x0), Not(x1)),
                    Implies(x1, x0),
                ),
                "A3",
            )
        )

    # Modal K base axiom (box distribution) — shared by K, T, S4, S5, GL.
    if logic in _modal_bases:
        schemas.append(
            (
                Implies(
                    Box(Implies(x0, x1)),
                    Implies(Box(x0), Box(x1)),
                ),
                "K",
            )
        )

    # Modal T: reflexivity (□φ → φ). Logic of knowledge / factive necessity.
    # Implied by S4 and S5; NOT compatible with GL (provability is not factive).
    if logic in ("modal_t", "modal_s4", "modal_s5"):
        schemas.append((Implies(Box(x0), x0), "T"))

    # Modal S4: positive introspection (□φ → □□φ). Logic of provability
    # without reflection. Independent in S4. C18: NOT seeded in S5 — there it is
    # derivable from K+T+5 (S5 = KT5; Chellas, Modal Logic §1.2), and seeding a
    # redundant axiom would distort the network signature.
    if logic == "modal_s4":
        schemas.append((Implies(Box(x0), Box(Box(x0))), "4"))

    # Modal S5: negative introspection (¬□φ → □¬□φ). Equivalent to
    # logical necessity / universal accessibility relation.
    if logic == "modal_s5":
        schemas.append(
            (
                Implies(Not(Box(x0)), Box(Not(Box(x0)))),
                "5",
            )
        )

    # Modal GL (Gödel-Löb): □(□φ → φ) → □φ. Provability logic in PA.
    # GL does NOT include T — provable does not imply true (incompleteness).
    if logic == "modal_gl":
        schemas.append(
            (
                Implies(
                    Box(Implies(Box(x0), x0)),
                    Box(x0),
                ),
                "GL",
            )
        )

    return schemas


def _make_metamath_predicate_axioms() -> List[Tuple[Formula, str]]:
    """Build Metamath set.mm predicate calculus axioms ax-4 through ax-13.

    These are schema-level axioms using PropVar metavariables and
    first-order variable names for quantified variables.
    """
    from core.formula import make_variable

    x0 = PropVar("x0")
    x1 = PropVar("x1")

    # Variable terms for predicate axioms
    vx = make_variable("x")
    vy = make_variable("y")
    vz = make_variable("z")

    schemas: List[Tuple[Formula, str]] = []

    # ax-4: ∀x(φ → ψ) → (∀xφ → ∀xψ)
    schemas.append((
        Implies(
            ForAll("x", Implies(x0, x1)),
            Implies(ForAll("x", x0), ForAll("x", x1)),
        ),
        "ax-4",
    ))

    # ax-5: φ → ∀xφ  (where x is not free in φ)
    # In set.mm this has a $d (distinct variable) constraint.
    # As a schema, we add it with the understanding that x0 has no free x.
    schemas.append((
        Implies(x0, ForAll("x", x0)),
        "ax-5",
    ))

    # ax-6: ¬∀x ¬(x = y)
    schemas.append((
        Not(ForAll("x", Not(Equality(vx, vy)))),
        "ax-6",
    ))

    # ax-7: x = y → (x = z → y = z)
    schemas.append((
        Implies(Equality(vx, vy), Implies(Equality(vx, vz), Equality(vy, vz))),
        "ax-7",
    ))

    # ax-8: x = y → (x ∈ z → y ∈ z)
    schemas.append((
        Implies(Equality(vx, vy), Implies(Membership(vx, vz), Membership(vy, vz))),
        "ax-8",
    ))

    # ax-9: x = y → (z ∈ x → z ∈ y)
    schemas.append((
        Implies(Equality(vx, vy), Implies(Membership(vz, vx), Membership(vz, vy))),
        "ax-9",
    ))

    # ax-10: ¬∀xφ → ∀x¬∀xφ
    schemas.append((
        Implies(
            Not(ForAll("x", x0)),
            ForAll("x", Not(ForAll("x", x0))),
        ),
        "ax-10",
    ))

    # ax-11: ∀x∀yφ → ∀y∀xφ
    schemas.append((
        Implies(
            ForAll("x", ForAll("y", x0)),
            ForAll("y", ForAll("x", x0)),
        ),
        "ax-11",
    ))

    # ax-12: x = y → (∀yφ → ∀x(x = y → φ))
    schemas.append((
        Implies(
            Equality(vx, vy),
            Implies(
                ForAll("y", x0),
                ForAll("x", Implies(Equality(vx, vy), x0)),
            ),
        ),
        "ax-12",
    ))

    # ax-13: ¬(x = y) → (y = z → ∀x(y = z))
    schemas.append((
        Implies(
            Not(Equality(vx, vy)),
            Implies(
                Equality(vy, vz),
                ForAll("x", Equality(vy, vz)),
            ),
        ),
        "ax-13",
    ))

    return schemas


def _make_metamath_definitional_axioms() -> List[Tuple[Formula, str]]:
    """Build Metamath set.mm definitional axioms df-an, df-or, df-bi, df-ex.

    These introduce abbreviations ∧, ∨, ↔, ∃ as biconditional axioms.
    In set.mm they are axioms (Variant B), not syntactic expansions.
    """
    x0 = PropVar("x0")
    x1 = PropVar("x1")

    schemas: List[Tuple[Formula, str]] = []

    # df-an: (φ ∧ ψ) ↔ ¬(φ → ¬ψ)
    schemas.append((
        Iff(And(x0, x1), Not(Implies(x0, Not(x1)))),
        "df-an",
    ))

    # df-or: (φ ∨ ψ) ↔ (¬φ → ψ)
    schemas.append((
        Iff(Or(x0, x1), Implies(Not(x0), x1)),
        "df-or",
    ))

    # df-bi: (φ ↔ ψ) ↔ ¬((φ → ψ) → ¬(ψ → φ))
    schemas.append((
        Iff(
            Iff(x0, x1),
            Not(Implies(Implies(x0, x1), Not(Implies(x1, x0)))),
        ),
        "df-bi",
    ))

    # df-ex: ∃xφ ↔ ¬∀x¬φ
    schemas.append((
        Iff(Exists("x", x0), Not(ForAll("x", Not(x0)))),
        "df-ex",
    ))

    return schemas


def _collect_term_symbols(formula: Formula, var_names: Set[str], const_names: Set[str]) -> None:
    """Collect first-order variable/constant symbols from a formula."""

    def _add_term(term: Term) -> None:
        if term.is_variable:
            var_names.add(term.name)
        else:
            const_names.add(term.name)

    t = type(formula)
    if t is Membership or t is Equality:
        _add_term(formula.left)
        _add_term(formula.right)
        return

    if t is Not or t is Box:
        _collect_term_symbols(formula.inner, var_names, const_names)
        return

    if t is And or t is Or or t is Iff:
        _collect_term_symbols(formula.left, var_names, const_names)
        _collect_term_symbols(formula.right, var_names, const_names)
        return

    if t is Implies:
        _collect_term_symbols(formula.antecedent, var_names, const_names)
        _collect_term_symbols(formula.consequent, var_names, const_names)
        return

    if t is ForAll or t is Exists:
        var_names.add(formula.variable)
        _collect_term_symbols(formula.body, var_names, const_names)
        return

    # Second-order and infinitary types — recurse into sub-formulas.
    from core.formula import ForAllPred, ExistsPred, PredApp, BigAnd, BigOr
    if t is ForAllPred or t is ExistsPred:
        _collect_term_symbols(formula.body, var_names, const_names)
        return

    if t is PredApp:
        for arg in formula.args:
            if arg.is_variable:
                var_names.add(arg.name)
            else:
                const_names.add(arg.name)
        return

    if t is BigAnd or t is BigOr:
        for sub in formula.formulas:
            _collect_term_symbols(sub, var_names, const_names)
        return

    # PropVar, PredicateVar, and unknown atom-like formulas carry no FO term symbols.


def _subst_pred_app(
    pred_name: str,
    param_vars: List[str],
    template_body: Formula,
    formula: Formula,
) -> Formula:
    """Substitute PredApp(pred_name, args) with template_body[params := args].

    This implements second-order instantiation: given a predicate template
    λ(v₁,...,vₙ).ψ(v₁,...,vₙ), replaces every P(t₁,...,tₙ) in formula
    with ψ[v₁:=t₁, ..., vₙ:=tₙ].

    Capture-avoidance: when descending under a first-order quantifier
    ∀v / ∃v whose variable v occurs FREE in template_body (excluding the
    formal parameters), the bound variable is α-renamed to a fresh name
    before substitution to prevent capture. This makes Pred-Inst sound
    even on formulas whose bound variable names happen to collide with
    template-free names.
    """
    t = type(formula)

    if t is PredApp:
        if formula.pred_var_name == pred_name:
            # Replace: P(t₁,...,tₙ) → template_body[params := args]
            result = template_body
            for pvar, arg in zip(param_vars, formula.args):
                result = result.substitute(pvar, arg)
            return result
        return formula

    if t is Implies:
        return Implies(
            _subst_pred_app(pred_name, param_vars, template_body, formula.antecedent),
            _subst_pred_app(pred_name, param_vars, template_body, formula.consequent),
        )
    if t is Not:
        return Not(_subst_pred_app(pred_name, param_vars, template_body, formula.inner))
    if t is And:
        return And(
            _subst_pred_app(pred_name, param_vars, template_body, formula.left),
            _subst_pred_app(pred_name, param_vars, template_body, formula.right),
        )
    if t is Or:
        return Or(
            _subst_pred_app(pred_name, param_vars, template_body, formula.left),
            _subst_pred_app(pred_name, param_vars, template_body, formula.right),
        )
    if t is Iff:
        return Iff(
            _subst_pred_app(pred_name, param_vars, template_body, formula.left),
            _subst_pred_app(pred_name, param_vars, template_body, formula.right),
        )
    if t is Box:
        return Box(_subst_pred_app(pred_name, param_vars, template_body, formula.inner))
    if t is ForAll:
        return _subst_pred_app_under_quant(
            ForAll, pred_name, param_vars, template_body, formula
        )
    if t is Exists:
        return _subst_pred_app_under_quant(
            Exists, pred_name, param_vars, template_body, formula
        )
    if t is ForAllPred:
        # If this re-binds the same pred var, stop
        if formula.pred_var == pred_name:
            return formula
        return ForAllPred(formula.pred_var, formula.arity,
                          _subst_pred_app(pred_name, param_vars, template_body, formula.body))
    if t is ExistsPred:
        if formula.pred_var == pred_name:
            return formula
        return ExistsPred(formula.pred_var, formula.arity,
                          _subst_pred_app(pred_name, param_vars, template_body, formula.body))
    if t is BigAnd:
        return BigAnd(tuple(
            _subst_pred_app(pred_name, param_vars, template_body, f)
            for f in formula.formulas
        ))
    if t is BigOr:
        return BigOr(tuple(
            _subst_pred_app(pred_name, param_vars, template_body, f)
            for f in formula.formulas
        ))
    # Atomic (Membership, Equality, PropVar, PredicateVar) — no PredApp inside
    return formula


def _subst_pred_app_under_quant(
    quant_cls,
    pred_name: str,
    param_vars: List[str],
    template_body: Formula,
    formula: Formula,
) -> Formula:
    """Capture-avoiding descent under a first-order quantifier (ForAll/Exists).

    If the quantifier's bound variable matches a free variable of
    ``template_body`` (excluding formal parameters), α-rename the bound
    variable to a fresh name before recursing so that template-free
    variables cannot be captured by the surrounding binder.
    """
    bound_var = formula.variable
    body = formula.body
    template_free = template_body.free_variables() - set(param_vars)

    if bound_var in template_free:
        avoid = template_free | body.free_variables() | {bound_var}
        fresh = Formula._fresh_variable(avoid, prefix="_pi_q")
        renamed_body = body.substitute(
            bound_var, Term(fresh, is_variable=True)
        )
        return quant_cls(
            fresh,
            _subst_pred_app(pred_name, param_vars, template_body, renamed_body),
        )

    return quant_cls(
        bound_var,
        _subst_pred_app(pred_name, param_vars, template_body, body),
    )


def _collect_pred_var_names(formula: Formula, acc: set) -> None:
    """Collect (name, arity) pairs for all ForAllPred/ExistsPred bindings in formula."""
    t = type(formula)
    if t is ForAllPred or t is ExistsPred:
        acc.add((formula.pred_var, formula.arity))
        _collect_pred_var_names(formula.body, acc)
    elif t is Implies:
        _collect_pred_var_names(formula.antecedent, acc)
        _collect_pred_var_names(formula.consequent, acc)
    elif t is Not or t is Box:
        _collect_pred_var_names(formula.inner, acc)
    elif t in (And, Or, Iff):
        _collect_pred_var_names(formula.left, acc)
        _collect_pred_var_names(formula.right, acc)
    elif t is ForAll or t is Exists:
        _collect_pred_var_names(formula.body, acc)
    elif t is BigAnd or t is BigOr:
        for f in formula.formulas:
            _collect_pred_var_names(f, acc)


def _contains_propvar(formula: Formula) -> bool:
    """True iff formula contains any PropVar metavariable (anywhere)."""
    t = type(formula)
    if t is PropVar:
        return True
    if t is Implies:
        return _contains_propvar(formula.antecedent) or _contains_propvar(formula.consequent)
    if t in (And, Or, Iff):
        return _contains_propvar(formula.left) or _contains_propvar(formula.right)
    if t is Not or t is Box:
        return _contains_propvar(formula.inner)
    if t is ForAll or t is Exists:
        return _contains_propvar(formula.body)
    if t is ForAllPred or t is ExistsPred:
        return _contains_propvar(formula.body)
    if t is BigAnd or t is BigOr:
        return any(_contains_propvar(f) for f in formula.formulas)
    # PropVar handled above; PredicateVar/PredApp/Membership/Equality contain none.
    return False


def _is_pure_propositional(formula: Formula) -> bool:
    """True iff ``formula`` is built ONLY from PropVar + propositional connectives
    (→, ∧, ∨, ↔, ¬, □) — i.e. contains NO ZFC/FO content (no ∈, =, quantifier,
    predicate). Such depth>0 schemas are the propositional-pattern explosion;
    pruning them (keeping ZFC-content + the propositional axioms) stops the
    self-breeding. See report §2.10 / ZFC_PRUNE_PROPOSITIONAL."""
    t = type(formula)
    if t is PropVar:
        return True
    if t is Implies:
        return (_is_pure_propositional(formula.antecedent)
                and _is_pure_propositional(formula.consequent))
    if t in (And, Or, Iff):
        return (_is_pure_propositional(formula.left)
                and _is_pure_propositional(formula.right))
    if t is Not or t is Box:
        return _is_pure_propositional(formula.inner)
    # Membership, Equality, ForAll, Exists, PredApp, PredicateVar, ForAllPred,
    # ExistsPred, BigAnd/BigOr → ZFC/FO content → not pure propositional.
    return False


def _propvar_name_occurs(name: str, formula: Formula) -> bool:
    """True iff a PropVar named ``name`` occurs anywhere in ``formula``."""
    t = type(formula)
    if t is PropVar:
        return formula.name == name
    if t is Implies:
        return (_propvar_name_occurs(name, formula.antecedent)
                or _propvar_name_occurs(name, formula.consequent))
    if t in (And, Or, Iff):
        return (_propvar_name_occurs(name, formula.left)
                or _propvar_name_occurs(name, formula.right))
    if t is Not or t is Box:
        return _propvar_name_occurs(name, formula.inner)
    if t is ForAll or t is Exists:
        return _propvar_name_occurs(name, formula.body)
    if t is ForAllPred or t is ExistsPred:
        return _propvar_name_occurs(name, formula.body)
    if t is BigAnd or t is BigOr:
        return any(_propvar_name_occurs(name, f) for f in formula.formulas)
    return False


def _contains_pred_ref(pred_name: str, formula: Formula) -> bool:
    """Check if formula contains any PredApp or ForAllPred/ExistsPred referencing pred_name."""
    t = type(formula)
    if t is PredApp:
        return formula.pred_var_name == pred_name
    if t is ForAllPred or t is ExistsPred:
        if formula.pred_var == pred_name:
            return True  # bound occurrence
        return _contains_pred_ref(pred_name, formula.body)
    if t is Implies:
        return (_contains_pred_ref(pred_name, formula.antecedent) or
                _contains_pred_ref(pred_name, formula.consequent))
    if t is Not or t is Box:
        return _contains_pred_ref(pred_name, formula.inner)
    if t in (And, Or, Iff):
        return (_contains_pred_ref(pred_name, formula.left) or
                _contains_pred_ref(pred_name, formula.right))
    if t is ForAll or t is Exists:
        return _contains_pred_ref(pred_name, formula.body)
    if t is BigAnd or t is BigOr:
        return any(_contains_pred_ref(pred_name, f) for f in formula.formulas)
    return False


class SchemaGenerator:
    """BFS generator for Hilbert schema proof graphs."""

    graph_kind = "primary_schema_formula_space"
    schema_instantiation_policy = "schema_unification_v1"

    def __init__(
        self,
        logic: str = "propositional",
        max_depth: Optional[int] = None,
        max_schemas: Optional[int] = None,
        max_formula_size: int = 500,
        enable_con3i: bool = False,
        enable_necessitation: bool = True,
        enable_quantifier_rules: bool = True,
        enable_generalization: bool = True,
        zfc_include_choice: bool = True,
        zfc_include_schema_instances: bool = True,
        sep_instance_limit: Optional[int] = None,
        rep_instance_limit: Optional[int] = None,
        propositional_base: Optional[str] = None,
        theory: Optional[str] = None,
        convert_to_primitive: Optional[bool] = None,
        n_workers: int = 1,
        use_pattern_unify: bool = False,
        sep_enum_k: Optional[int] = None,
        prune_propositional: bool = False,
        reuse_core: bool = False,
        mp_parallel_threshold: int = 50,
        mp_result_batch_size: int = 10_000,
        provenance_mode: str = "full",
        subsumption_mode: str = "off",
        size_cap_strict: Optional[bool] = None,
        checkpoint_mode: Optional[str] = None,
        beam_width: Optional[int] = None,
        beam_threshold: Optional[float] = None,
        utility_ranker: Optional["UtilityRanker"] = None,
        stop_on_watchlist: bool = False,
        beam_controller: Optional["AdaptiveBeam"] = None,
        _use_env_overrides: bool = True,
    ):
        self.logic = logic
        self.max_depth = max_depth
        self.max_schemas = max_schemas
        # C10: None ⇒ unbounded. Saturation is an EMPTY frontier, never a
        # counter hit. `_schema_cap` is the numeric guard used by rule loops;
        # ∞ when unbounded so no guard ever fires silently.
        self._schema_cap: float = max_schemas if max_schemas is not None else math.inf
        self.max_formula_size = max_formula_size
        configured_checkpoint_mode = (
            "sync" if checkpoint_mode is None else str(checkpoint_mode).strip().lower()
        )
        if configured_checkpoint_mode not in CHECKPOINT_MODES:
            raise ValueError(
                f"checkpoint_mode must be one of {sorted(CHECKPOINT_MODES)}, "
                f"got {checkpoint_mode!r}"
            )
        environment_checkpoint_mode = (
            _checkpoint_mode_from_environment() if _use_env_overrides else None
        )
        if (
            checkpoint_mode is not None
            and environment_checkpoint_mode is not None
            and environment_checkpoint_mode != configured_checkpoint_mode
        ):
            raise ValueError(
                "checkpoint_mode conflicts with the explicit environment switch: "
                f"{configured_checkpoint_mode!r} != {environment_checkpoint_mode!r}"
            )
        self.checkpoint_mode = environment_checkpoint_mode or configured_checkpoint_mode
        # C3 / I9: proof-length (depth) is the SCIENTIFIC boundary; formula size
        # is only a generous SAFETY cap. Per крит §4 a binding size-cap silently
        # drops cut-detour theorems, so the object would no longer be a faithful
        # proof-length-d closure (RESEARCH_VISION §6). It must not bind in normal
        # runs. ZFC_SIZE_CAP_STRICT=1 turns a bind into a hard RuntimeError (for
        # calibration / CI); otherwise a bind is reported loudly at end of run.
        self.size_cap_strict = (
            bool(size_cap_strict) if size_cap_strict is not None
            else (
                _use_env_overrides
                and os.environ.get("ZFC_SIZE_CAP_STRICT", "").strip() == "1"
            )
        )
        self.zfc_include_choice = zfc_include_choice
        self.zfc_include_schema_instances = zfc_include_schema_instances
        self.sep_instance_limit = sep_instance_limit
        self.rep_instance_limit = rep_instance_limit
        self.n_workers = max(1, int(n_workers))
        # Phase A I4: pattern-unification engine (faithful, lazy ?P instances).
        # Env override enables it without code changes. Default off (eager path).
        self.use_pattern_unify = bool(use_pattern_unify) or (
            _use_env_overrides
            and os.environ.get("ZFC_PATTERN_UNIFY", "").strip() == "1"
        )
        # Option B (principled Separation): if set, the axiom layer uses ALL
        # φ of complexity ≤ k instead of the hand-picked 12 (uniform, convergent
        # to the true ZFC schema). None = legacy eager (no change).
        _env_k = (
            os.environ.get("ZFC_SEP_ENUM_K", "").strip()
            if _use_env_overrides else ""
        )
        if sep_enum_k is not None:
            try:
                parsed_sep_enum_k = int(sep_enum_k)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"sep_enum_k must be an integer >= 1, got {sep_enum_k!r}") from exc
        elif _env_k:
            try:
                parsed_sep_enum_k = int(_env_k)
            except ValueError as exc:
                raise ValueError(f"ZFC_SEP_ENUM_K must be an integer >= 1, got {_env_k!r}") from exc
        else:
            parsed_sep_enum_k = None
        if parsed_sep_enum_k is not None:
            if parsed_sep_enum_k < 1:
                raise ValueError(f"sep_enum_k must be >= 1, got {parsed_sep_enum_k!r}")
            if sep_instance_limit is not None or rep_instance_limit is not None:
                raise ValueError(
                    "sep_instance_limit/rep_instance_limit cannot be combined with "
                    "sep_enum_k; Option B is fixed as ZFC_base + Sep<=k + Rep4"
                )
        self.sep_enum_k = parsed_sep_enum_k
        # §13–14: demand-driven Separation (env ZFC_SEP_MODE=demand). Instead of the
        # eager enumeration of all φ≤k (≈16–30% trivial seed, §13.1), seed only the
        # LITERAL predicates (atoms + negations, k≤2, incl Russell ¬(z∈z)) and grow
        # Separation ON DEMAND from the unary first-order predicates the logic actually
        # CONSTRUCTS, harvested per depth (_apply_separation_demand). Generative /
        # predicative; preserves the scale-free content topology (proof §14, Thm 1–2).
        # Default off → zero regression. ZFC_SEP_PRED_MAXSIZE = predicate weight bound.
        self._sep_demand_mode = (
            str(os.environ.get("ZFC_SEP_MODE", "")).strip().lower() == "demand"
        )
        try:
            self._sep_pred_maxsize = int(os.environ.get("ZFC_SEP_PRED_MAXSIZE", "7"))
        except ValueError:
            self._sep_pred_maxsize = 7
        self._sep_demand_seen: set = set()
        self._sep_demand_added = 0
        # ZFC-relevance pruning (§ FORMULA_SPACE / report): drop pure-propositional
        # DERIVED schemas (depth>0) so propositional patterns don't breed each other
        # (the dominant explosion). Keeps ZFC-content nodes + the propositional
        # AXIOMS (depth 0) as rules. NOT faithful to the full proof-length closure
        # (loses propositional detours) — validate ZFC-hub equivalence via A/B.
        self.prune_propositional = bool(prune_propositional) or (
            os.environ.get("ZFC_PRUNE_PROPOSITIONAL", "").strip() == "1"
        )
        # Content-space: drop top-level conjunction nodes (∧ = edge, not vertex);
        # node set = irreducible ZFC facts, logic = d0 rules (the parameter). §13.
        self._content_space = os.environ.get("ZFC_CONTENT_SPACE", "").strip() == "1"
        # Reuse-core mode: tally last-layer premise reuse instead of materializing
        # the inert frontier (§2.13). hub_score(P) = out_degree(P) + reuse_tally(P).
        self.reuse_core = bool(reuse_core) or (
            os.environ.get("ZFC_REUSE_CORE", "").strip() == "1"
        )
        self._reuse_core = self.reuse_core
        # Per-premise frontier reuse COUNT (not a (pid,hash) edge set — that was
        # O(283M) at d3 and caused node-memory blowup / swap-hang). With
        # detachable wildcards emitted once, non-detachable wildcards produce a
        # DISTINCT ψ per minor → firing count ≈ distinct out-degree, so a counter
        # is both O(#nodes) memory and rank-faithful.
        self._frontier_outdeg: Dict[int, int] = {}
        self.mp_parallel_threshold = max(1, int(mp_parallel_threshold))
        self.mp_result_batch_size = max(1, int(mp_result_batch_size))
        if provenance_mode not in PROVENANCE_MODES:
            raise ValueError(
                f"Unknown provenance_mode={provenance_mode!r}; "
                f"expected one of {sorted(PROVENANCE_MODES)}"
            )
        self.provenance_mode = provenance_mode
        if subsumption_mode not in SUBSUMPTION_MODES:
            raise ValueError(
                f"Unknown subsumption_mode={subsumption_mode!r}; "
                f"expected one of {sorted(SUBSUMPTION_MODES)}"
            )
        self.subsumption_mode = subsumption_mode

        # Subsumption-mode state (lazy: only allocate when active).
        if self.subsumption_mode == "off":
            self._subsumption_index: Optional[SubsumptionIndex] = None
        else:
            self._subsumption_index = SubsumptionIndex()
        self._subsumed_derivations: Dict[int, List[SubsumedRecord]] = {}
        # Maps retired entry id → id of the subsumer that replaced it (full mode only).
        self._retired: Dict[int, int] = {}

        # Two configurations require a canonical merge order.
        #
        # * Online subsumption is an order-sensitive state transition: accepting
        #   a general schema changes whether later instances are materialised.
        # * Full provenance records every rediscovery hyperedge.  A depth-3
        #   audit found that merging worker batches by arrival order preserves
        #   the layered vertex set but can change the proof-hyperedge set.  Until
        #   an external canonical reorderer is available, serial rule traversal
        #   is the exact implementation of that scientific object.
        #
        # An in-memory reorder buffer is intentionally avoided: at depth 3 it
        # can retain an entire slow predecessor chunk.  Parallel generation is
        # still available for first-discovery provenance with subsumption off.
        self._ordered_rule_application_reason: Optional[str]
        if self.subsumption_mode != "off":
            self._ordered_rule_application_reason = "online_subsumption"
        elif self.provenance_mode == "full":
            self._ordered_rule_application_reason = "full_provenance"
        else:
            self._ordered_rule_application_reason = None
        self._ordered_rule_application_required = (
            self._ordered_rule_application_reason is not None
        )

        # ── Directed growth (полезность формулы, given-clause beam) ──────
        # docs/DIRECTED_GROWTH_LITERATURE.md §3. Если beam_width задан, новые
        # схемы попадают в ПАССИВНЫЙ ПУЛ, а посылками следующего раунда
        # становятся только top-B по U(φ) (utility_v1). Объект при этом —
        # НЕ полное proof-length-замыкание, а «направленное замыкание»;
        # политика отбора фиксируется в stats["growth_policy"].
        _env_beam = (
            os.environ.get("ZFC_BEAM_WIDTH", "").strip()
            if _use_env_overrides else ""
        )
        if beam_width is None and _env_beam:
            beam_width = int(_env_beam)
        if beam_width is not None and beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width!r}")
        self.beam_controller = beam_controller
        if beam_controller is not None and beam_width is None:
            beam_width = beam_controller.width
        self.beam_width = beam_width
        # Порог полезности вместо фикс. count: активируются ВСЕ φ с U(φ) ≥ τ
        # (число на слой плавает по логике, не навязано). beam_width в этом
        # режиме — лишь безопасный потолок (по умолчанию практически снят).
        _env_thr = (os.environ.get("ZFC_BEAM_THRESHOLD", "").strip()
                    if _use_env_overrides else "")
        if beam_threshold is None and _env_thr:
            beam_threshold = _env_thr if _env_thr == "auto" else float(_env_thr)
        self.beam_threshold = beam_threshold   # число, либо "auto" (Otsu на скорах слоя)
        if self.beam_threshold is not None and self.beam_width is None:
            self.beam_width = 10 ** 9
        # Доля beam, отдаваемая СТАРЕЙШИМ из пула (age-weight ratio как в given-clause
        # ATP: возраст→полнота, вес/U→фокус). 0=чистый best-first, 1=чистый FIFO.
        self._beam_age_frac = float(os.environ.get("ZFC_BEAM_AGE_FRAC", "0.25") or 0.25)
        self.utility = utility_ranker
        self.stop_on_watchlist = bool(stop_on_watchlist)
        if self.beam_width is not None and self.subsumption_mode == "full":
            raise ValueError(
                "beam mode (directed growth v1) supports subsumption_mode "
                "off/forward only; 'full' backward-retirement would have to "
                "also retire pooled entries (not implemented)"
            )
        # Пассивный пул: id → entry (создана, но не активирована как посылка).
        self._passive_pool: Dict[int, SchemaEntry] = {}
        # Счётчик переоткрытий (rediscovery) для компоненты U.
        self._rediscovery_counts: Dict[int, int] = {}
        self._activated_ids: Set[int] = set()
        # Кэш U(φ): id → (score, rediscovery_при_вычислении, hits_version).
        self._utility_cache: Dict[int, Tuple[float, int, int]] = {}

        # Derive propositional_base from legacy logic= if not explicitly given
        if propositional_base is not None:
            self.propositional_base = propositional_base
        elif logic == "metamath":
            self.propositional_base = "metamath"
        elif logic == "metamath_prop":
            self.propositional_base = "metamath_prop"
        elif logic == "modal_k":
            self.propositional_base = "modal_k"
        else:
            self.propositional_base = "h16"

        # Derive theory from legacy logic= if not explicitly given
        if theory is not None:
            self.theory = theory
        elif logic in ("zfc", "metamath"):
            self.theory = "zfc"
        else:
            self.theory = None

        _fol_theories = {"zfc", "izf", "kp", "lomegaomega"}
        _sol_theories = {"zfc2", "msol", "henkin_sol"}
        _all_fo_like = _fol_theories | _sol_theories
        _modal_bases = {"modal_k", "modal_t", "modal_s4", "modal_s5", "modal_gl"}
        self.enable_con3i = enable_con3i and self.propositional_base not in _modal_bases
        self.enable_necessitation = enable_necessitation and self.propositional_base in _modal_bases
        self.enable_quantifier_rules = enable_quantifier_rules and self.theory in _all_fo_like
        # C15-∃ (I8): native ∃-intro/∃-elim rules. Gated to the intuit/minimal
        # bases, which keep ∃ native and otherwise have no rule for it. Classical
        # h16 expands ∃ via convert_to_primitive (no ∃ nodes) and metamath uses
        # df-ex, so neither base gets these — matching the C15 connective gating.
        self.enable_exists_rules = (
            self.enable_quantifier_rules
            and self.propositional_base in ("h16_intuit", "h16_minimal")
        )
        self.enable_generalization = (
            enable_generalization
            and (self.theory in _all_fo_like or self.propositional_base == "metamath")
        )
        self.enable_pred_rules = self.theory in _sol_theories

        # Resolve convert_to_primitive: only auto-True for classical FOL
        # (h16 base), where the primitive-form definitions are sound.
        # For metamath (df-* bridges), intuitionistic, minimal, and modal
        # bases the conversion is either redundant or methodologically
        # unsound (classical equivalences fail in IPC).
        if convert_to_primitive is None:
            self.convert_to_primitive = (self.propositional_base == "h16")
        else:
            self.convert_to_primitive = bool(convert_to_primitive)

        # Graph storage
        # Defer the 2-section projection (derived view, not read during
        # generation) → build it lazily/post-hoc. Removes the dominant online
        # cost (_add_projection_edge) from the hot merge and keeps it out of
        # checkpoints. Set ZFC_EAGER_PROJECTION=1 to force the old eager path
        # (e.g. for A/B graph-identity checks).
        defer_projection = os.environ.get("ZFC_EAGER_PROJECTION", "").strip() != "1"
        self.graph = ProofGraph(
            graph_kind=self.graph_kind,
            projection_kind="directed_2_section",
            defer_projection=defer_projection,
            metadata={
                "canonicalization": "structural_formula_form_v1",
                "schema_instantiation_policy": self.schema_instantiation_policy,
                "contains_schema_vertices": True,
                "rule_merge_policy": self._effective_rule_merge_policy(),
                **provenance_metadata(self.provenance_mode),
            },
        )
        self._entries: Dict[str, SchemaEntry] = {}
        self._by_depth: Dict[int, List[SchemaEntry]] = {}
        self._formula_intern_pool: Dict[str, Formula] = {}
        self._next_id: int = 0

        # Symbol pools for ZFC quantifier rules
        self._term_vars: Set[str] = set()
        self._term_consts: Set[str] = set()
        # Cached _candidate_terms() result; invalidated when term pools grow.
        self._candidate_terms_cache: Optional[List[Term]] = None
        self._reset_transient_state()

        # Stats
        self.stats: Dict[str, Any] = {
            "mp_attempts": 0,
            "mp_successes": 0,
            "mp_filtered_size": 0,
            "mp_filtered_dup": 0,
            "con3i_added": 0,
            "nec_added": 0,
            "a4_added": 0,
            "a5_added": 0,
            "q1_added": 0,
            "q2_added": 0,
            "exists_intro_added": 0,
            "exists_elim_added": 0,
            "gen_added": 0,
            "pred_inst_added": 0,
            "pred_gen_added": 0,
            "zfc_axioms_added": 0,
            "mp_candidate_pairs_raw": 0,
            "mp_candidate_pairs_filtered": 0,
            "mp_pass_sec": 0.0,
            "quantifier_rules_sec": 0.0,
            "generalization_sec": 0.0,
            "checkpoint_sec": 0.0,
            "schemas_by_depth": {},
            "timings_by_depth": {},
            "canonicalization": "structural_formula_form_v1",
            "graph_kind": self.graph_kind,
            "projection_kind": "directed_2_section",
            "schema_instantiation_policy": self.schema_instantiation_policy,
            "contains_schema_vertices": True,
            "rule_merge_policy": self._effective_rule_merge_policy(),
            "mp_parallel_forced_serial_passes": 0,
            "mp_parallel_forced_serial_subsumption_passes": 0,
            "quantifier_parallel_forced_serial_passes": 0,
            "quantifier_parallel_forced_serial_subsumption_passes": 0,
            "mp_result_batch_size": self.mp_result_batch_size,
            **provenance_metadata(self.provenance_mode),
            "rediscovery_hyperedges_skipped": 0,
            "macro_rules": ["con3i"] if self.enable_con3i else [],
            "convert_to_primitive": self.convert_to_primitive,
            "axiom_form": "primitive" if self.convert_to_primitive else "native",
        }
        if self.beam_width is not None:
            self.stats["growth_policy"] = {
                "mode": "directed_v1",
                "beam_width": self.beam_width,
                **(self.beam_controller.metadata()
                   if self.beam_controller is not None else {}),
                **(self.utility.metadata() if self.utility is not None else {}),
            }
            self.stats["watchlist_hits"] = {}
            self.stats["activated_by_depth"] = {}
            self.stats["pool_size_by_depth"] = {}

    def set_max_schemas(self, max_schemas: Optional[int]) -> None:
        """Set the schema-count guard and its internal numeric cap together."""
        self.max_schemas = max_schemas
        self._schema_cap = max_schemas if max_schemas is not None else math.inf

    def _reset_transient_state(self) -> None:
        """Reset in-memory MP indexes rebuilt on init/load."""
        self._all_implication_candidates: List[MPCandidate] = []
        self._implication_candidates_by_tag: Dict[int, List[MPCandidate]] = {}
        self._wildcard_implication_candidates: List[MPCandidate] = []
        self._implication_candidate_by_id: Dict[int, MPCandidate] = {}
        # C6: id→entry reverse index so `_entry_by_id` is O(1) instead of an
        # O(|entries|) scan. Derived from `_entries`, so it is reset here and
        # rebuilt on checkpoint load (see `_rebuild_transient_state`).
        self._id_to_entry: Dict[int, SchemaEntry] = {}

    def _register_entry(self, entry: SchemaEntry) -> None:
        """Register a new entry in transient MP indexes."""
        if type(entry.formula) is not Implies:
            return

        ant = entry.formula.antecedent
        antecedent_tag = id(type(ant))
        constraints = tuple(detect_implicit_dvar_constraints(entry.formula))
        # I5: in pattern mode a free predicate-metavar antecedent ?P(...) is
        # flexible (unifies with many minors) → bucket it as a wildcard so it
        # is tried against every minor, not only PredApp-headed ones.
        is_wildcard = (antecedent_tag == _PROPVAR_TOP_TAG) or (
            self.use_pattern_unify and type(ant) is PredApp
        )
        # Detachment memo: a PropVar-antecedent major ``P → ψ`` with ``P ∉ ψ``
        # and no distinct-variable constraints yields the SAME canonical ψ for
        # every minor under MP. Precompute it once so the MP pass can skip the
        # per-minor schema_mp + canonicalize (object-preserving fast path).
        detached_result = None
        if (
            antecedent_tag == _PROPVAR_TOP_TAG
            and not constraints
            and os.environ.get("ZFC_DISABLE_DETACH") != "1"
            and not _propvar_name_occurs(ant.name, entry.formula.consequent)
        ):
            detached_result = canonicalize_with_hash(entry.formula.consequent)
        candidate = MPCandidate(
            entry=entry,
            major_vars=frozenset(collect_vars(entry.formula)),
            antecedent_tag=antecedent_tag,
            antecedent_is_wildcard=is_wildcard,
            dvar_constraints=constraints,
            detached_result=detached_result,
        )
        self._all_implication_candidates.append(candidate)
        self._implication_candidate_by_id[entry.id] = candidate
        if candidate.antecedent_is_wildcard:
            self._wildcard_implication_candidates.append(candidate)
        else:
            self._implication_candidates_by_tag.setdefault(
                candidate.antecedent_tag, []
            ).append(candidate)

    def _rebuild_transient_state(self) -> None:
        """Rebuild transient MP indexes after loading a checkpoint."""
        self._reset_transient_state()
        for entry in sorted(self._entries.values(), key=lambda item: item.id):
            self._id_to_entry[entry.id] = entry
            self._register_entry(entry)

    def _intern_formula(self, normalized_hash: str, formula: Formula) -> Formula:
        """Return the canonical in-memory root object for a normalized formula."""
        interned = self._formula_intern_pool.get(normalized_hash)
        if interned is not None:
            return interned
        self._formula_intern_pool[normalized_hash] = formula
        return formula

    def _rebuild_formula_intern_pool(self) -> None:
        """Rebuild root-formula interning after checkpoint load."""
        self._formula_intern_pool = {
            entry.schema_hash: entry.formula
            for entry in self._entries.values()
        }

    @staticmethod
    def _candidate_lists_for_minor(
        minor_formula: Formula,
        *,
        all_candidates: List[MPCandidate],
        wildcard_candidates: List[MPCandidate],
        candidates_by_tag: Dict[int, List[MPCandidate]],
    ) -> Tuple[List[MPCandidate], ...]:
        """Return candidate buckets compatible with a given minor formula."""
        minor_tag = id(type(minor_formula))
        if minor_tag == _PROPVAR_TOP_TAG:
            return (all_candidates,)
        tagged = candidates_by_tag.get(minor_tag)
        if tagged:
            return (wildcard_candidates, tagged)
        return (wildcard_candidates,)

    @staticmethod
    def _count_candidate_lists(candidate_lists: Tuple[List[MPCandidate], ...]) -> int:
        """Return the total number of candidates across multiple buckets."""
        return sum(len(bucket) for bucket in candidate_lists)

    def _rediscovery_signal(self, entry: SchemaEntry) -> int:
        """Сигнал «естественной сходимости» для компоненты rediscovery U(φ)
        и ключа кэша U. Дефолт (utility_v1): сырой счётчик переоткрытий.

        Hook G17 (utility_v2_mindep): сабкласс возвращает m_indep — число
        различных форвардных наборов посылок (ведётся инкрементально через
        ``_on_rediscovery_event``). Дефолт — бит-в-бит прежнее поведение.
        """
        return self._rediscovery_counts.get(entry.id, 0)

    def _on_rediscovery_event(self, existing: SchemaEntry,
                              premise_ids: List[int]) -> None:
        """Колбэк переоткрытия существующей схемы (beam-режим). Дефолт —
        no-op; сабкласс G17 копит различные форвардные наборы посылок."""

    def _beam_score_perturb(self, score: float, entry: SchemaEntry) -> float:
        """Перторбация U(φ) при ранжировании бима (дефолт: identity).

        Hook эскалационной ветки G15a (джиттер): сабкласс добавляет шум
        поверх сырого U. Кэш U хранит сырое значение — шум применяется
        на чтении и пересэмплируется каждый раунд. Дефолт — без
        изменений (бит-в-бит).
        """
        return score

    def _beam_tiebreak_key(self, entry: SchemaEntry):
        """Тай-брейк ранжирования бима среди равных U(φ) (дефолт: id).

        Hook для ансамблей траекторий (G15a, experiments/wild_growth):
        сабкласс может вернуть случайный ключ от сид-RNG, рандомизируя
        выбор ТОЛЬКО внутри блоков равного U. Дефолт (entry.id)
        сохраняет прежний порядок бит-в-бит. Age-четверть бима этим
        hook'ом не затрагивается (анти-starvation floor).
        """
        return entry.id

    def _activate_from_pool(self, round_depth: int) -> int:
        """Directed growth: активировать top-B пассивного пула по U(φ).

        Активированные схемы попадают в bucket ``round_depth`` (посылки
        следующего раунда) и регистрируются в MP-индексах. Остальные ждут
        в пуле (могут быть активированы позже — given-clause семантика).
        """
        if not self._passive_pool:
            return 0
        k = self.beam_width or 0
        if self.beam_controller is not None:
            hits_total = (
                len(self.utility.hits) if self.utility is not None else 0
            )
            k = self.beam_controller.next_width(hits_total)
            self.beam_width = k
            self.stats.setdefault("beam_by_depth", {})[round_depth] = k
        # Компас новизны (compression-objective): периодически перестроить
        # библиотеку абстракций из активированных вершин, чтобы novelty(φ)
        # отражала текущий веер (docs §3.13). Любой importance_model с .refresh.
        im = getattr(self.utility, "importance_model", None) if self.utility else None
        if im is not None and hasattr(im, "refresh"):
            self._compass_round = getattr(self, "_compass_round", 0) + 1
            every = getattr(im, "refresh_every", 2) or 2
            if (not getattr(im, "ready", False)) or (self._compass_round % every == 0):
                acts = [self._id_to_entry[i].formula
                        for i in self._activated_ids if i in self._id_to_entry]
                im.refresh(acts)
        best_score = None
        if self.utility is not None:
            # Age/weight-ratio (как в E prover): четверть бима — старейшие из
            # пула (гарантия от starvation/потери полноты в пределе), осталь-
            # ное — top по U(φ).
            k_age = (min(4, len(self._passive_pool)) if self.beam_threshold is not None
                     else (min(k, int(round(k * self._beam_age_frac))) if k >= 4 else 0))
            by_age = sorted(self._passive_pool.values(), key=lambda e: e.id)[:k_age]
            age_ids = {e.id for e in by_age}
            # Кэшированное ранжирование: U(φ) пересчитывается только если
            # изменились rediscovery-счётчик формулы или множество hits.
            hv = self.utility.hits_version
            scored: List[Tuple[float, SchemaEntry]] = []
            for e in self._passive_pool.values():
                if e.id in age_ids:
                    continue
                rc = self._rediscovery_signal(e)
                cached = self._utility_cache.get(e.id)
                if cached is not None and cached[1] == rc and cached[2] == hv:
                    s = cached[0]
                else:
                    s = self.utility.score(e.formula, e.schema_hash, rediscovery=rc)
                    self._utility_cache[e.id] = (s, rc, hv)
                scored.append((self._beam_score_perturb(s, e), e))
            import heapq
            if self.beam_threshold is not None:
                # Порог: все φ с U(φ) ≥ τ (k — безопасный потолок против OOM).
                # τ="auto" → Otsu на скорах слоя (макс. межклассовая дисперсия):
                # число активируемых ПЛАВАЕТ по форме распределения, не фикс. top-B.
                thr = self.beam_threshold
                if thr == "auto":
                    _sv = sorted(s for s, _ in scored)
                    if len(_sv) >= 2:
                        _tot = sum(_sv); _w0 = _s0 = 0.0; _best = -1.0; thr = _sv[0]
                        for _i in range(len(_sv) - 1):
                            _w0 += 1; _s0 += _sv[_i]; _w1 = len(_sv) - _w0
                            _m0 = _s0 / _w0; _m1 = (_tot - _s0) / _w1
                            _val = _w0 * _w1 * (_m0 - _m1) ** 2
                            if _val > _best:
                                _best = _val; thr = (_sv[_i] + _sv[_i + 1]) / 2.0
                    else:
                        thr = _sv[0] if _sv else 0.0
                passing = sorted(((s, e) for s, e in scored if s >= thr),
                                 key=lambda p: (-p[0], self._beam_tiebreak_key(p[1])))
                passing = passing[: max(0, k - len(by_age))]
                chosen = by_age + [entry for _, entry in passing]
                if scored:
                    best_score = max(s for s, _ in scored)
            else:
                ranked = heapq.nsmallest(
                    k - len(by_age), scored,
                    key=lambda p: (-p[0], self._beam_tiebreak_key(p[1])),
                )
                chosen = by_age + [entry for _, entry in ranked]
                if ranked:
                    best_score = ranked[0][0]
            for entry in chosen:
                self._utility_cache.pop(entry.id, None)
        else:
            # Без ранкера — FIFO по id (given-clause «по возрасту»).
            chosen = sorted(self._passive_pool.values(), key=lambda e: e.id)[:k]
        for entry in chosen:
            self._passive_pool.pop(entry.id, None)
            self._activated_ids.add(entry.id)
            self._by_depth.setdefault(round_depth, []).append(entry)
            self._register_entry(entry)
        self.stats.setdefault("activated_by_depth", {})[round_depth] = len(chosen)
        self.stats.setdefault("pool_size_by_depth", {})[round_depth] = len(
            self._passive_pool
        )
        hits = (
            f", watchlist {len(self.utility.hits)}/{self.utility.watchlist_size}"
            if self.utility is not None else ""
        )
        score_txt = f", top U={best_score:.2f}" if best_score is not None else ""
        print(
            f"  Beam: activated {len(chosen)} of pool "
            f"{len(self._passive_pool) + len(chosen):,}{score_txt}{hits}",
            flush=True,
        )
        return len(chosen)

    @staticmethod
    def _checkpoint_depth(path: Path) -> int:
        """Extract numeric depth from checkpoint filename."""
        match = re.search(r"checkpoint_d(\d+)\.pkl\.gz$", path.name)
        if match:
            return int(match.group(1))
        return -1

    def generate(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> ProofGraph:
        """Run BFS schema generation and return ProofGraph.

        Args:
            checkpoint_dir: If set, save checkpoint after each depth to this directory.
        """
        t_start = time.time()

        if self.beam_width is not None and checkpoint_dir is not None:
            raise ValueError(
                "directed growth (beam mode) v1 does not support checkpoints; "
                "run without checkpoint_dir"
            )

        if not self._entries:
            self._init_axioms()
        logger.info(
            "Schema generation: propositional_base=%s, theory=%s, depth-0 schemas=%d",
            self.propositional_base,
            self.theory,
            len(self._entries),
        )

        start_depth = (max(self._by_depth.keys()) + 1) if self._by_depth else 1

        depth = start_depth
        while True:
            if self.max_depth is not None and depth > self.max_depth:
                logger.info("Reached max_depth safety cap=%d. Stopping.", self.max_depth)
                print(f"  Reached depth safety cap ({self.max_depth}).")
                self.stats["stopped_reason"] = "max_depth"
                break

            t_depth = time.time()
            new_count, depth_metrics = self._generate_depth(depth)

            # Directed growth: отобрать top-B из пула как посылки следующего
            # раунда. activated=0 ⇔ пул пуст (истинная сатурация в beam-режиме).
            activated = 0
            if self.beam_width is not None:
                activated = self._activate_from_pool(depth)

            elapsed = time.time() - t_depth
            depth_metrics["depth_sec"] = round(elapsed, 6)

            self.stats["schemas_by_depth"][depth] = new_count
            total = len(self._entries)
            self.stats["mp_candidate_pairs_raw"] = (
                self.stats.get("mp_candidate_pairs_raw", 0)
                + depth_metrics.get("mp_candidate_pairs_raw", 0)
            )
            # mp_attempts keeps the pre-bucket-filter count (raw pairs) so that
            # summaries remain comparable across the bucketization rewrite.
            self.stats["mp_attempts"] = self.stats["mp_candidate_pairs_raw"]
            self.stats["mp_candidate_pairs_filtered"] = (
                self.stats.get("mp_candidate_pairs_filtered", 0)
                + depth_metrics.get("mp_candidate_pairs_filtered", 0)
            )
            self.stats["mp_pass_sec"] = (
                self.stats.get("mp_pass_sec", 0.0)
                + depth_metrics.get("mp_pass_sec", 0.0)
            )
            self.stats["quantifier_rules_sec"] = (
                self.stats.get("quantifier_rules_sec", 0.0)
                + depth_metrics.get("quantifier_rules_sec", 0.0)
            )
            self.stats["generalization_sec"] = (
                self.stats.get("generalization_sec", 0.0)
                + depth_metrics.get("generalization_sec", 0.0)
            )

            logger.info(
                "  Depth %d: +%d schemas (total %d) [%.1fs]",
                depth,
                new_count,
                total,
                elapsed,
            )
            print(
                f"  Depth {depth}: +{new_count} schemas"
                f" (total {total}) [{elapsed:.1f}s]"
            )

            # Save checkpoint after each depth.
            # If ZFC_ASYNC_CHECKPOINT=1 is enabled, every Nth checkpoint
            # (controlled by ZFC_SYNC_CHECKPOINT_EVERY, default 2) is forced
            # to sync mode so we never lose more than N depths of work to a
            # mid-write SIGTERM.
            if checkpoint_dir is not None:
                self._update_stats()
                t_checkpoint = time.time()
                sync_every = int(os.environ.get("ZFC_SYNC_CHECKPOINT_EVERY", "2") or "0")
                async_enabled = self._effective_checkpoint_mode() == "async"
                force_sync = async_enabled and sync_every > 0 and (depth % sync_every == 0)
                if force_sync:
                    self._save_checkpoint_sync(checkpoint_dir)
                else:
                    self.save_checkpoint(checkpoint_dir)
                checkpoint_sec = time.time() - t_checkpoint
                depth_metrics["checkpoint_sec"] = round(checkpoint_sec, 6)
                self.stats["checkpoint_sec"] = (
                    self.stats.get("checkpoint_sec", 0.0) + checkpoint_sec
                )
                print(f"  Checkpoint saved to {checkpoint_dir}")
            else:
                depth_metrics["checkpoint_sec"] = 0.0

            depth_metrics["new_schemas"] = new_count
            depth_metrics["total_schemas"] = total
            self.stats.setdefault("timings_by_depth", {})[depth] = depth_metrics

            if (
                self.stop_on_watchlist
                and self.utility is not None
                and self.utility.all_hit()
            ):
                print(
                    f"  ★ Watchlist complete "
                    f"({self.utility.watchlist_size}/{self.utility.watchlist_size}) "
                    f"at depth {depth}. Stopping."
                )
                self.stats["stopped_reason"] = "watchlist_complete"
                break

            if self.beam_width is not None:
                # Beam-сатурация: нечего активировать И ничего не породилось.
                if activated == 0 and new_count == 0:
                    print(f"  Saturated (beam: empty pool) at depth {depth}.")
                    self.stats["stopped_reason"] = "saturated"
                    break
                if total >= self._schema_cap:
                    print(
                        f"  ⚠ TRUNCATED at schema limit ({self.max_schemas}) "
                        f"— NOT saturated."
                    )
                    self.stats["stopped_reason"] = "max_schemas"
                    break
                depth += 1
                continue

            if new_count == 0:
                if self.reuse_core and depth >= self.max_depth:
                    # NOT saturation: the last layer was tallied (frontier reuse),
                    # not materialized. The d-layer signal is in _frontier_outdeg.
                    print(f"  Reuse-core: depth {depth} frontier tallied "
                          f"({len(self._frontier_outdeg):,} core nodes), not materialized.")
                    self.stats["stopped_reason"] = "reuse_core_frontier_tallied"
                    self.stats["reuse_core_target_depth"] = depth
                    break
                logger.info("No new schemas at depth %d - saturated.", depth)
                print(f"  Saturated at depth {depth}.")
                self.stats["stopped_reason"] = "saturated"
                break
            if total >= self._schema_cap:
                logger.warning(
                    "Reached max_schemas=%s — graph TRUNCATED, not saturated. "
                    "Raise/clear max_schemas for a complete object.",
                    self.max_schemas,
                )
                print(f"  ⚠ TRUNCATED at schema limit ({self.max_schemas}) — NOT saturated.")
                self.stats["stopped_reason"] = "max_schemas"
                break

            depth += 1

        # C3 / I9 faithfulness gate: if the size safety-cap actually bound, the
        # object silently lost cut-detour theorems (крит §4) and is NOT a faithful
        # proof-length-d closure. Surface it loudly (strict mode already raised).
        size_dropped = self.stats.get("mp_filtered_size", 0)
        self.stats["size_cap_bound"] = bool(size_dropped)
        if size_dropped:
            logger.warning(
                "size-cap BOUND: %d formulas dropped (size > %d). The "
                "proof-length object is NOT faithful (cut-detours lost, крит §4). "
                "Raise max_formula_size or lower d; set ZFC_SIZE_CAP_STRICT=1 to fail hard.",
                size_dropped, self.max_formula_size,
            )
            print(f"  ⚠ SIZE-CAP BOUND: {size_dropped} formulas dropped "
                  f"(> {self.max_formula_size}) — object NOT faithful (крит §4).")

        elapsed_total = time.time() - t_start
        self._update_stats()
        self.stats["runtime_sec"] = self.stats.get("runtime_sec", 0) + round(elapsed_total, 2)

        logger.info(
            "Schema generation complete: %d schemas, max_depth=%d, %.1fs",
            len(self._entries),
            self.stats["max_depth_reached"],
            elapsed_total,
        )
        return self.graph

    def _effective_rule_merge_policy(self) -> str:
        """Return the rule-application order actually used by this generator."""

        if self._ordered_rule_application_reason == "online_subsumption":
            return "canonical_serial_online_subsumption"
        if self._ordered_rule_application_reason == "full_provenance":
            return "canonical_serial_full_provenance"
        if self.n_workers == 1:
            return "canonical_serial"
        return "bounded_parallel_arrival_order"

    def _update_stats(self) -> None:
        """Update aggregate stats from current state."""
        self.stats["total_schemas"] = len(self._entries)
        self.stats["vertices"] = len(self._entries)
        if self.reuse_core and self._frontier_outdeg:
            # Persist the un-materialized last-layer reuse signal: per materialized
            # node, its frontier reuse count. hub_score = projection out-degree +
            # this. Without it the d-layer signal is lost on save.
            id2hash = {e.id: h for h, e in self._entries.items()}
            self.stats["reuse_core_frontier_outdeg"] = {
                id2hash[pid]: c for pid, c in self._frontier_outdeg.items()
                if pid in id2hash
            }
            self.stats["reuse_core_frontier_edges"] = sum(self._frontier_outdeg.values())
        self.stats["hyperedges"] = self.graph.num_hyperedges()
        self.stats["projection_edges"] = self.graph.num_edges()
        self.stats["max_depth_reached"] = max(self._by_depth.keys()) if self._by_depth else 0
        self.stats["max_formula_size"] = self.max_formula_size
        self.stats["canonicalization"] = "structural_formula_form_v1"
        self.stats["graph_kind"] = self.graph_kind
        self.stats["projection_kind"] = "directed_2_section"
        self.stats["schema_instantiation_policy"] = self.schema_instantiation_policy
        self.stats["contains_schema_vertices"] = True
        self.stats["rule_merge_policy"] = self._effective_rule_merge_policy()
        self.stats.setdefault("mp_parallel_forced_serial_passes", 0)
        self.stats.setdefault("mp_parallel_forced_serial_subsumption_passes", 0)
        self.stats.setdefault("quantifier_parallel_forced_serial_passes", 0)
        self.stats.setdefault(
            "quantifier_parallel_forced_serial_subsumption_passes", 0
        )
        self.stats["mp_result_batch_size"] = self.mp_result_batch_size
        self.stats.update(provenance_metadata(self.provenance_mode))
        self.stats.setdefault("rediscovery_hyperedges_skipped", 0)
        self.stats["macro_rules"] = ["con3i"] if self.enable_con3i else []
        self.stats["convert_to_primitive"] = self.convert_to_primitive
        self.stats["axiom_form"] = (
            "primitive" if self.convert_to_primitive else "native"
        )

    # ── Checkpointing ─────────────────────────────────────────────────

    def _effective_checkpoint_mode(self) -> str:
        """Resolve the operational mode, allowing a launch-wrapper override."""
        return _checkpoint_mode_from_environment() or self.checkpoint_mode

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Save full generator state to a pickle file.

        If ``ZFC_CHECKPOINT_LOCAL`` env var is set to a directory, the
        checkpoint is first written to that local-disk path (typically
        node-local ``/tmp`` or ``$TMPDIR``), then moved to the shared
        target. This avoids slow gzip-over-NFS stalls that look like
        full-process freezes to HPC monitoring.

        Mode ``async`` (legacy ``ZFC_ASYNC_CHECKPOINT=1``) performs the write
        in a forked child while the parent continues.  Opt-in mode
        ``fork_sync`` (``ZFC_CHECKPOINT_MODE=fork_sync``,
        ``ZFC_FORK_SYNC_CHECKPOINT=1``, or ``checkpoint_mode='fork_sync'``)
        forks and waits.  In that mode the child calls ``os._exit`` immediately
        after the atomic commit, while the Pickler is still referenced by the
        serialization frame.  This prevents a very large Pickler memo from
        being destructed in the long-lived generator process.  Default
        ``sync`` behavior is unchanged.

        Args:
            path: Directory or file path. If directory, saves as
                  ``checkpoint_d{depth}.pkl.gz`` inside it.

        Returns:
            Path to the saved checkpoint file (in the original ``path``).
        """
        mode = self._effective_checkpoint_mode()
        if mode == "async":
            return self._save_checkpoint_async(path)
        if mode == "fork_sync":
            return self._save_checkpoint_fork_sync(path)
        return self._save_checkpoint_sync(path)

    def _resolve_checkpoint_target(self, path: Union[str, Path]) -> Path:
        """Resolve the public checkpoint path without serializing anything."""
        resolved = Path(path)
        if resolved.is_dir() or not resolved.suffix:
            resolved.mkdir(parents=True, exist_ok=True)
            depth = max(self._by_depth.keys()) if self._by_depth else 0
            resolved = resolved / f"checkpoint_d{depth}.pkl.gz"
        return resolved

    @staticmethod
    def _block_checkpoint_signals() -> Optional[Set[signal.Signals]]:
        """Close the fork/state-publication race in the parent when possible."""
        if not hasattr(signal, "pthread_sigmask"):
            return None
        blocked = {signal.SIGTERM}
        if hasattr(signal, "SIGUSR1"):
            blocked.add(signal.SIGUSR1)
        try:
            return signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _restore_signal_mask(mask: Optional[Set[signal.Signals]]) -> None:
        if mask is None or not hasattr(signal, "pthread_sigmask"):
            return
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        except (OSError, ValueError):
            pass

    def _wait_fork_sync_checkpoint(self) -> Path:
        """Wait for the active child and convert exit/signal status to errors."""
        pid = getattr(self, "_fork_sync_checkpoint_pid", None)
        resolved = getattr(self, "_fork_sync_checkpoint_path", None)
        read_fd = getattr(self, "_fork_sync_checkpoint_error_fd", None)
        if pid is None or resolved is None or read_fd is None:
            raise CheckpointChildError("no fork-synchronous checkpoint is active")

        status: Optional[int] = None
        try:
            while status is None:
                try:
                    waited_pid, waited_status = os.waitpid(pid, 0)
                except InterruptedError:
                    continue
                except ChildProcessError as exc:
                    raise CheckpointChildError(
                        f"checkpoint child {pid} was reaped unexpectedly"
                    ) from exc
                if waited_pid == pid:
                    status = waited_status

            chunks: List[bytes] = []
            while True:
                try:
                    chunk = os.read(read_fd, 4096)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                chunks.append(chunk)
            detail = b"".join(chunks).decode("utf-8", errors="replace").strip()
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass
            if getattr(self, "_fork_sync_checkpoint_pid", None) == pid:
                self._fork_sync_checkpoint_pid = None
                self._fork_sync_checkpoint_path = None
                self._fork_sync_checkpoint_error_fd = None

        assert status is not None
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            logger.info("Fork-synchronous checkpoint committed: %s", resolved)
            return resolved
        if os.WIFSIGNALED(status):
            signal_number = os.WTERMSIG(status)
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = str(signal_number)
            raise CheckpointChildError(
                f"checkpoint child {pid} terminated by signal {signal_name}"
            )
        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else None
        suffix = f": {detail}" if detail else ""
        raise CheckpointChildError(
            f"checkpoint child {pid} failed with exit code {exit_code}{suffix}"
        )

    def _save_checkpoint_fork_sync(self, path: Union[str, Path]) -> Path:
        """Fork, atomically save in the child, then synchronously reap it.

        The successful child never returns through ``_save_checkpoint_sync``:
        it exits from inside that serialization frame immediately after the
        final atomic replace.  CPython therefore does not walk and clear the
        Pickler memo (observed to take tens of minutes at 44+ GB footprints).

        A bounded pipe carries a traceback on ordinary child failure.  Exit by
        signal is reported from ``waitpid``.  SIGTERM/SIGUSR1 are blocked only
        across ``fork`` and publication of the active-child fields, preventing
        a graceful-shutdown handler from starting a second writer in that tiny
        race window.  A re-entrant save while the parent waits joins the active
        child instead of forking another one.
        """
        if not hasattr(os, "fork"):
            raise CheckpointChildError(
                "fork_sync checkpoint mode requires os.fork (POSIX)"
            )
        resolved = self._resolve_checkpoint_target(path)
        active_pid = getattr(self, "_fork_sync_checkpoint_pid", None)
        if active_pid is not None:
            active_path = getattr(self, "_fork_sync_checkpoint_path", None)
            if active_path != resolved:
                raise CheckpointChildError(
                    f"checkpoint child {active_pid} is already writing {active_path}; "
                    f"cannot concurrently write {resolved}"
                )
            return self._wait_fork_sync_checkpoint()

        # Do not duplicate buffered parent output into the child; the success
        # path flushes once immediately before os._exit.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        read_fd, write_fd = os.pipe()
        old_mask = self._block_checkpoint_signals()
        try:
            pid = os.fork()
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            self._restore_signal_mask(old_mask)
            raise

        if pid == 0:  # child
            try:
                os.close(read_fd)
                _reset_child_signal_handlers()
                self._restore_signal_mask(old_mask)
                self._save_checkpoint_sync(path, _exit_after_commit=True)
            except BaseException as exc:
                message = (
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                ).encode("utf-8", errors="replace")[:4096]
                try:
                    os.write(write_fd, message)
                except OSError:
                    pass
                os._exit(70)
            # The success path exits inside _save_checkpoint_sync.  Reaching
            # this line means that contract was broken.
            os._exit(71)

        # Parent publishes the child state while checkpoint-related signals
        # remain blocked, then restores the original mask before waiting.
        os.close(write_fd)
        self._fork_sync_checkpoint_pid = pid
        self._fork_sync_checkpoint_path = resolved
        self._fork_sync_checkpoint_error_fd = read_fd
        self._restore_signal_mask(old_mask)
        return self._wait_fork_sync_checkpoint()

    def _save_checkpoint_async(self, path: Union[str, Path]) -> Path:
        """Fork a child to handle pickle+gzip+write; parent returns immediately.

        Before forking, joins any previous async write so we never have two
        concurrent writes on the same target. After fork, parent does nothing
        and child performs the sync save then exits cleanly.
        """
        prev_pid = getattr(self, "_async_checkpoint_pid", None)
        if prev_pid is not None:
            try:
                os.waitpid(prev_pid, 0)
            except ChildProcessError:
                pass
            self._async_checkpoint_pid = None

        # Resolve target path now so we can return it before forking
        resolved = self._resolve_checkpoint_target(path)

        pid = os.fork()
        if pid == 0:
            # Child must not inherit the main process's SIGTERM handler, else a
            # preemption signal re-enters checkpoint logic recursively.
            _reset_child_signal_handlers()
            try:
                self._save_checkpoint_sync(path, _exit_after_commit=True)
            except BaseException as exc:  # pragma: no cover
                print(f"    Async checkpoint child error: {exc!r}", flush=True)
                os._exit(1)
            os._exit(0)

        self._async_checkpoint_pid = pid
        logger.info("Async checkpoint started: pid=%d -> %s", pid, resolved)
        return resolved

    def _save_checkpoint_sync(
        self,
        path: Union[str, Path],
        *,
        _exit_after_commit: bool = False,
    ) -> Path:
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            depth = max(self._by_depth.keys()) if self._by_depth else 0
            path = path / f"checkpoint_d{depth}.pkl.gz"

        path.parent.mkdir(parents=True, exist_ok=True)

        # Optional fast-local staging
        local_dir_env = os.environ.get("ZFC_CHECKPOINT_LOCAL", "").strip()
        if local_dir_env:
            # Namespace node-local staging by SLURM job / array-task (fallback
            # PID) so co-located jobs never collide on the SAME shared /tmp
            # filename. Without this, one job's os.replace() moves the .tmp out
            # from under another → FileNotFoundError (observed on cHARISMa with
            # multiple jobs/array-tasks sharing a node, all staging to /tmp).
            job_token = os.environ.get("SLURM_JOB_ID", "") or str(os.getpid())
            array_task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
            if array_task:
                job_token = f"{job_token}_{array_task}"
            local_dir = Path(local_dir_env) / f"zfc_stage_{job_token}"
            local_dir.mkdir(parents=True, exist_ok=True)
            stage_path = local_dir / path.name
        else:
            stage_path = path

        # Split state into config (small) and heavy data to reduce peak
        # memory during serialization.  Pickler streams each dump() call
        # individually, so the runtime only needs to hold one chunk at a time.
        config = {
            "logic": self.logic,
            "propositional_base": self.propositional_base,
            "theory": self.theory,
            "max_depth": self.max_depth,
            "max_schemas": self.max_schemas,
            "max_formula_size": self.max_formula_size,
            "n_workers": self.n_workers,
            "mp_parallel_threshold": self.mp_parallel_threshold,
            "mp_result_batch_size": self.mp_result_batch_size,
            "provenance_mode": self.provenance_mode,
            "subsumption_mode": self.subsumption_mode,
            "checkpoint_mode": self._effective_checkpoint_mode(),
            "subsumed_derivations": self._subsumed_derivations,
            "retired": self._retired,
            "enable_con3i": self.enable_con3i,
            "enable_necessitation": self.enable_necessitation,
            "enable_quantifier_rules": self.enable_quantifier_rules,
            "enable_generalization": self.enable_generalization,
            "zfc_include_choice": self.zfc_include_choice,
            "zfc_include_schema_instances": self.zfc_include_schema_instances,
            "sep_instance_limit": self.sep_instance_limit,
            "rep_instance_limit": self.rep_instance_limit,
            "use_pattern_unify": self.use_pattern_unify,
            "sep_enum_k": self.sep_enum_k,
            "size_cap_strict": self.size_cap_strict,
            "convert_to_primitive": self.convert_to_primitive,
            "next_id": self._next_id,
            "term_vars": self._term_vars,
            "term_consts": self._term_consts,
            "stats": self.stats,
        }

        # Atomic write: write to temp file, fsync, then rename.
        # For large graphs, gzip+pickle is single-threaded and can take many
        # minutes — emit per-section progress so HPC monitoring sees activity.
        # Optimization: gzip compresslevel=1 instead of default 9 (~5x faster
        # writes on shared filesystems, ~20% larger file). The graph IS pickled
        # in full because rediscovery edges (added by _add_schema on dedup hit)
        # are NOT recoverable from entries' premise_ids alone.
        n_entries = len(self._entries)
        n_edges = self.graph.num_edges() if hasattr(self.graph, "num_edges") else 0
        verbose = n_entries >= 100_000
        if verbose:
            print(
                f"    Checkpoint: writing {n_entries:,} entries, {n_edges:,} edges -> {path.name}",
                flush=True,
            )
        # Unique temp file per write. Under preemption several checkpoint
        # writers can run at once within one job (async fork-child, forced
        # sync, SIGTERM-handler save). A fixed "<name>.tmp" makes them race:
        # one writer's os.replace() moves the temp out from under another →
        # FileNotFoundError. mkstemp gives each writer a private path; the
        # final atomic replace onto the shared name is last-writer-wins and
        # never errors.
        import tempfile
        fd, _tmp_name = tempfile.mkstemp(
            dir=str(stage_path.parent),
            prefix=stage_path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(_tmp_name)
        serialization_complete = False
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            # The Pickler owns a very large, fully reachable memo while writing
            # depth-3 graphs. CPython 3.13 may otherwise start cyclic GC during
            # dump(), repeatedly traverse that memo, and turn an atomic save
            # into an hours-long pause. Reference counting remains active; no
            # object is released or omitted from the serialization.
            gc.disable()
        try:
            try:
                with gzip.open(os.fdopen(fd, "wb", closefd=False), "wb",
                               compresslevel=1) as gz:
                    p = pickle.Pickler(gz, protocol=pickle.HIGHEST_PROTOCOL)
                    t0 = time.time()
                    p.dump(config)
                    if verbose:
                        print(f"    Checkpoint: config dumped [{time.time()-t0:.0f}s]", flush=True)
                    t0 = time.time()
                    p.dump(self._by_depth)
                    if verbose:
                        print(f"    Checkpoint: by_depth dumped [{time.time()-t0:.0f}s]", flush=True)
                    t0 = time.time()
                    p.dump(self._entries)
                    if verbose:
                        print(f"    Checkpoint: entries dumped [{time.time()-t0:.0f}s]", flush=True)
                    t0 = time.time()
                    p.dump(self.graph)
                    if verbose:
                        print(f"    Checkpoint: graph dumped [{time.time()-t0:.0f}s]", flush=True)
                os.fsync(fd)
                serialization_complete = True
            finally:
                os.close(fd)
                if not serialization_complete:
                    tmp_path.unlink(missing_ok=True)
        finally:
            # fork_sync/async success exits later in this same frame via
            # os._exit. Re-enabling cyclic GC before the atomic replace lets
            # CPython 3.13 immediately traverse the still-live Pickler memo and
            # can delay publication for hours. The short-lived child must keep
            # GC disabled until that exit; an in-process save restores it.
            if gc_was_enabled and not _exit_after_commit:
                gc.enable()

        try:
            if stage_path != path:
                # Local staging done — move to node-local final name (atomic,
                # unique source → never FileNotFoundError), then copy to shared FS
                # through a private temp + atomic replace.
                t0 = time.time()
                os.replace(str(tmp_path), str(stage_path))
                import shutil
                fd2, _tmp2 = tempfile.mkstemp(
                    dir=str(path.parent), prefix=path.name + ".", suffix=".tmp2",
                )
                os.close(fd2)
                shared_tmp = Path(_tmp2)
                try:
                    shutil.copyfile(str(stage_path), shared_tmp)
                    os.replace(str(shared_tmp), str(path))
                finally:
                    shared_tmp.unlink(missing_ok=True)
                if verbose:
                    print(f"    Checkpoint: synced to shared FS [{time.time()-t0:.0f}s]",
                          flush=True)
            else:
                os.replace(str(tmp_path), str(path))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        if _exit_after_commit:
            # Deliberately exit *inside this frame*.  In CPython, local ``p``
            # still owns the Pickler memo here. Returning first would trigger
            # Pickler_dealloc/PyMemoTable_Clear in the child and can turn a
            # completed large checkpoint into tens of minutes of pure cleanup.
            # os._exit performs no Python-level finalization; the file has
            # already been closed, fsynced, and atomically committed above.
            os._exit(0)
        logger.info("Checkpoint saved: %s (%d schemas)", path, len(self._entries))
        return path

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "SchemaGenerator":
        """Load generator state from a checkpoint file.

        Args:
            path: Path to checkpoint file (``.pkl.gz``).
                  If a directory, loads the latest ``checkpoint_d*.pkl.gz``.

        Returns:
            SchemaGenerator with restored state, ready to call ``generate()``
            to continue from the next depth.
        """
        path = Path(path)

        # If directory, find latest checkpoint
        if path.is_dir():
            candidates = sorted(
                path.glob("checkpoint_d*.pkl.gz"),
                key=cls._checkpoint_depth,
            )
            if not candidates:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            path = candidates[-1]

        keep_load_state = (
            os.environ.get("ZFC_CHECKPOINT_LOAD_KEEP_GC_DISABLED", "").strip()
            == "1"
        )
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            # Large unpickles allocate millions of fully reachable objects.
            # CPython 3.13 may repeatedly scan them with cyclic GC while the
            # Unpickler is still filling its memo. Reference counting remains
            # active, so disabling cyclic collection does not alter the state.
            gc.disable()
        try:
            with gzip.open(path, "rb") as f:
                u = pickle.Unpickler(f)
                first = u.load()

                if isinstance(first, dict) and "entries" in first:
                    # Legacy single-dict format
                    state = first
                    config = state
                    by_depth = state["by_depth"]
                    entries = state["entries"]
                    graph = state.get("graph")
                else:
                    # Chunked format. May contain (config, by_depth, entries) — the
                    # current layout — or the legacy (config, by_depth, entries, graph).
                    config = first
                    by_depth = u.load()
                    entries = u.load()
                    try:
                        graph = u.load()
                    except EOFError:
                        graph = None  # graph not pickled — will be rebuilt below
            # On multi-million-object CPython 3.13 loads, destroying the
            # Unpickler memo can itself take hours. The opt-in large-load mode
            # retains it until process exit, where the OS reclaims it in O(1).
            if not keep_load_state:
                del u
        finally:
            if gc_was_enabled and not keep_load_state:
                gc.enable()

        if isinstance(config.get("stats"), dict) and config["stats"].get(
            "growth_policy"
        ):
            raise RuntimeError(
                "checkpoint was produced by directed growth (beam mode); "
                "v1 does not support resume — rerun from scratch"
            )

        gen = cls(
            logic=config["logic"],
            max_depth=config["max_depth"],
            max_schemas=config["max_schemas"],
            max_formula_size=config["max_formula_size"],
            enable_con3i=config["enable_con3i"],
            enable_necessitation=config["enable_necessitation"],
            enable_quantifier_rules=config["enable_quantifier_rules"],
            enable_generalization=config["enable_generalization"],
            zfc_include_choice=config["zfc_include_choice"],
            zfc_include_schema_instances=config["zfc_include_schema_instances"],
            sep_instance_limit=config.get("sep_instance_limit"),
            rep_instance_limit=config.get("rep_instance_limit"),
            propositional_base=config.get("propositional_base"),
            theory=config.get("theory"),
            convert_to_primitive=config.get("convert_to_primitive"),
            n_workers=config.get("n_workers", 1),
            use_pattern_unify=config.get("use_pattern_unify", False),
            sep_enum_k=config.get("sep_enum_k"),
            mp_parallel_threshold=config.get("mp_parallel_threshold", 50),
            mp_result_batch_size=config.get("mp_result_batch_size", 10_000),
            provenance_mode=config.get("provenance_mode", "full"),
            subsumption_mode=config.get("subsumption_mode", "off"),
            checkpoint_mode=config.get("checkpoint_mode", "sync"),
            size_cap_strict=config.get("size_cap_strict", False),
            _use_env_overrides=False,
        )

        gen._entries = entries
        gen._by_depth = by_depth
        gen._next_id = config["next_id"]
        gen._term_vars = config["term_vars"]
        gen._term_consts = config["term_consts"]
        gen.stats = config["stats"]
        gen._subsumed_derivations = config.get("subsumed_derivations", {}) or {}
        gen._retired = config.get("retired", {}) or {}
        if keep_load_state:
            gen._checkpoint_unpickler = u

        if graph is None:
            t0 = time.time()
            print(f"  Rebuilding graph from {len(entries):,} entries...", flush=True)
            gen.graph = gen._rebuild_graph_from_entries()
            print(f"  Graph rebuilt: {gen.graph.num_edges():,} edges "
                  f"[{time.time()-t0:.1f}s]", flush=True)
        else:
            gen.graph = graph
        gen.graph.graph_kind = gen.graph_kind
        gen.graph.projection_kind = "directed_2_section"
        if not hasattr(gen.graph, "metadata"):
            gen.graph.metadata = {}
        gen.graph.metadata.update({
            "canonicalization": "structural_formula_form_v1",
            "schema_instantiation_policy": gen.schema_instantiation_policy,
            "contains_schema_vertices": True,
            **provenance_metadata(gen.provenance_mode),
        })
        gen._rebuild_formula_intern_pool()
        gen._rebuild_transient_state()
        # Rebuild subsumption index from live (non-retired) entries.
        if gen._subsumption_index is not None:
            for entry in gen._entries.values():
                if entry.id in gen._retired:
                    continue
                gen._subsumption_index.insert(entry.id, entry.formula)

        depth = max(gen._by_depth.keys()) if gen._by_depth else 0
        logger.info(
            "Checkpoint loaded: %s (%d schemas, depth %d)",
            path, len(gen._entries), depth,
        )
        print(
            f"  Loaded checkpoint: {len(gen._entries)} schemas, "
            f"depth {depth}, from {path}"
        )
        return gen

    def _rebuild_graph_from_entries(self) -> "ProofGraph":
        """Reconstruct the ProofGraph from ``_entries`` after a checkpoint load.

        Re-creates nodes (one per entry) and edges (from each entry's
        ``premise_ids``). This is the inverse of skipping the graph in
        ``save_checkpoint`` and produces an equivalent graph structurally;
        only rule-multiplicity attributes from rediscovery edges are lost
        (they're not relevant for resume semantics — the BFS only writes,
        never reads, the ``rules`` edge list).
        """
        graph = ProofGraph(
            graph_kind=self.graph_kind,
            projection_kind="directed_2_section",
            metadata={
                "canonicalization": "structural_formula_form_v1",
                "schema_instantiation_policy": self.schema_instantiation_policy,
                "contains_schema_vertices": True,
                **provenance_metadata(self.provenance_mode),
            },
        )
        for entry in sorted(self._entries.values(), key=lambda e: e.id):
            node = ProofNode(
                id=entry.id,
                formula=entry.formula,
                normalized_hash=entry.schema_hash,
                depth=entry.depth,
                rule=entry.rule,
                premises=entry.premise_ids,
                label=entry.rule if entry.depth == 0 else None,
                is_axiom=(entry.depth == 0),
            )
            graph.add_node(node)
            if entry.premise_ids:
                premises = [pid for pid in entry.premise_ids if pid in graph]
                if not premises:
                    continue
                graph.add_hyperedge(
                    premises,
                    entry.id,
                    rule=entry.rule,
                    is_macro=(entry.rule == "con3i"),
                    source_schema=entry.rule,
                )
        return graph

    def _init_axioms(self) -> None:
        """Add depth-0 schema formulas.

        All axioms are normalize_schema'd here so that ``_add_schema`` can
        skip the redundant normalization step on its hot path.
        """
        for formula, rule_name in _make_axiom_schemas(self.propositional_base):
            self._add_schema(normalize_schema(formula), depth=0,
                             rule=rule_name, premise_ids=[])

        if self.propositional_base == "metamath":
            self._init_metamath_axioms()

        # C16: FOL-with-equality axioms for h16-family first-order theories.
        # `=` is a primitive predicate; without these it is uninterpreted
        # (x=x unprovable, equals not substitutable). metamath supplies these
        # via ax-7/8/9 (in _init_metamath_axioms), so it is excluded here.
        if self.propositional_base in ("h16", "h16_intuit", "h16_minimal") and \
                self.theory in {"zfc", "izf", "kp", "lomegaomega",
                                "zfc2", "msol", "henkin_sol"}:
            self._init_equality_axioms()

        # C15: intuitionistic/minimal connective axioms. These bases keep ∧,∨,↔
        # native (classical expansion is unsound in IPC), but had no rules for
        # them → inert connectives. Add the standard Heyting schemas.
        if self.propositional_base in ("h16_intuit", "h16_minimal"):
            self._init_intuitionistic_connective_axioms()

        primitive = self.convert_to_primitive
        if self.theory == "zfc":
            self._init_zfc_axioms(convert_to_primitive=primitive)
        elif self.theory == "izf":
            self._init_theory_axioms("izf", convert_to_primitive=primitive)
        elif self.theory == "kp":
            self._init_theory_axioms("kp", convert_to_primitive=primitive)
        elif self.theory == "zfc2":
            self._init_sol_axioms("zfc2", convert_to_primitive=primitive)
        elif self.theory == "henkin_sol":
            self._init_sol_axioms("henkin_sol", convert_to_primitive=primitive)
        elif self.theory == "msol":
            self._init_sol_axioms("msol", convert_to_primitive=primitive)
        elif self.theory == "lomegaomega":
            self._init_theory_axioms("lomegaomega", convert_to_primitive=primitive)

    def _init_intuitionistic_connective_axioms(self) -> None:
        """Add Heyting/IPC axioms for ∧, ∨, ↔ (C15) for intuit/minimal bases.

        Without these, native ∧/∨/↔ in the theory axioms are inert (e.g. A∧B ⊬
        A). All schemas below are valid in minimal logic (no ex falso, no LEM),
        so they serve both h16_intuit and h16_minimal. Classical h16 expands
        these connectives away (convert_to_primitive); metamath uses df-*
        bridges — hence this is gated to the intuit/minimal bases only.

        (Native existential ∃ is handled by ∃-intro/∃-elim, the duals of Q1/Q2,
        applied as quantifier rules — see `enable_exists_rules` and
        `_apply_quantifier_rules` (C15-∃ / I8).)
        """
        from core.formula import And, Or, Iff, Implies as _Imp
        x0, x1, x2 = PropVar("x0"), PropVar("x1"), PropVar("x2")
        conn_axioms = [
            ("AND-I",  _Imp(x0, _Imp(x1, And(x0, x1)))),
            ("AND-EL", _Imp(And(x0, x1), x0)),
            ("AND-ER", _Imp(And(x0, x1), x1)),
            ("OR-IL",  _Imp(x0, Or(x0, x1))),
            ("OR-IR",  _Imp(x1, Or(x0, x1))),
            ("OR-E",   _Imp(_Imp(x0, x2), _Imp(_Imp(x1, x2), _Imp(Or(x0, x1), x2)))),
            ("IFF-EL", _Imp(Iff(x0, x1), _Imp(x0, x1))),
            ("IFF-ER", _Imp(Iff(x0, x1), _Imp(x1, x0))),
            ("IFF-I",  _Imp(_Imp(x0, x1), _Imp(_Imp(x1, x0), Iff(x0, x1)))),
        ]
        # TEST (ZFC_NO_CONNECTIVE_INTRO): drop the connective-INTRODUCTION rules
        # (AND-I, OR-IL/IR, IFF-I) that BUILD top-level ∧/∨/↔ from facts. Hypothesis:
        # these are content-neutral machinery (the And-plateau), so removing them
        # collapses the graph to ~ZFC content facts without losing content.
        if os.environ.get("ZFC_NO_CONNECTIVE_INTRO"):
            _intro = {"AND-I", "OR-IL", "OR-IR", "IFF-I"}
            conn_axioms = [(n, a) for n, a in conn_axioms if n not in _intro]
        for name, axiom in conn_axioms:
            self._add_schema(normalize_schema(axiom), depth=0,
                             rule=name, premise_ids=[])

    def _init_equality_axioms(self) -> None:
        """Add FOL-with-equality axioms (C16) for h16-family FO theories.

        Mirrors set.mm ax-7/8/9 plus explicit reflexivity. With these,
        symmetry and transitivity are derivable (refl + sub-eq), and equals are
        substitutable into the only atomic predicates (= and ∈); substitution
        into compound formulas follows by induction over →,¬,∀. This is the
        standard sound+complete equality axiomatization for FOL over {∈,=}.
        Schemas use free FO variables (canonicalized to v_j), exactly like the
        metamath predicate axioms. Equality axioms use only →,=,∈, so they are
        intuitionistically valid and apply to h16_intuit / h16_minimal too.
        """
        from core.formula import make_variable, Equality, Membership, Implies as _Imp
        x = make_variable("x")
        y = make_variable("y")
        z = make_variable("z")
        eq_axioms = [
            ("EQ:refl", Equality(x, x)),                                  # x = x
            ("EQ:sub-eq",                                                 # ax-7
             _Imp(Equality(x, y), _Imp(Equality(x, z), Equality(y, z)))),
            ("EQ:sub-mem-l",                                              # ax-8
             _Imp(Equality(x, y), _Imp(Membership(x, z), Membership(y, z)))),
            ("EQ:sub-mem-r",                                              # ax-9
             _Imp(Equality(x, y), _Imp(Membership(z, x), Membership(z, y)))),
        ]
        for name, axiom in eq_axioms:
            self._add_schema(normalize_schema(axiom), depth=0,
                             rule=name, premise_ids=[])

    def _init_zfc_axioms(self, convert_to_primitive: bool) -> None:
        """Add ZFC axioms as depth-0 formulas.

        Args:
            convert_to_primitive: If True (H1-H6 mode), convert to {→, ¬, ∀}.
                If False (metamath mode), keep ∧, ∨, ↔, ∃ — definitional
                axioms df-* serve as bridges.
        """
        if self._sep_demand_mode:
            # §13: demand-driven — seed base ZFC + LITERAL Separation predicates
            # (k≤2: atoms + negations, incl Russell ¬(z∈z)); compound Separation
            # grows on demand from derived predicates (_apply_separation_demand).
            zfc_axioms = ZFCAxioms.get_all_axioms(
                include_choice=self.zfc_include_choice, include_schemas=False,
            )
            zfc_axioms.update(ZFCAxioms.get_separation_instances_upto_k(2))
            zfc_axioms.update(ZFCAxioms.get_common_schema_instances(
                sep_limit=0, rep_limit=None,
            ))
            # Mark the seeded literal predicates as already-instantiated so the
            # demand pass does not re-add them.
            from core.formula_enum import enumerate_predicates
            for phi in enumerate_predicates(2):
                self._sep_demand_seen.add(self._sep_pred_key(phi))
            self.stats["sep_mode"] = "demand"
        elif self.sep_enum_k is not None:
            # Option B: principled Separation = all φ of complexity ≤ k
            # plus the explicit Rep4 debt marker. Replacement enumeration over
            # ?F is a follow-up, but the current finite base must keep Rep4.
            zfc_axioms = ZFCAxioms.get_all_axioms(
                include_choice=self.zfc_include_choice, include_schemas=False,
            )
            zfc_axioms.update(ZFCAxioms.get_separation_instances_upto_k(self.sep_enum_k))
            zfc_axioms.update(ZFCAxioms.get_common_schema_instances(
                sep_limit=0,
                rep_limit=None,
            ))
            self.stats["sep_enum_k"] = self.sep_enum_k
        elif self.use_pattern_unify:
            # I7 (faithful): one schematic ?P-Separation replaces the enumerated
            # 12 instances; ?P is bound on demand by pattern-MP. Base axioms only,
            # no eager sep/rep enumeration. (Replacement schematic: TODO; eager
            # rep omitted in this mode pending its ?F-schema.)
            zfc_axioms = ZFCAxioms.get_all_axioms(
                include_choice=self.zfc_include_choice, include_schemas=False,
            )
            zfc_axioms["ax-sep-schema"] = ZFCAxioms.schematic_separation()
            zfc_axioms["ax-rep-schema"] = ZFCAxioms.schematic_replacement()
        else:
            zfc_axioms = ZFCAxioms.get_all_axioms(
                include_choice=self.zfc_include_choice,
                include_schemas=self.zfc_include_schema_instances,
                use_schema_instances=self.zfc_include_schema_instances,
                sep_instance_limit=self.sep_instance_limit,
                rep_instance_limit=self.rep_instance_limit,
            )
        for name, axiom in zfc_axioms.items():
            if convert_to_primitive:
                axiom = to_primitive_form(axiom)
            if self._add_schema(normalize_schema(axiom), depth=0,
                                rule=f"ZFC:{name}", premise_ids=[]):
                self.stats["zfc_axioms_added"] += 1

        # Baseline variable pool for quantifier rules.
        self._term_vars.update({"x", "y", "z"})
        self._candidate_terms_cache = None

    def _init_theory_axioms(self, theory: str, convert_to_primitive: bool) -> None:
        """Load axioms for non-ZFC theories (IZF, KP) from their modules."""
        if theory == "izf":
            from core.axioms_izf import IZFAxioms
            axioms_cls = IZFAxioms
        elif theory == "kp":
            from core.axioms_kp import KPAxioms
            axioms_cls = KPAxioms
        elif theory == "lomegaomega":
            from core.axioms_lomegaomega import LOmegaOmegaAxioms
            axioms_cls = LOmegaOmegaAxioms
        else:
            raise ValueError(f"Unknown theory: {theory}")

        axioms = axioms_cls.get_all_axioms(
            include_schemas=self.zfc_include_schema_instances,
        )
        for name, axiom in axioms.items():
            if convert_to_primitive:
                axiom = to_primitive_form(axiom)
            if self._add_schema(normalize_schema(axiom), depth=0,
                                rule=f"{theory.upper()}:{name}", premise_ids=[]):
                self.stats["zfc_axioms_added"] += 1
        self._term_vars.update({"x", "y", "z"})
        self._candidate_terms_cache = None

    def _init_sol_axioms(self, theory: str, convert_to_primitive: bool) -> None:
        """Load axioms for second-order theories (ZFC₂, Henkin SOL, MSOL)."""
        from core.axioms_zfc2 import ZFC2Axioms, HenkinSOLAxioms, MSOLAxioms

        if theory == "zfc2":
            axioms_cls = ZFC2Axioms
        elif theory == "henkin_sol":
            axioms_cls = HenkinSOLAxioms
        elif theory == "msol":
            axioms_cls = MSOLAxioms
        else:
            raise ValueError(f"Unknown SOL theory: {theory}")

        axioms = axioms_cls.get_all_axioms(
            include_choice=self.zfc_include_choice,
            include_schemas=self.zfc_include_schema_instances,
        )
        tag = theory.upper()
        for name, axiom in axioms.items():
            if convert_to_primitive:
                axiom = to_primitive_form(axiom)
            if self._add_schema(normalize_schema(axiom), depth=0,
                                rule=f"{tag}:{name}", premise_ids=[]):
                self.stats["zfc_axioms_added"] += 1
        self._term_vars.update({"x", "y", "z"})
        self._candidate_terms_cache = None

    def _init_metamath_axioms(self) -> None:
        """Add Metamath predicate calculus and definitional axioms."""
        for formula, rule_name in _make_metamath_predicate_axioms():
            self._add_schema(normalize_schema(formula), depth=0,
                             rule=rule_name, premise_ids=[])

        for formula, rule_name in _make_metamath_definitional_axioms():
            self._add_schema(normalize_schema(formula), depth=0,
                             rule=rule_name, premise_ids=[])

    def _mp_pass(
        self,
        *,
        minors: List[SchemaEntry],
        depth: int,
        all_candidates: List[MPCandidate],
        wildcard_candidates: List[MPCandidate],
        candidates_by_tag: Dict[int, List[MPCandidate]],
        metrics: Dict[str, Any],
    ) -> int:
        """Run one MP pass over the given minors, optionally in parallel.

        Without online subsumption, serial and parallel branches produce the
        same formula set (the master remains the sole graph writer). Online
        subsumption is order-sensitive, so it always uses the canonical serial
        traversal. This gives ``n_workers > 1`` the same scientific graph and
        full provenance as ``n_workers == 1`` without an unbounded reorder
        buffer at large depths.
        """
        if not minors:
            return 0

        # Reuse-core frontier: detachable wildcard majors (P→ψ, P∉ψ) produce the
        # SAME ψ for every minor. Don't fan them out across all minors (millions
        # of identical-ψ emissions + IPC) — credit each such major's single ψ
        # ONCE here, and remove them from the candidate pool so workers skip them.
        # The per-minor credits are uniform across minors (every minor matches a
        # wildcard) → skipping them is rank-preserving for the hub signature.
        if self.reuse_core and depth > 0 and depth >= self.max_depth:
            detachables = [c for c in wildcard_candidates
                           if c.detached_result is not None]
            if detachables:
                for c in detachables:
                    # one ψ per detachable major → credit its reuse once
                    self._frontier_outdeg[c.entry.id] = \
                        self._frontier_outdeg.get(c.entry.id, 0) + 1
                wildcard_candidates = [c for c in wildcard_candidates
                                       if c.detached_result is None]
                all_candidates = [c for c in all_candidates
                                  if c.detached_result is None]

        # ── Exact bulk path for detachable wildcard majors ──────────────────
        # A vacuous-antecedent wildcard major (P→ψ with P∉ψ) detaches to the
        # SAME canonical ψ for EVERY minor. In non-"full" provenance the repeat
        # firings add no graph edge (only a rediscovery counter), and
        # node.premises already records a single discovery — so materializing ψ
        # ONCE and tallying the (k−1) repeats is bit-exact for the graph (nodes,
        # edges, premises) while skipping the O(minors×detach) fan and its IPC
        # (≈97–99% of d≥2 wildcard work). Gated to configs where a vacuous
        # firing's only per-call effect is node / dup / size / prune — no
        # per-minor, premise-keyed side effects (reuse_core, content_space,
        # beam rediscovery, subsumption, and "full" provenance are excluded).
        if (
            os.environ.get("ZFC_BULK_DETACH", "1") != "0"
            and self.provenance_mode != "full"
            and self.beam_width is None
            and not self._reuse_core
            and not self._content_space
            and self._subsumption_index is None
        ):
            detachables = [c for c in wildcard_candidates
                           if c.detached_result is not None]
            if detachables:
                nminors = len(minors)
                tail = nminors - 1
                rep_id = minors[0].id
                _CTR = ("mp_filtered_dup", "rediscovery_hyperedges_skipped",
                        "mp_filtered_size", "pruned_propositional")
                for c in detachables:
                    if len(self._entries) >= self._schema_cap:
                        break
                    result, det_hash = c.detached_result
                    pre = {k: self.stats.get(k, 0) for k in _CTR}
                    is_new = self._add_schema(
                        result, depth, "MP", [rep_id, c.entry.id],
                        precomputed_hash=det_hash,
                    )
                    # every minor fires this vacuous major exactly once
                    self.stats["mp_successes"] += nminors
                    self.stats["bulk_detach_firings"] = (
                        self.stats.get("bulk_detach_firings", 0) + nminors)
                    if is_new:
                        new_count += 1
                        if tail:
                            # the (k−1) repeats are rediscoveries of this ψ
                            self.stats["mp_filtered_dup"] = (
                                self.stats.get("mp_filtered_dup", 0) + tail)
                            self.stats["rediscovery_hyperedges_skipped"] = (
                                self.stats.get("rediscovery_hyperedges_skipped", 0)
                                + tail)
                    elif tail:
                        # representative was rejected (dup/size/prune); the
                        # repeats reject identically — replay its counter delta.
                        for k in _CTR:
                            d = self.stats.get(k, 0) - pre[k]
                            if d:
                                self.stats[k] = self.stats.get(k, 0) + d * tail
                wildcard_candidates = [c for c in wildcard_candidates
                                       if c.detached_result is None]
                all_candidates = [c for c in all_candidates
                                  if c.detached_result is None]

        # Serial fallback: small batches, single worker, empty candidate pool,
        # or an order-sensitive online-subsumption configuration. Parallel
        # workers stream result batches by completion time; applying those
        # batches directly would make acceptance/retirement scheduler-dependent.
        parallel_requested = (
            self.n_workers > 1
            and len(minors) >= self.mp_parallel_threshold
            and (len(all_candidates) + len(wildcard_candidates)) > 0
        )
        if parallel_requested and self._ordered_rule_application_required:
            self.stats["mp_parallel_forced_serial_passes"] = (
                self.stats.get("mp_parallel_forced_serial_passes", 0) + 1
            )
            metrics["mp_parallel_forced_serial_passes"] = (
                metrics.get("mp_parallel_forced_serial_passes", 0) + 1
            )
            if self._ordered_rule_application_reason == "online_subsumption":
                self.stats["mp_parallel_forced_serial_subsumption_passes"] = (
                    self.stats.get(
                        "mp_parallel_forced_serial_subsumption_passes", 0
                    )
                    + 1
                )
                metrics["mp_parallel_forced_serial_subsumption_passes"] = (
                    metrics.get(
                        "mp_parallel_forced_serial_subsumption_passes", 0
                    )
                    + 1
                )
            if self.stats["mp_parallel_forced_serial_passes"] == 1:
                print(
                    "    MP deterministic gate: "
                    f"{self._ordered_rule_application_reason} forces canonical "
                    f"serial merge (requested workers={self.n_workers})",
                    flush=True,
                )
        use_parallel = (
            parallel_requested and not self._ordered_rule_application_required
        )

        new_count = 0

        # Heartbeat cadence: emit a progress line every N processed pairs/results.
        # Keeps HPC monitoring from flagging the master process as idle during
        # long sequential phases (master-side dedup with millions of MP hits).
        HEARTBEAT_PAIRS = 500_000

        if not use_parallel:
            processed_pairs = 0
            t_hb = time.time()
            for minor_idx, minor in enumerate(minors):
                if len(self._entries) >= self._schema_cap:
                    return new_count
                candidate_lists = self._candidate_lists_for_minor(
                    minor.formula,
                    all_candidates=all_candidates,
                    wildcard_candidates=wildcard_candidates,
                    candidates_by_tag=candidates_by_tag,
                )
                metrics["mp_candidate_pairs_filtered"] += self._count_candidate_lists(
                    candidate_lists
                )
                for bucket in candidate_lists:
                    for major in bucket:
                        det = major.detached_result
                        if det is not None:
                            # Vacuous-antecedent wildcard major: precomputed ψ,
                            # pass its hash so _add_schema skips canonicalize.
                            result, det_hash = det
                        else:
                            # Serial path: schema_mp skips its canonicalize because
                            # _add_schema below runs canonicalize_with_hash anyway.
                            result = schema_mp(
                                minor.formula,
                                major.entry.formula,
                                major_vars=major.major_vars,
                                antecedent_tag=major.antecedent_tag,
                                antecedent_is_wildcard=major.antecedent_is_wildcard,
                                inputs_canonical=True,
                                dvar_constraints=list(major.dvar_constraints),
                                canonicalize_output=False,
                                use_pattern_unify=self.use_pattern_unify,
                            )
                            det_hash = None
                        processed_pairs += 1
                        if processed_pairs % HEARTBEAT_PAIRS == 0:
                            print(
                                f"    MP serial: {processed_pairs:,} pairs, "
                                f"+{new_count} new, minor {minor_idx+1}/{len(minors)} "
                                f"[{time.time()-t_hb:.0f}s since last]",
                                flush=True,
                            )
                            t_hb = time.time()
                        if result is not None:
                            if self._add_schema(result, depth, "MP", [minor.id, major.entry.id],
                                                precomputed_hash=det_hash):
                                new_count += 1
                            self.stats["mp_successes"] += 1
            return new_count

        # Parallel pass: split minors into chunks, process via Pool, merge sequentially
        n_workers = self.n_workers
        # Aim for ~4 chunks per worker for load balancing
        target_chunks = max(n_workers * 4, 1)
        chunk_size = max(1, (len(minors) + target_chunks - 1) // target_chunks)
        # Build minor payload + filtered-pair metric in one pass.
        minor_payload: List[Tuple[int, Formula]] = []
        for m in minors:
            minor_payload.append((m.id, m.formula))
            candidate_lists = self._candidate_lists_for_minor(
                m.formula,
                all_candidates=all_candidates,
                wildcard_candidates=wildcard_candidates,
                candidates_by_tag=candidates_by_tag,
            )
            metrics["mp_candidate_pairs_filtered"] += self._count_candidate_lists(
                candidate_lists
            )
        chunks = [
            minor_payload[i : i + chunk_size]
            for i in range(0, len(minor_payload), chunk_size)
        ]
        print(
            f"    MP parallel: {len(minors):,} minors x ~{(len(all_candidates)+len(wildcard_candidates)):,} majors, "
            f"{n_workers} workers, {len(chunks)} chunks",
            flush=True,
        )

        ctx = _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else _mp.get_context()
        merged = 0
        chunks_done = 0
        t_hb = time.time()
        task_queue = ctx.Queue()
        # Bound the result queue so workers cannot accumulate unbounded MP hits
        # in master-side IPC buffers while dedup/provenance is serialized.
        result_queue = ctx.Queue(maxsize=max(4, n_workers * 2))
        workers = [
            ctx.Process(
                target=_mp_worker_stream_chunks,
                args=(
                    task_queue,
                    result_queue,
                    all_candidates,
                    wildcard_candidates,
                    candidates_by_tag,
                    self.mp_result_batch_size,
                    self.use_pattern_unify,
                ),
            )
            for _ in range(n_workers)
        ]

        for proc in workers:
            proc.start()

        # Merge one batch into the shared graph. Returns True if the schema cap
        # was hit (caller must stop). Shared by the parallel-drain loop and the
        # serial-fallback path so both behave identically.
        def _merge_batch(batch) -> bool:
            nonlocal new_count, merged, t_hb
            for item in batch:
                if len(self._entries) >= self._schema_cap:
                    return True
                minor_id, major_id, result, precomputed_hash = item
                if self._add_schema(
                    result, depth, "MP", [minor_id, major_id],
                    precomputed_hash=precomputed_hash,
                ):
                    new_count += 1
                merged += 1
                if merged % HEARTBEAT_PAIRS == 0:
                    print(
                        f"    MP merge: {merged:,} results processed, "
                        f"+{new_count} new, chunk {chunks_done}/{len(chunks)} "
                        f"[{time.time()-t_hb:.0f}s since last]",
                        flush=True,
                    )
                    t_hb = time.time()
            return False

        # Watchdog: a forked worker can wedge (PyPy+fork Queue lock, OOM-stall on
        # a co-located node) while staying alive (exitcode=None), so the dead-worker
        # check never fires and the master spins on get(timeout) forever. If no
        # message arrives for STALL_SECS, abandon the workers and finish the
        # unfinished chunks serially in the master — the pass can never hang.
        STALL_SECS = int(os.environ.get("ZFC_MP_STALL_SECS", "300") or "300")
        last_progress = time.time()
        done_chunks: Set[int] = set()
        stalled = False
        cap_hit = False
        try:
            for chunk_index, chunk in enumerate(chunks):
                task_queue.put((chunk_index, chunk))
            for _ in workers:
                task_queue.put(None)

            workers_done = 0
            while workers_done < len(workers):
                try:
                    message, _chunk_index, payload = result_queue.get(timeout=5)
                except queue.Empty:
                    failed = [
                        proc for proc in workers
                        if proc.exitcode not in (None, 0)
                    ]
                    if failed:
                        for proc in workers:
                            proc.terminate()
                        exitcodes = [proc.exitcode for proc in failed]
                        raise RuntimeError(
                            f"MP worker exited without a result message: {exitcodes}"
                        )
                    if time.time() - last_progress > STALL_SECS:
                        alive = [(p.pid, p.is_alive()) for p in workers]
                        print(
                            f"    !! MP STALL: no worker progress for {STALL_SECS}s "
                            f"(workers={alive}, chunks {len(done_chunks)}/{len(chunks)} "
                            f"done) -> abandoning workers, finishing remainder serially.",
                            flush=True,
                        )
                        for proc in workers:
                            proc.terminate()
                        stalled = True
                        break
                    continue
                last_progress = time.time()
                if message == "results":
                    batch, raw_success_count = payload
                    # Raw success count includes worker-side dedup drops, so
                    # mp_successes matches a non-pre-dedup run.
                    self.stats["mp_successes"] += raw_success_count
                    if _merge_batch(batch):
                        cap_hit = True
                        for proc in workers:
                            proc.terminate()
                        break
                elif message == "chunk_done":
                    chunks_done += 1
                    done_chunks.add(_chunk_index)
                elif message == "worker_done":
                    workers_done += 1
                elif message == "error":
                    for proc in workers:
                        proc.terminate()
                    exc_repr, tb = payload
                    raise RuntimeError(f"MP worker failed: {exc_repr}\n{tb}")
                else:
                    for proc in workers:
                        proc.terminate()
                    raise RuntimeError(f"Unknown MP worker message: {message!r}")
        finally:
            for proc in workers:
                proc.join(timeout=1)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            task_queue.close()
            result_queue.close()

        if cap_hit:
            return new_count

        if stalled:
            # Re-run the chunks that never reported chunk_done, in this process.
            # Chunks that streamed partial batches before the stall get re-merged;
            # _add_schema dedups materialized nodes, so only frontier reuse tallies
            # for those few chunks may be marginally inflated (logged below).
            remaining = [chunks[i] for i in range(len(chunks)) if i not in done_chunks]
            print(
                f"    MP serial-fallback: replaying {len(remaining)}/{len(chunks)} "
                f"unfinished chunks in master.",
                flush=True,
            )
            _mp_worker_init(
                all_candidates, wildcard_candidates, candidates_by_tag,
                self.use_pattern_unify,
            )
            for chunk in remaining:
                for batch, raw_count in _mp_worker_result_batches(
                    chunk, self.mp_result_batch_size
                ):
                    self.stats["mp_successes"] += raw_count
                    if _merge_batch(batch):
                        return new_count

        return new_count

    def _generate_depth(self, depth: int) -> Tuple[int, Dict[str, Any]]:
        """Generate new schemas at a depth via MP and enabled side rules."""
        new_count = 0
        metrics: Dict[str, Any] = {
            "mp_candidate_pairs_raw": 0,
            "mp_candidate_pairs_filtered": 0,
            "mp_pass_sec": 0.0,
            "quantifier_rules_sec": 0.0,
            "generalization_sec": 0.0,
        }

        prev_depth = list(self._by_depth.get(depth - 1, []))
        all_implications = list(self._all_implication_candidates)
        all_implications_by_tag = {
            tag: list(bucket)
            for tag, bucket in self._implication_candidates_by_tag.items()
        }
        wildcard_implications = list(self._wildcard_implication_candidates)

        print(
            f"  Depth {depth} starting: prev_depth={len(prev_depth):,}, "
            f"all_schemas={len(self._entries):,}, "
            f"implications={len(all_implications):,} "
            f"(wildcards={len(wildcard_implications):,})",
            flush=True,
        )

        prev_implications: List[MPCandidate] = []
        prev_implications_by_tag: Dict[int, List[MPCandidate]] = {}
        prev_wildcard_implications: List[MPCandidate] = []
        for entry in prev_depth:
            candidate = self._implication_candidate_by_id.get(entry.id)
            if candidate is None:
                continue
            prev_implications.append(candidate)
            if candidate.antecedent_is_wildcard:
                prev_wildcard_implications.append(candidate)
            else:
                prev_implications_by_tag.setdefault(
                    candidate.antecedent_tag, []
                ).append(candidate)

        # older_schemas = all entries with depth < depth-1; built directly
        # from _by_depth instead of filtering a full copy of _entries.values().
        older_schemas: List[SchemaEntry] = []
        for d in range(depth - 1):
            bucket = self._by_depth.get(d)
            if bucket:
                older_schemas.extend(bucket)
        metrics["mp_candidate_pairs_raw"] += len(prev_depth) * len(all_implications)
        metrics["mp_candidate_pairs_raw"] += len(older_schemas) * len(prev_implications)

        t_mp = time.time()

        # Pass 1: minor in prev_depth, major implication anywhere.
        new_count += self._mp_pass(
            minors=prev_depth,
            depth=depth,
            all_candidates=all_implications,
            wildcard_candidates=wildcard_implications,
            candidates_by_tag=all_implications_by_tag,
            metrics=metrics,
        )
        if len(self._entries) >= self._schema_cap:
            metrics["mp_pass_sec"] = round(time.time() - t_mp, 6)
            return new_count, metrics

        # Pass 2: old minor, major in prev_depth implications.
        new_count += self._mp_pass(
            minors=older_schemas,
            depth=depth,
            all_candidates=prev_implications,
            wildcard_candidates=prev_wildcard_implications,
            candidates_by_tag=prev_implications_by_tag,
            metrics=metrics,
        )
        if len(self._entries) >= self._schema_cap:
            metrics["mp_pass_sec"] = round(time.time() - t_mp, 6)
            return new_count, metrics

        metrics["mp_pass_sec"] = round(time.time() - t_mp, 6)

        # con3i: from (phi->psi) derive (~psi->~phi)
        if self.enable_con3i:
            t_con3i = time.time()
            con3i_added = 0
            HB_CON = max(50_000, len(prev_depth) // 10) if prev_depth else 0
            for idx, s in enumerate(prev_depth):
                if len(self._entries) >= self._schema_cap:
                    return new_count, metrics
                if type(s.formula) is Implies:
                    result = normalize_schema(
                        Implies(Not(s.formula.consequent), Not(s.formula.antecedent))
                    )
                    if self._add_schema(result, depth, "con3i", [s.id]):
                        new_count += 1
                        con3i_added += 1
                        self.stats["con3i_added"] += 1
                if HB_CON and (idx + 1) % HB_CON == 0:
                    print(f"    con3i: {idx+1:,}/{len(prev_depth):,} prev entries, "
                          f"+{con3i_added} new [{time.time()-t_con3i:.0f}s]", flush=True)
            print(f"    con3i done: +{con3i_added} [{time.time()-t_con3i:.0f}s]", flush=True)

        # Necessitation: from phi derive box(phi)
        if self.enable_necessitation:
            for s in prev_depth:
                if len(self._entries) >= self._schema_cap:
                    return new_count, metrics
                result = normalize_schema(Box(s.formula))
                if self._add_schema(result, depth, "Nec", [s.id]):
                    new_count += 1
                    self.stats["nec_added"] += 1

        if self.enable_quantifier_rules:
            t_quantifier = time.time()
            n_q = self._apply_quantifier_rules(prev_depth, depth)
            new_count += n_q
            metrics["quantifier_rules_sec"] = round(time.time() - t_quantifier, 6)
            print(f"    Q1/Q2 done: +{n_q} [{time.time()-t_quantifier:.0f}s]", flush=True)
            if len(self._entries) >= self._schema_cap:
                return new_count, metrics

        if self.enable_generalization:
            t_generalization = time.time()
            n_g = self._apply_generalization(prev_depth, depth)
            new_count += n_g
            metrics["generalization_sec"] = round(
                time.time() - t_generalization, 6
            )
            print(f"    Gen done: +{n_g} [{time.time()-t_generalization:.0f}s]", flush=True)

        # BigAnd projection: from ⋀(φ₁,...,φₙ) derive each φᵢ
        if self.theory == "lomegaomega":
            t_ba = time.time()
            n_b = self._apply_bigand_projection(prev_depth, depth)
            new_count += n_b
            print(f"    BigAnd-proj done: +{n_b} [{time.time()-t_ba:.0f}s]", flush=True)

        # SOL predicate rules: Pred-Inst and Pred-Gen
        if self.enable_pred_rules:
            t_pi = time.time()
            n_pi = self._apply_pred_inst(prev_depth, depth)
            new_count += n_pi
            print(f"    Pred-Inst done: +{n_pi} [{time.time()-t_pi:.0f}s]", flush=True)
            if len(self._entries) >= self._schema_cap:
                return new_count, metrics
            t_pg = time.time()
            n_pg = self._apply_pred_gen(prev_depth, depth)
            new_count += n_pg
            print(f"    Pred-Gen done: +{n_pg} [{time.time()-t_pg:.0f}s]", flush=True)

        # §13: demand-driven Separation — instantiate Sep on the unary predicates
        # this depth's schemas CONSTRUCTED (generative comprehension, not eager enum).
        if self._sep_demand_mode:
            t_sd = time.time()
            n_sd = self._apply_separation_demand(prev_depth, depth)
            new_count += n_sd
            print(f"    Sep-demand done: +{n_sd} "
                  f"(distinct preds so far {len(self._sep_demand_seen):,}) "
                  f"[{time.time()-t_sd:.0f}s]", flush=True)

        return new_count, metrics

    def _sep_pred_key(self, phi: Formula) -> str:
        """Canonical key of a comprehension predicate (dedup up to renaming)."""
        return canonicalize_with_hash(phi)[1]

    def _harvest_predicates(self, formula: Formula) -> List[Formula]:
        """§13.5: collect unary first-order comprehension predicates the schema
        CONSTRUCTED — ZFC-content subformulas with a free FO variable and no
        propositional metavariable. The chosen free var is renamed to `z` so the
        result is a φ(z) ready for `separation_instance`. Deduped per call."""
        out: List[Formula] = []
        local_seen: set = set()
        stack = [formula]
        while stack:
            f = stack.pop()
            for attr in ("antecedent", "consequent", "left", "right",
                         "inner", "body", "operand"):
                child = getattr(f, attr, None)
                # only descend into Formula children (Terms lack bound_variables)
                if child is not None and hasattr(child, "bound_variables"):
                    stack.append(child)
            sub = getattr(f, "formulas", None)
            if sub:
                stack.extend(sub)
            # candidate predicate: carries ZFC content, no metavar, has a free FO var
            if _is_pure_propositional(f) or _contains_propvar(f):
                continue
            fv = sorted(f.free_variables())
            if not fv or "z" in f.bound_variables():
                continue
            # every free FO var is a valid comprehension variable (the rest are
            # parameters) → yield one predicate φ(z) per choice of distinguished var.
            for z in fv:
                phi = f if z == "z" else f.substitute(z, make_variable("z"))
                k = str(phi)
                if k in local_seen:
                    continue
                local_seen.add(k)
                out.append(phi)
        return out

    def _apply_separation_demand(self, prev_depth: List[SchemaEntry], depth: int) -> int:
        """§13: for each NEW unary predicate constructed at this depth, add the
        Separation instance `∀x∃y∀z(z∈y ↔ (z∈x ∧ φ(z)))` with a provenance edge
        from the schema that built the predicate. Weight-bounded; canonical dedup."""
        from core.formula_enum import complexity
        added = 0
        for s in prev_depth:
            if len(self._entries) >= self._schema_cap:
                break
            for phi in self._harvest_predicates(s.formula):
                if complexity(phi) > self._sep_pred_maxsize:
                    continue
                key = self._sep_pred_key(phi)
                if key in self._sep_demand_seen:
                    continue
                self._sep_demand_seen.add(key)
                axiom = ZFCAxioms.separation_instance(phi, "z")
                if self.convert_to_primitive:
                    axiom = to_primitive_form(axiom)
                if self._add_schema(normalize_schema(axiom), depth,
                                    "Sep-demand", [s.id]):
                    added += 1
                    self._sep_demand_added += 1
        return added

    def _apply_quantifier_rules(self, prev_depth: List[SchemaEntry], depth: int) -> int:
        """Generate Q1/Q2 axiom-schema instances as implication vertices.

        Each instance is linked back to the originating ∀x.φ via
        ``premise_ids=[entry.id]`` so the graph reflects the structural
        dependency: Q1/Q2 instances are *parametric* axiom-templates
        synthesised from the universal hypothesis above them.

        - Q1-inst: ∀x.φ(x) → φ(t)  (for each suitable term t)
        - Q2-inst: ∀x.(φ → ψ) → (φ → ∀x.ψ)  (when x ∉ FV(φ))

        MP then closes the gap by combining the instance with ∀x.φ.
        """
        # Parallel fast path: large prev_depth and >1 worker. As with MP,
        # online subsumption preserves the serial entry order because every
        # accepted instance immediately changes the index for later instances.
        parallel_requested = (
            self.n_workers > 1
            and len(prev_depth) >= self.mp_parallel_threshold
        )
        if parallel_requested and not self._ordered_rule_application_required:
            return self._apply_quantifier_rules_parallel(prev_depth, depth)
        if parallel_requested and self._ordered_rule_application_required:
            self.stats["quantifier_parallel_forced_serial_passes"] = (
                self.stats.get("quantifier_parallel_forced_serial_passes", 0) + 1
            )
            if self._ordered_rule_application_reason == "online_subsumption":
                self.stats[
                    "quantifier_parallel_forced_serial_subsumption_passes"
                ] = (
                    self.stats.get(
                        "quantifier_parallel_forced_serial_subsumption_passes", 0
                    )
                    + 1
                )

        count = 0
        terms = self._candidate_terms()
        t0 = time.time()
        HB = max(20_000, len(prev_depth) // 10) if prev_depth else 0

        for idx, entry in enumerate(prev_depth):
            if len(self._entries) >= self._schema_cap:
                return count
            formula = entry.formula

            # Q1-inst: from ∀x.φ(x), generate implication ∀x.φ(x) → φ(t)
            if type(formula) is ForAll:
                for term in terms:
                    if not is_free_for(term, formula.variable, formula.body):
                        continue
                    substituted = normalize_schema(
                        formula.body.substitute(formula.variable, term)
                    )
                    q1_impl = normalize_schema(Implies(formula, substituted))
                    if self._add_schema(q1_impl, depth, "Q1-inst", [entry.id]):
                        count += 1
                        self.stats["q1_added"] += 1
                    if len(self._entries) >= self._schema_cap:
                        return count

            # Q2-inst: from ∀x.(φ→ψ), generate ∀x.(φ→ψ) → (φ→∀x.ψ)
            # Side condition: x ∉ FV(φ).
            #
            # Soundness fix: skip when α contains any PropVar metavariable.
            # Reason: is_not_free_in(var, PropVar) trivially returns True
            # because PropVars carry no FO free variables, but a future MP
            # substitution can bind that PropVar to a formula containing var
            # free, producing an unsound Q2 instance. The conservative skip
            # only excludes schema-level Q2 instances whose validity depends
            # on a side condition we cannot verify until instantiation. (Under
            # canonical naming, vertices' free FO vars are v_j and bound vars
            # are b_k, so no concrete FOL Q2 instance is lost in the primary
            # pipeline; only the metavariable case is filtered.)
            if type(formula) is ForAll and type(formula.body) is Implies:
                alpha = formula.body.antecedent
                beta = formula.body.consequent
                var = formula.variable
                if is_not_free_in(var, alpha) and not _contains_propvar(alpha):
                    consequent = normalize_schema(Implies(alpha, ForAll(var, beta)))
                    q2_impl = normalize_schema(Implies(formula, consequent))
                    if self._add_schema(q2_impl, depth, "Q2-inst", [entry.id]):
                        count += 1
                        self.stats["q2_added"] += 1
                    if len(self._entries) >= self._schema_cap:
                        return count

            # C15-∃ (I8): native existential rules for intuit/minimal logic.
            if self.enable_exists_rules:
                # ∃-intro: from ∃x.φ, generate φ[t/x] → ∃x.φ  (dual of Q1).
                if type(formula) is Exists:
                    for term in terms:
                        if not is_free_for(term, formula.variable, formula.body):
                            continue
                        substituted = normalize_schema(
                            formula.body.substitute(formula.variable, term)
                        )
                        ei_impl = normalize_schema(Implies(substituted, formula))
                        if self._add_schema(ei_impl, depth, "Exists-Intro", [entry.id]):
                            count += 1
                            self.stats["exists_intro_added"] += 1
                        if len(self._entries) >= self._schema_cap:
                            return count

                # ∃-elim: from ∀x.(φ→ψ), generate ∀x.(φ→ψ) → (∃x.φ → ψ),
                # side condition x ∉ FV(ψ)  (dual of Q2). The same PropVar guard
                # as Q2 applies — to the consequent ψ here — because a later MP
                # could bind a PropVar ψ to a formula with x free, breaking the
                # side condition we cannot verify until instantiation.
                if type(formula) is ForAll and type(formula.body) is Implies:
                    alpha = formula.body.antecedent
                    beta = formula.body.consequent
                    var = formula.variable
                    if is_not_free_in(var, beta) and not _contains_propvar(beta):
                        consequent = normalize_schema(
                            Implies(Exists(var, alpha), beta))
                        ee_impl = normalize_schema(Implies(formula, consequent))
                        if self._add_schema(ee_impl, depth, "Exists-Elim", [entry.id]):
                            count += 1
                            self.stats["exists_elim_added"] += 1
                        if len(self._entries) >= self._schema_cap:
                            return count

            if HB and (idx + 1) % HB == 0:
                print(f"    Q1/Q2: {idx+1:,}/{len(prev_depth):,} entries, "
                      f"+{count} so far [{time.time()-t0:.0f}s]", flush=True)

        return count

    def _apply_quantifier_rules_parallel(
        self, prev_depth: List[SchemaEntry], depth: int
    ) -> int:
        """Parallel Q1/Q2 generation: workers compute candidate instances,
        master applies _add_schema sequentially to preserve single-writer dedup.
        """
        count = 0
        terms = self._candidate_terms()
        n_workers = self.n_workers
        target_chunks = max(n_workers * 4, 1)
        chunk_size = max(1, (len(prev_depth) + target_chunks - 1) // target_chunks)
        payload = [(e.id, e.formula) for e in prev_depth]
        chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]
        t0 = time.time()
        print(
            f"    Q1/Q2 parallel: {len(prev_depth):,} entries x {len(terms):,} terms, "
            f"{n_workers} workers, {len(chunks)} chunks",
            flush=True,
        )

        ctx = _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else _mp.get_context()
        with ctx.Pool(n_workers, initializer=_reset_child_signal_handlers) as pool:
            for batch in pool.imap_unordered(
                _q12_worker_wrapper,
                [(c, terms, self.enable_exists_rules) for c in chunks]
            ):
                for formula, entry_id, rule in batch:
                    if len(self._entries) >= self._schema_cap:
                        pool.terminate()
                        return count
                    if self._add_schema(formula, depth, rule, [entry_id]):
                        count += 1
                        _stat = {
                            "Q1-inst": "q1_added", "Q2-inst": "q2_added",
                            "Exists-Intro": "exists_intro_added",
                            "Exists-Elim": "exists_elim_added",
                        }[rule]
                        self.stats[_stat] += 1
        print(f"    Q1/Q2 parallel done: +{count} [{time.time()-t0:.0f}s]", flush=True)
        return count

    def _apply_generalization(self, prev_depth: List[SchemaEntry], depth: int) -> int:
        """Apply Gen: from φ infer ∀x.φ where x ∈ FV(φ).

        Restricts the binder variable to free variables of φ. Vacuous
        bindings ∀x.φ with x ∉ FV(φ) are logically equivalent to φ
        and only inflate the graph with structurally-distinct duplicates.
        """
        count = 0
        t0 = time.time()
        HB = max(20_000, len(prev_depth) // 10) if prev_depth else 0
        for idx, entry in enumerate(prev_depth):
            if len(self._entries) >= self._schema_cap:
                return count
            free_vars = sorted(entry.formula.free_variables())
            for var in free_vars:
                result = normalize_schema(ForAll(var, entry.formula))
                if self._add_schema(result, depth, "Gen", [entry.id]):
                    count += 1
                    self.stats["gen_added"] += 1
                if len(self._entries) >= self._schema_cap:
                    return count
            if HB and (idx + 1) % HB == 0:
                print(f"    Gen: {idx+1:,}/{len(prev_depth):,} entries, "
                      f"+{count} so far [{time.time()-t0:.0f}s]", flush=True)
        return count

    def _apply_bigand_projection(self, prev_depth: List[SchemaEntry], depth: int) -> int:
        """BigAnd projection: from ⋀(φ₁,...,φₙ) derive each φᵢ.

        This is the L_{ω₁,ω} elimination rule for countable conjunctions.
        """
        from core.formula import BigAnd as _BigAnd
        count = 0
        for entry in prev_depth:
            if len(self._entries) >= self._schema_cap:
                return count
            if type(entry.formula) is _BigAnd:
                for sub in entry.formula.formulas:
                    result = normalize_schema(sub)
                    if self._add_schema(result, depth, "BigAnd-proj", [entry.id]):
                        count += 1
                    if len(self._entries) >= self._schema_cap:
                        return count
        return count

    def _apply_pred_inst(self, prev_depth: List[SchemaEntry], depth: int) -> int:
        """Pred-Inst: from ∀P^n.φ(P) derive ∀P^n.φ(P) → φ[P := concrete].

        Generates implication vertices (like Q1-inst) so MP can close the gap.
        For each ForAllPred formula, instantiate P with concrete predicate templates.
        """
        count = 0
        for entry in prev_depth:
            if len(self._entries) >= self._schema_cap:
                return count
            formula = entry.formula
            if type(formula) is not ForAllPred:
                continue

            pred_name = formula.pred_var
            arity = formula.arity
            templates = self._candidate_predicates(arity)

            for param_vars, template_body in templates:
                # Substitute PredApp(pred_name, args) → template_body[params := args]
                instantiated = _subst_pred_app(
                    pred_name, param_vars, template_body, formula.body
                )
                instantiated = normalize_schema(instantiated)
                # Generate implication: ∀P^n.φ(P) → φ[P:=concrete]
                impl = normalize_schema(Implies(formula, instantiated))
                if self._add_schema(impl, depth, "Pred-Inst", [entry.id]):
                    count += 1
                    self.stats["pred_inst_added"] += 1
                if len(self._entries) >= self._schema_cap:
                    return count
        return count

    def _apply_pred_gen(self, prev_depth: List[SchemaEntry], depth: int) -> int:
        """Pred-Gen: from φ derive ∀P^n.φ when P does not occur in φ.

        Analogous to Gen but for second-order variables.
        Only adds ∀P^n.φ for predicate names already seen in the graph.
        """
        count = 0
        # Collect pred var names from SOL axioms
        pred_names = set()
        for entry in self._entries.values():
            _collect_pred_var_names(entry.formula, pred_names)
        if not pred_names:
            return count

        for entry in prev_depth:
            if len(self._entries) >= self._schema_cap:
                return count
            formula = entry.formula
            # Check that the formula doesn't already contain the pred var
            for pname, parity in sorted(pred_names):
                if _contains_pred_ref(pname, formula):
                    continue
                result = normalize_schema(ForAllPred(pname, parity, formula))
                if self._add_schema(result, depth, "Pred-Gen", [entry.id]):
                    count += 1
                    self.stats["pred_gen_added"] += 1
                if len(self._entries) >= self._schema_cap:
                    return count
        return count

    def _candidate_predicates(self, arity: int) -> List[Tuple[List[str], Formula]]:
        """Return concrete predicate templates for Pred-Inst.

        Each template is (param_var_names, body) where body uses param vars.
        For arity 1: P(z) := z∈a, z=a, ¬(z∈a)
        For arity 2: F(w,z) := w=z (identity), z=a (constant)
        """
        if arity == 1:
            pv = make_variable("_pi0")
            av = make_variable("a")
            return [
                (["_pi0"], Membership(pv, av)),      # P(z) := z ∈ a
                (["_pi0"], Equality(pv, av)),         # P(z) := z = a
                (["_pi0"], Not(Membership(pv, av))),  # P(z) := ¬(z ∈ a)
            ]
        elif arity == 2:
            pv0 = make_variable("_pi0")
            pv1 = make_variable("_pi1")
            av = make_variable("a")
            return [
                (["_pi0", "_pi1"], Equality(pv0, pv1)),  # F(w,z) := w = z
                (["_pi0", "_pi1"], Equality(pv1, av)),    # F(w,z) := z = a
            ]
        else:
            return []

    def _candidate_terms(self) -> List[Term]:
        """Finite term pool for Q1 instantiations.

        Includes both variables and constants observed in the graph so far.
        Standard Q1 (∀x.φ → φ[t/x]) admits any term t free for x; the
        side-condition `is_free_for` filters unsound substitutions in
        ``_apply_quantifier_rules``.

        Cached: rebuilds only when ``_term_vars`` / ``_term_consts`` change
        (cache invalidated by ``_add_schema`` when new symbols are observed).
        """
        if self._candidate_terms_cache is not None:
            return self._candidate_terms_cache

        terms: List[Term] = [
            Term(name=v, is_variable=True) for v in sorted(self._term_vars)
        ]
        terms.extend(
            Term(name=c, is_variable=False) for c in sorted(self._term_consts)
        )

        if not terms:
            terms = [Term(name="x", is_variable=True), Term(name="y", is_variable=True)]
        self._candidate_terms_cache = terms
        return terms

    def _entry_by_id(self, node_id: int) -> Optional[SchemaEntry]:
        """Reverse lookup id → SchemaEntry, O(1) via `_id_to_entry` (C6).

        Retired entries stay in `_entries` (non-destructive retirement), so
        they remain resolvable here — matching the previous linear-scan
        behaviour exactly.
        """
        return self._id_to_entry.get(node_id)

    def _record_subsumed_derivation(
        self,
        *,
        subsumer_entry: SchemaEntry,
        dropped_formula: Formula,
        rule: str,
        premise_ids: List[int],
        depth: int,
    ) -> None:
        """Persist enough provenance to reconstruct the dropped ψ.

        Stores ``SubsumedRecord(parents, rule, sigma, depth)`` against the
        surviving subsumer. ``sigma`` is the substitution returned by
        :func:`schema_match` — the only datum not derivable from the
        subsumer's canonical formula alone.
        """
        sigma = schema_match(subsumer_entry.formula, dropped_formula) or {}
        sigma_items: Tuple[Tuple[str, Formula], ...] = tuple(
            sorted(sigma.items(), key=lambda kv: kv[0])
        )
        record = SubsumedRecord(
            parents=tuple(premise_ids),
            rule=rule,
            sigma=sigma_items,
            depth=depth,
        )
        self._subsumed_derivations.setdefault(subsumer_entry.id, []).append(record)

    def _retire_subsumed_by(self, subsumer_entry: SchemaEntry) -> None:
        """Mark every existing entry that the new subsumer ⊒-dominates.

        Retirement is non-destructive: ids stay in ``self._entries`` and the
        hyperedges remain in the graph for provenance. The retired flag is
        consulted by BFS to skip further rule application on the retired
        entry, and downstream tools can render the subsumption edge.
        """
        if self._subsumption_index is None:
            return
        for nid in self._subsumption_index.find_subsumed_by(subsumer_entry.formula):
            if nid == subsumer_entry.id:
                continue
            if nid in self._retired:
                continue
            retired_entry = self._entry_by_id(nid)
            if retired_entry is None:
                continue
            self._retired[nid] = subsumer_entry.id
            # Record a provenance entry so the original derivation is
            # recoverable: σ(subsumer) = retired_entry.formula.
            self._record_subsumed_derivation(
                subsumer_entry=subsumer_entry,
                dropped_formula=retired_entry.formula,
                rule=retired_entry.rule,
                premise_ids=retired_entry.premise_ids,
                depth=retired_entry.depth,
            )
            self.stats["backward_retired"] = (
                self.stats.get("backward_retired", 0) + 1
            )
            # Drop from the subsumption index so future queries don't re-hit it.
            self._subsumption_index.remove(nid)

    def _add_schema(
        self,
        formula: Formula,
        depth: int,
        rule: str,
        premise_ids: List[int],
        *,
        precomputed_hash: Optional[str] = None,
    ) -> bool:
        """Add a schema if new and within size bound.

        The formula is canonicalized here, so deduplication, graph storage, and
        MP indexing use one structural vertex policy.

        If ``precomputed_hash`` is provided (e.g. by an MP worker that already
        canonicalized the formula), the canonicalization step is skipped and
        the supplied hash is used directly.
        """
        # Reuse-core mode (§2.13): the LAST layer is ~87–99% inert frontier
        # leaves (out-degree 0) that carry no hub signal. Don't materialize them
        # as nodes — just COUNT premise reuse (O(#nodes) memory, not an O(#edges)
        # set). Detachable wildcards are emitted once upstream, so non-detachable
        # firings give a distinct ψ per minor → count ≈ distinct out-degree.
        # No canonicalize needed (counter keyed by premise id).
        if self._reuse_core and depth > 0 and depth >= self.max_depth:
            for pid in premise_ids:
                self._frontier_outdeg[pid] = self._frontier_outdeg.get(pid, 0) + 1
            return False

        if precomputed_hash is None:
            # Fused canonicalize + structural-hash: one AST traversal instead of two.
            formula, h = canonicalize_with_hash(formula)
        else:
            h = precomputed_hash

        # Depth-0 axioms define the base theory and are not filtered by the
        # experiment's growth bound.
        if depth > 0 and formula.size() > self.max_formula_size:
            self.stats["mp_filtered_size"] += 1
            if self.size_cap_strict:
                raise RuntimeError(
                    f"size-cap bound at depth {depth}: formula size "
                    f"{formula.size()} > max_formula_size {self.max_formula_size}. "
                    "The proof-length-d object is no longer faithful (cut-detour "
                    "theorems dropped, крит §4). Raise max_formula_size or lower d."
                )
            return False

        # ZFC-relevance pruning: drop pure-propositional DERIVED schemas so they
        # don't breed further (the dominant explosion). Depth-0 axioms (incl. the
        # propositional H-axioms, kept as MP rules) are exempt.
        if depth > 0 and self.prune_propositional and _is_pure_propositional(formula):
            self.stats["pruned_propositional"] = (
                self.stats.get("pruned_propositional", 0) + 1
            )
            return False

        # Content-space (ZFC_CONTENT_SPACE): do NOT materialize a top-level
        # conjunction node. `A∧B` is ∧-intro of facts that are already present
        # (its content = the conjuncts, kept as separate nodes); the §13 gate
        # showed content never depends on derived conjunctions beyond their parts.
        # This removes the ∧-bundling explosion (≈96.7% of ZFC nodes) natively,
        # leaving the irreducible-ZFC-fact network (logic = d0 rules = the parameter).
        if (self._content_space and depth > 0
                and type(formula).__name__ in ("And", "Conjunction")):
            for pid in premise_ids:
                self._frontier_outdeg[pid] = self._frontier_outdeg.get(pid, 0) + 1
            self.stats["content_space_dropped_conj"] = (
                self.stats.get("content_space_dropped_conj", 0) + 1
            )
            return False

        # ── Forward subsumption check ────────────────────────────────
        # If an existing entry strictly subsumes the incoming formula, drop
        # the incoming and record provenance against the subsumer. The
        # canonical-hash dedup below would only catch syntactic equality;
        # subsumption catches the more-general structural case.
        if self._subsumption_index is not None and h not in self._entries:
            subsumer_id = self._subsumption_index.find_subsumer(formula)
            if subsumer_id is not None:
                subsumer_entry = self._entry_by_id(subsumer_id)
                if subsumer_entry is not None and subsumer_entry.schema_hash != h:
                    self._record_subsumed_derivation(
                        subsumer_entry=subsumer_entry,
                        dropped_formula=formula,
                        rule=rule,
                        premise_ids=premise_ids,
                        depth=depth,
                    )
                    self.stats["subsumed_dropped"] = (
                        self.stats.get("subsumed_dropped", 0) + 1
                    )
                    return False

        existing = self._entries.get(h)
        if existing is not None:
            if self.beam_width is not None:
                # Сигнал «естественной сходимости» для U(φ): сколько раз
                # независимые выводы переоткрыли эту схему.
                self._rediscovery_counts[existing.id] = (
                    self._rediscovery_counts.get(existing.id, 0) + 1
                )
                self._on_rediscovery_event(existing, premise_ids)
            projection_premises = [pid for pid in premise_ids if pid != existing.id]
            if premise_ids and self.provenance_mode == "full":
                self.graph.add_hyperedge(
                    premise_ids,
                    existing.id,
                    rule=rule,
                    is_macro=(rule == "con3i"),
                    source_schema=rule,
                    approximation_metadata={"rediscovery": True},
                    projection_premises=projection_premises,
                )
            elif premise_ids:
                self.stats["rediscovery_hyperedges_skipped"] = (
                    self.stats.get("rediscovery_hyperedges_skipped", 0) + 1
                )
            self.stats["mp_filtered_dup"] += 1
            return False

        formula = self._intern_formula(h, formula)
        node_id = self._next_id
        self._next_id += 1

        entry = SchemaEntry(
            id=node_id,
            formula=formula,
            schema_hash=h,
            depth=depth,
            rule=rule,
            premise_ids=premise_ids,
        )
        self._entries[h] = entry
        self._id_to_entry[node_id] = entry  # C6: maintain O(1) id→entry index

        # Watchlist-прогресс (вехи к цели; активен при заданном ranker'е).
        if self.utility is not None:
            hit_name = self.utility.check_hit(h)
            if hit_name is not None:
                self.stats.setdefault("watchlist_hits", {})[hit_name] = depth
                print(
                    f"    ★ watchlist hit: {hit_name} at depth {depth} "
                    f"({len(self.utility.hits)}/{self.utility.watchlist_size})",
                    flush=True,
                )

        if self.beam_width is not None and depth > 0:
            # Directed growth: схема создана (узел/гиперребро/субсумпция —
            # выше), но как посылка будущих шагов ждёт активации по U(φ).
            self._passive_pool[node_id] = entry
        else:
            self._by_depth.setdefault(depth, []).append(entry)
            self._register_entry(entry)

        node = ProofNode(
            id=node_id,
            formula=formula,
            normalized_hash=h,
            depth=depth,
            rule=rule,
            premises=premise_ids,
            label=rule if depth == 0 else None,
            is_axiom=(depth == 0),
        )
        self.graph.add_node(node)
        if premise_ids:
            self.graph.add_hyperedge(
                premise_ids,
                node_id,
                rule=rule,
                is_macro=(rule == "con3i"),
                source_schema=rule,
                approximation_metadata={"rediscovery": False},
            )

        if self.theory in {"zfc", "izf", "kp", "lomegaomega", "zfc2", "msol", "henkin_sol"}:
            var_names: Set[str] = set()
            const_names: Set[str] = set()
            _collect_term_symbols(formula, var_names, const_names)
            new_vars = var_names - self._term_vars
            new_consts = const_names - self._term_consts
            if new_vars:
                self._term_vars.update(new_vars)
            if new_consts:
                self._term_consts.update(new_consts)
            if new_vars or new_consts:
                self._candidate_terms_cache = None

        # ── Subsumption index: register this entry, optionally retire others.
        if self._subsumption_index is not None:
            self._subsumption_index.insert(node_id, formula)
            if self.subsumption_mode == "full":
                self._retire_subsumed_by(entry)

        return True
