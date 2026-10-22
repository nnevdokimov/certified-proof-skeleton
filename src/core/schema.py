"""
Schema-level operations for Hilbert proof graphs.

In the schema graph, PropVars are metavariables that range over
arbitrary formulas. Two formulas that differ only in PropVar naming
are the same schema: p->(q->p) == r->(s->r) == x0->(x1->x0).

Core operations:
  - normalize_schema: canonical PropVar renaming (DFS first-appearance)
  - apply_subst: apply {var -> formula} substitution
  - unify: most general unifier (Martelli-Montanari)
  - rename_apart: rename PropVars to avoid clashes before unification
  - schema_hash: canonical hash after normalization
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Set, Tuple

from .formula import (
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
)

# Cache type objects to avoid repeated attribute lookups in hot loops.
_PropVar = PropVar
_Implies = Implies
_And = And
_Or = Or
_Iff = Iff
_Not = Not
_Box = Box
_ForAll = ForAll
_Exists = Exists
_Membership = Membership
_Equality = Equality
_PredicateVar = PredicateVar
_PredApp = PredApp
_ForAllPred = ForAllPred
_ExistsPred = ExistsPred
_BigAnd = BigAnd
_BigOr = BigOr

# Types with .left/.right binary structure.
_BINARY_TYPES = (_And, _Or, _Iff)
# Types with .inner unary structure.
_UNARY_TYPES = (_Not, _Box)
# Quantifier types with .variable/.body.
_QUANT_TYPES = (_ForAll, _Exists)
# Second-order quantifier types with .pred_var/.arity/.body.
_QUANT_PRED_TYPES = (_ForAllPred, _ExistsPred)
# Atomic FO types (no PropVars inside).
_ATOMIC_TYPES = (_Membership, _Equality)
# Infinitary types with .formulas tuple.
_BIG_TYPES = (_BigAnd, _BigOr)


# ═══════════════════════════════════════════════════════════════════════════
#  Variable collection (with caching)
# ═══════════════════════════════════════════════════════════════════════════

# Per-formula caches using WeakValueDictionary won't work with frozen
# dataclasses, so we use a bounded LRU-style cache keyed by (id, hash).
# The (id, hash) pair avoids stale entries from recycled ids.
_vars_cache: Dict[Tuple[int, int], frozenset] = {}
_vars_ordered_cache: Dict[Tuple[int, int], tuple] = {}
_CACHE_MAX = 500_000


def invalidate_caches() -> None:
    """Clear formula-level caches (call between runs if needed)."""
    _vars_cache.clear()
    _vars_ordered_cache.clear()


def _cache_key(formula: Formula) -> Tuple[int, int]:
    return (id(formula), hash(formula))


def collect_vars(formula: Formula) -> Set[str]:
    """Collect all PropVar names (as a set)."""
    key = _cache_key(formula)
    cached = _vars_cache.get(key)
    if cached is not None:
        return set(cached)
    result: Set[str] = set()
    _collect_vars_rec(formula, result)
    if len(_vars_cache) < _CACHE_MAX:
        _vars_cache[key] = frozenset(result)
    return result


def _collect_vars_rec(f: Formula, acc: Set[str]) -> None:
    t = type(f)
    if t is _PropVar:
        acc.add(f.name)
    elif t is _Implies:
        _collect_vars_rec(f.antecedent, acc)
        _collect_vars_rec(f.consequent, acc)
    elif t in (_And, _Or, _Iff):
        _collect_vars_rec(f.left, acc)
        _collect_vars_rec(f.right, acc)
    elif t is _Not or t is _Box:
        _collect_vars_rec(f.inner, acc)
    elif t is _ForAll or t is _Exists:
        _collect_vars_rec(f.body, acc)
    elif t is _ForAllPred or t is _ExistsPred:
        _collect_vars_rec(f.body, acc)
    elif t in (_BigAnd, _BigOr):
        for sub in f.formulas:
            _collect_vars_rec(sub, acc)
    # PredicateVar and PredApp contain no PropVars


def collect_vars_ordered(formula: Formula) -> List[str]:
    """Collect PropVar names in DFS first-appearance order."""
    key = _cache_key(formula)
    cached = _vars_ordered_cache.get(key)
    if cached is not None:
        return list(cached)
    seen: Set[str] = set()
    order: List[str] = []
    _collect_vars_ordered_rec(formula, seen, order)
    if len(_vars_ordered_cache) < _CACHE_MAX:
        _vars_ordered_cache[key] = tuple(order)
    return order


def _collect_vars_ordered_rec(f: Formula, seen: Set[str], order: List[str]) -> None:
    t = type(f)
    if t is _PropVar:
        if f.name not in seen:
            seen.add(f.name)
            order.append(f.name)
    elif t is _Implies:
        _collect_vars_ordered_rec(f.antecedent, seen, order)
        _collect_vars_ordered_rec(f.consequent, seen, order)
    elif t in (_And, _Or, _Iff):
        _collect_vars_ordered_rec(f.left, seen, order)
        _collect_vars_ordered_rec(f.right, seen, order)
    elif t is _Not or t is _Box:
        _collect_vars_ordered_rec(f.inner, seen, order)
    elif t is _ForAll or t is _Exists:
        _collect_vars_ordered_rec(f.body, seen, order)
    elif t is _ForAllPred or t is _ExistsPred:
        _collect_vars_ordered_rec(f.body, seen, order)
    elif t in (_BigAnd, _BigOr):
        for sub in f.formulas:
            _collect_vars_ordered_rec(sub, seen, order)
    # PredicateVar and PredApp contain no PropVars


# ═══════════════════════════════════════════════════════════════════════════
#  Substitution
# ═══════════════════════════════════════════════════════════════════════════

def apply_subst(subst: Dict[str, Formula], formula: Formula) -> Formula:
    """Apply substitution {var_name → formula} to a formula."""
    t = type(formula)
    if t is _PropVar:
        return subst.get(formula.name, formula)
    if t is _Implies:
        return Implies(
            apply_subst(subst, formula.antecedent),
            apply_subst(subst, formula.consequent),
        )
    if t in (_And, _Or, _Iff):
        return t(
            apply_subst(subst, formula.left),
            apply_subst(subst, formula.right),
        )
    if t is _Not:
        return Not(apply_subst(subst, formula.inner))
    if t is _Box:
        return Box(apply_subst(subst, formula.inner))
    if t is _ForAll or t is _Exists:
        return t(formula.variable, apply_subst(subst, formula.body))
    if t is _ForAllPred or t is _ExistsPred:
        return t(formula.pred_var, formula.arity, apply_subst(subst, formula.body))
    if t in (_BigAnd, _BigOr):
        return t(tuple(apply_subst(subst, f) for f in formula.formulas))
    # Membership, Equality, PredicateVar, PredApp, or unknown — no PropVars inside
    return formula


# ═══════════════════════════════════════════════════════════════════════════
#  Canonical normalization
# ═══════════════════════════════════════════════════════════════════════════

# AC-idempotent normalization of ∧/∨ (semantic quotient, opt-in via
# ZFC_AC_NORMALIZE=1). Collapses associativity/commutativity/idempotence of
# conjunction & disjunction (A∧A→A, A∧B=B∧A, (A∧B)∧C=A∧B∧C) — sound in
# classical/intuitionistic/minimal logic — so propositional-packaging
# duplicates (the §2.8 ∧-intro junk) merge into one node. Read at import; set
# the env BEFORE importing this module (true for HPC/subprocess A/B).
_AC_NORMALIZE = os.environ.get("ZFC_AC_NORMALIZE", "").strip() == "1"


def _flatten_ac(f: Formula, cls, out: List[Formula]) -> None:
    """Collect AC-normalized operands of a maximal ``cls`` (And/Or) chain."""
    if type(f) is cls:
        _flatten_ac(f.left, cls, out)
        _flatten_ac(f.right, cls, out)
    else:
        out.append(_ac_normalize(f))


def _ac_normalize(f: Formula) -> Formula:
    """Recursively normalize ∧/∨ by flatten + dedup (idempotence) + sort
    (commutativity), rebuilt left-assoc. Other constructors are recursed
    structurally; SOL binders / atoms are returned unchanged."""
    t = type(f)
    if t is And or t is Or:
        ops: List[Formula] = []
        _flatten_ac(f, t, ops)
        seen: Dict[str, Formula] = {}
        for op in ops:
            seen.setdefault(_structural_string(op), op)
        operands = [seen[k] for k in sorted(seen)]
        if len(operands) == 1:
            return operands[0]  # A∧A → A
        acc = operands[0]
        for op in operands[1:]:
            acc = t(acc, op)
        return acc
    if t is Implies:
        return Implies(_ac_normalize(f.antecedent), _ac_normalize(f.consequent))
    if t is Iff:
        return Iff(_ac_normalize(f.left), _ac_normalize(f.right))
    if t is Not:
        return Not(_ac_normalize(f.inner))
    if t is Box:
        return Box(_ac_normalize(f.inner))
    if t is ForAll:
        return ForAll(f.variable, _ac_normalize(f.body))
    if t is Exists:
        return Exists(f.variable, _ac_normalize(f.body))
    return f  # atoms, PropVar, SOL binders (Срез 1) — untouched


def canonicalize_formula(formula: Formula) -> Formula:
    """Return the structural canonical representative for a graph vertex.

    This is a syntactic quotient by renaming only:
      * schema PropVars are renamed by first appearance to x0, x1, ...
      * free first-order variables are renamed by first appearance to v0, v1, ...
      * bound first-order variables are alpha-normalized by scope
      * predicate variables are renamed by binding/first appearance to P0, P1, ...

    Constants, constructor order, predicate argument order, and connective order
    are preserved. No semantic laws such as commutativity, associativity,
    equality symmetry, or logical equivalence are applied.
    """
    if _AC_NORMALIZE:
        formula = _ac_normalize(formula)
    state = {
        "prop_map": {},
        "free_var_map": {},
        "pred_map": {},
        "bound_counter": 0,
        "prop_counter": 0,
        "free_var_counter": 0,
        "pred_counter": 0,
    }
    return _canonicalize_rec(formula, fo_env={}, pred_env={}, state=state)


def _canonical_term(term: Term, fo_env: Dict[str, str], state: Dict[str, object]) -> Term:
    if not term.is_variable:
        return term
    if term.name in fo_env:
        return Term(fo_env[term.name], is_variable=True)

    free_var_map: Dict[str, str] = state["free_var_map"]  # type: ignore[assignment]
    if term.name not in free_var_map:
        idx = state["free_var_counter"]  # type: ignore[assignment]
        free_var_map[term.name] = f"v{idx}"
        state["free_var_counter"] = idx + 1  # type: ignore[operator]
    return Term(free_var_map[term.name], is_variable=True)


def _canonical_pred_name(
    name: str,
    pred_env: Dict[str, str],
    state: Dict[str, object],
) -> str:
    if name in pred_env:
        return pred_env[name]

    pred_map: Dict[str, str] = state["pred_map"]  # type: ignore[assignment]
    if name not in pred_map:
        idx = state["pred_counter"]  # type: ignore[assignment]
        pred_map[name] = f"P{idx}"
        state["pred_counter"] = idx + 1  # type: ignore[operator]
    return pred_map[name]


def _canonicalize_rec(
    f: Formula,
    *,
    fo_env: Dict[str, str],
    pred_env: Dict[str, str],
    state: Dict[str, object],
) -> Formula:
    t = type(f)
    if t is _PropVar:
        prop_map: Dict[str, str] = state["prop_map"]  # type: ignore[assignment]
        if f.name not in prop_map:
            idx = state["prop_counter"]  # type: ignore[assignment]
            prop_map[f.name] = f"x{idx}"
            state["prop_counter"] = idx + 1  # type: ignore[operator]
        return PropVar(prop_map[f.name])

    if t is _Membership:
        return Membership(
            _canonical_term(f.left, fo_env, state),
            _canonical_term(f.right, fo_env, state),
        )

    if t is _Equality:
        return Equality(
            _canonical_term(f.left, fo_env, state),
            _canonical_term(f.right, fo_env, state),
        )

    if t is _Implies:
        return Implies(
            _canonicalize_rec(f.antecedent, fo_env=fo_env, pred_env=pred_env, state=state),
            _canonicalize_rec(f.consequent, fo_env=fo_env, pred_env=pred_env, state=state),
        )

    if t in (_And, _Or, _Iff):
        return t(
            _canonicalize_rec(f.left, fo_env=fo_env, pred_env=pred_env, state=state),
            _canonicalize_rec(f.right, fo_env=fo_env, pred_env=pred_env, state=state),
        )

    if t is _Not:
        return Not(_canonicalize_rec(f.inner, fo_env=fo_env, pred_env=pred_env, state=state))

    if t is _Box:
        return Box(_canonicalize_rec(f.inner, fo_env=fo_env, pred_env=pred_env, state=state))

    if t is _ForAll or t is _Exists:
        idx = state["bound_counter"]  # type: ignore[assignment]
        canonical_name = f"b{idx}"
        state["bound_counter"] = idx + 1  # type: ignore[operator]
        new_env = dict(fo_env)
        new_env[f.variable] = canonical_name
        return t(
            canonical_name,
            _canonicalize_rec(f.body, fo_env=new_env, pred_env=pred_env, state=state),
        )

    if t is _PredicateVar:
        return PredicateVar(_canonical_pred_name(f.name, pred_env, state), f.arity)

    if t is _PredApp:
        return PredApp(
            _canonical_pred_name(f.pred_var_name, pred_env, state),
            tuple(_canonical_term(arg, fo_env, state) for arg in f.args),
        )

    if t is _ForAllPred or t is _ExistsPred:
        idx = state["pred_counter"]  # type: ignore[assignment]
        canonical_name = f"P{idx}"
        state["pred_counter"] = idx + 1  # type: ignore[operator]
        new_pred_env = dict(pred_env)
        new_pred_env[f.pred_var] = canonical_name
        return t(
            canonical_name,
            f.arity,
            _canonicalize_rec(f.body, fo_env=fo_env, pred_env=new_pred_env, state=state),
        )

    if t in (_BigAnd, _BigOr):
        return t(tuple(
            _canonicalize_rec(sub, fo_env=fo_env, pred_env=pred_env, state=state)
            for sub in f.formulas
        ))

    return f

def normalize_schema(formula: Formula) -> Formula:
    """Rename PropVars by DFS first-appearance order: x0, x1, x2, …

    p→(q→p)  →  x0→(x1→x0)
    q→(p→q)  →  x0→(x1→x0)   (same schema)
    p→(p→p)  →  x0→(x0→x0)   (different — 1 variable)
    """
    var_order = collect_vars_ordered(formula)
    if not var_order:
        return formula
    mapping: Dict[str, Formula] = {}
    for i, old_name in enumerate(var_order):
        mapping[old_name] = PropVar(f"x{i}")
    return apply_subst(mapping, formula)


def schema_canonical_string(formula: Formula) -> str:
    """Canonical string representation for the graph vertex quotient."""
    normalized = canonicalize_formula(formula)
    return _structural_string(normalized)


def canonicalize_with_hash(formula: Formula) -> Tuple[Formula, str]:
    """Canonicalize + compute hash in a single AST traversal.

    Equivalent to ``(canonicalize_formula(f), schema_hash_normalized(canonicalize_formula(f)))``
    but avoids the second traversal: structural-string parts are collected
    during the canonicalization recursion and joined once at the end.
    """
    if _AC_NORMALIZE:
        formula = _ac_normalize(formula)
    state = {
        "prop_map": {},
        "free_var_map": {},
        "pred_map": {},
        "bound_counter": 0,
        "prop_counter": 0,
        "free_var_counter": 0,
        "pred_counter": 0,
    }
    parts: List[str] = []
    canon = _canonicalize_with_string_rec(
        formula, fo_env={}, pred_env={}, state=state, parts=parts
    )
    digest = hashlib.md5("".join(parts).encode()).hexdigest()
    return canon, digest


def _canonicalize_with_string_rec(
    f: Formula,
    *,
    fo_env: Dict[str, str],
    pred_env: Dict[str, str],
    state: Dict[str, object],
    parts: List[str],
) -> Formula:
    """Fused canonicalize + structural_string traversal.

    Mirrors ``_canonicalize_rec`` exactly in terms of canonicalization,
    but also appends structural-string fragments to ``parts`` so the caller
    can produce ``_structural_string(canonicalize_formula(f))`` without a
    second traversal.
    """
    t = type(f)
    if t is _PropVar:
        prop_map: Dict[str, str] = state["prop_map"]  # type: ignore[assignment]
        if f.name not in prop_map:
            idx = state["prop_counter"]  # type: ignore[assignment]
            prop_map[f.name] = f"x{idx}"
            state["prop_counter"] = idx + 1  # type: ignore[operator]
        new_name = prop_map[f.name]
        parts.append(f"PROP({new_name})")
        return PropVar(new_name)

    if t is _Membership:
        left = _canonical_term(f.left, fo_env, state)
        right = _canonical_term(f.right, fo_env, state)
        parts.append(f"MEM({_term_string(left)},{_term_string(right)})")
        return Membership(left, right)

    if t is _Equality:
        left = _canonical_term(f.left, fo_env, state)
        right = _canonical_term(f.right, fo_env, state)
        parts.append(f"EQ({_term_string(left)},{_term_string(right)})")
        return Equality(left, right)

    if t is _Implies:
        parts.append("IMP(")
        a = _canonicalize_with_string_rec(
            f.antecedent, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(",")
        c = _canonicalize_with_string_rec(
            f.consequent, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(")")
        return Implies(a, c)

    if t in (_And, _Or, _Iff):
        parts.append(f"{f.formula_type.value.upper()}(")
        left = _canonicalize_with_string_rec(
            f.left, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(",")
        right = _canonicalize_with_string_rec(
            f.right, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(")")
        return t(left, right)

    if t is _Not:
        parts.append("NOT(")
        inner = _canonicalize_with_string_rec(
            f.inner, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(")")
        return Not(inner)

    if t is _Box:
        parts.append("BOX(")
        inner = _canonicalize_with_string_rec(
            f.inner, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(")")
        return Box(inner)

    if t is _ForAll or t is _Exists:
        idx = state["bound_counter"]  # type: ignore[assignment]
        canonical_name = f"b{idx}"
        state["bound_counter"] = idx + 1  # type: ignore[operator]
        new_env = dict(fo_env)
        new_env[f.variable] = canonical_name
        tag = "FORALL" if t is _ForAll else "EXISTS"
        parts.append(f"{tag}({canonical_name},")
        body = _canonicalize_with_string_rec(
            f.body, fo_env=new_env, pred_env=pred_env, state=state, parts=parts
        )
        parts.append(")")
        return t(canonical_name, body)

    if t is _PredicateVar:
        name = _canonical_pred_name(f.name, pred_env, state)
        parts.append(f"PVAR({name},{f.arity})")
        return PredicateVar(name, f.arity)

    if t is _PredApp:
        name = _canonical_pred_name(f.pred_var_name, pred_env, state)
        args = tuple(_canonical_term(arg, fo_env, state) for arg in f.args)
        arg_strs = ",".join(_term_string(arg) for arg in args)
        parts.append(f"PAPP({name},{arg_strs})")
        return PredApp(name, args)

    if t is _ForAllPred or t is _ExistsPred:
        idx = state["pred_counter"]  # type: ignore[assignment]
        canonical_name = f"P{idx}"
        state["pred_counter"] = idx + 1  # type: ignore[operator]
        new_pred_env = dict(pred_env)
        new_pred_env[f.pred_var] = canonical_name
        tag = "FALLP" if t is _ForAllPred else "EXISTSP"
        parts.append(f"{tag}({canonical_name},{f.arity},")
        body = _canonicalize_with_string_rec(
            f.body, fo_env=fo_env, pred_env=new_pred_env, state=state, parts=parts
        )
        parts.append(")")
        return t(canonical_name, f.arity, body)

    if t in (_BigAnd, _BigOr):
        tag = "BIGAND" if t is _BigAnd else "BIGOR"
        parts.append(f"{tag}(")
        new_subs = []
        for i, sub in enumerate(f.formulas):
            if i > 0:
                parts.append(",")
            new_subs.append(_canonicalize_with_string_rec(
                sub, fo_env=fo_env, pred_env=pred_env, state=state, parts=parts
            ))
        parts.append(")")
        return t(tuple(new_subs))

    # Fallback for unknown types: behave like _canonicalize_rec + repr.
    parts.append(repr(f))
    return f


def schema_hash(formula: Formula) -> str:
    """Canonical hash of a normalized schema."""
    canonical = schema_canonical_string(formula)
    return hashlib.md5(canonical.encode()).hexdigest()


def schema_hash_normalized(formula: Formula) -> str:
    """Canonical hash assuming the formula is already structurally canonical.

    Use only when the caller can guarantee
    ``formula == canonicalize_formula(formula)``.
    """
    canonical = _structural_string(formula)
    return hashlib.md5(canonical.encode()).hexdigest()


def _term_string(term: Term) -> str:
    tag = "V" if term.is_variable else "C"
    return f"{tag}:{term.name}"


def _structural_string(formula: Formula) -> str:
    """Serialize an already-canonical formula without semantic rewrites."""
    t = type(formula)
    if t is _PropVar:
        return f"PROP({formula.name})"
    if t is _Membership:
        return f"MEM({_term_string(formula.left)},{_term_string(formula.right)})"
    if t is _Equality:
        return f"EQ({_term_string(formula.left)},{_term_string(formula.right)})"
    if t is _Implies:
        return (
            f"IMP({_structural_string(formula.antecedent)},"
            f"{_structural_string(formula.consequent)})"
        )
    if t in (_And, _Or, _Iff):
        return (
            f"{formula.formula_type.value.upper()}("
            f"{_structural_string(formula.left)},{_structural_string(formula.right)})"
        )
    if t is _Not:
        return f"NOT({_structural_string(formula.inner)})"
    if t is _Box:
        return f"BOX({_structural_string(formula.inner)})"
    if t is _ForAll:
        return f"FORALL({formula.variable},{_structural_string(formula.body)})"
    if t is _Exists:
        return f"EXISTS({formula.variable},{_structural_string(formula.body)})"
    if t is _PredicateVar:
        return f"PVAR({formula.name},{formula.arity})"
    if t is _PredApp:
        args = ",".join(_term_string(arg) for arg in formula.args)
        return f"PAPP({formula.pred_var_name},{args})"
    if t is _ForAllPred:
        return (
            f"FALLP({formula.pred_var},{formula.arity},"
            f"{_structural_string(formula.body)})"
        )
    if t is _ExistsPred:
        return (
            f"EXISTSP({formula.pred_var},{formula.arity},"
            f"{_structural_string(formula.body)})"
        )
    if t is _BigAnd:
        return "BIGAND(" + ",".join(_structural_string(f) for f in formula.formulas) + ")"
    if t is _BigOr:
        return "BIGOR(" + ",".join(_structural_string(f) for f in formula.formulas) + ")"
    return repr(formula)


# ═══════════════════════════════════════════════════════════════════════════
#  Rename-apart (avoid variable clashes before unification)
# ═══════════════════════════════════════════════════════════════════════════

_rename_counter: int = 0


def _fresh_prefix() -> str:
    global _rename_counter
    _rename_counter += 1
    return f"_r{_rename_counter}_"


def rename_apart(formula: Formula, avoid: Set[str]) -> Tuple[Formula, Dict[str, str]]:
    """Rename all PropVars in formula so they don't clash with `avoid`.

    Returns (renamed_formula, mapping {old_name → new_name}).

    Fast path: if formula's vars don't intersect ``avoid``, no rename is
    needed — return the original formula. This avoids the O(size) AST
    rebuild via apply_subst for the typical canonical-MP case where minor
    and major have disjoint variable namespaces after canonicalize.
    """
    formula_vars = collect_vars(formula)
    if not (formula_vars & avoid):
        return formula, {}
    # Conflict exists: rename to guarantee disjoint variable sets
    prefix = _fresh_prefix()
    mapping: Dict[str, Formula] = {}
    name_map: Dict[str, str] = {}
    for v in collect_vars_ordered(formula):
        new_name = f"{prefix}{v}"
        mapping[v] = PropVar(new_name)
        name_map[v] = new_name
    return apply_subst(mapping, formula), name_map


# ═══════════════════════════════════════════════════════════════════════════
#  Occurs check
# ═══════════════════════════════════════════════════════════════════════════

def occurs_in(var_name: str, formula: Formula) -> bool:
    """Does PropVar(var_name) appear anywhere inside formula?"""
    t = type(formula)
    if t is _PropVar:
        return formula.name == var_name
    if t is _Implies:
        return occurs_in(var_name, formula.antecedent) or occurs_in(var_name, formula.consequent)
    if t in (_And, _Or, _Iff):
        return occurs_in(var_name, formula.left) or occurs_in(var_name, formula.right)
    if t is _Not or t is _Box:
        return occurs_in(var_name, formula.inner)
    if t is _ForAll or t is _Exists:
        return occurs_in(var_name, formula.body)
    if t is _ForAllPred or t is _ExistsPred:
        return occurs_in(var_name, formula.body)
    if t in (_BigAnd, _BigOr):
        return any(occurs_in(var_name, f) for f in formula.formulas)
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Unification (Martelli–Montanari)
# ═══════════════════════════════════════════════════════════════════════════

def unify(
    f1: Formula,
    f2: Formula,
    subst: Optional[Dict[str, Formula]] = None,
) -> Optional[Dict[str, Formula]]:
    """Compute the most general unifier of f1 and f2.

    PropVars are treated as metavariables that can be bound to any formula.
    Returns substitution dict {var_name → Formula} or None if unification fails.

    Uses occurs check to prevent infinite terms.
    """
    if subst is None:
        subst = {}

    # Apply current substitution before comparing
    f1 = _walk(f1, subst)
    f2 = _walk(f2, subst)

    # Identical after substitution
    if f1 is f2 or f1 == f2:
        return subst

    t1 = type(f1)
    t2 = type(f2)

    # Variable cases
    if t1 is _PropVar:
        return _bind(f1.name, f2, subst)
    if t2 is _PropVar:
        return _bind(f2.name, f1, subst)

    # Structural cases — same constructor
    if t1 is not t2:
        return None

    if t1 is _Implies:
        subst = unify(f1.antecedent, f2.antecedent, subst)
        if subst is None:
            return None
        return unify(f1.consequent, f2.consequent, subst)

    if t1 in (_And, _Or, _Iff):
        subst = unify(f1.left, f2.left, subst)
        if subst is None:
            return None
        return unify(f1.right, f2.right, subst)

    if t1 is _Not or t1 is _Box:
        return unify(f1.inner, f2.inner, subst)

    if t1 is _Membership or t1 is _Equality:
        if _terms_equal(f1.left, f2.left) and _terms_equal(f1.right, f2.right):
            return subst
        return None

    if t1 is _ForAll or t1 is _Exists:
        # Align both bound variables to one fresh name (alpha-equivalence safe).
        fresh = _fresh_bound_name(f1, f2)
        left_body = f1.body.substitute(f1.variable, Term(fresh, is_variable=True))
        right_body = f2.body.substitute(f2.variable, Term(fresh, is_variable=True))
        return unify(left_body, right_body, subst)

    if t1 is _ForAllPred or t1 is _ExistsPred:
        # Arities must match; align bound predicate variable names.
        if f1.arity != f2.arity:
            return None
        # Alpha-rename: substitute pred_var references via a fresh name.
        fresh_p = _fresh_pred_name(f1, f2)
        left_body = _subst_pred_var(f1.pred_var, fresh_p, f1.body)
        right_body = _subst_pred_var(f2.pred_var, fresh_p, f2.body)
        return unify(left_body, right_body, subst)

    if t1 is _PredApp:
        # Same predicate variable name and same arity of args required.
        if f1.pred_var_name != f2.pred_var_name or len(f1.args) != len(f2.args):
            return None
        for a1, a2 in zip(f1.args, f2.args):
            if a1 != a2:
                return None
        return subst

    if t1 in (_BigAnd, _BigOr):
        if len(f1.formulas) != len(f2.formulas):
            return None
        for sub1, sub2 in zip(f1.formulas, f2.formulas):
            subst = unify(sub1, sub2, subst)
            if subst is None:
                return None
        return subst

    # Unknown formula types — structural equality
    return subst if f1 == f2 else None


# ═══════════════════════════════════════════════════════════════════════════
#  One-way matching (subsumption)
# ═══════════════════════════════════════════════════════════════════════════

def match(
    pattern: Formula,
    target: Formula,
) -> Optional[Dict[str, Formula]]:
    """One-way matching: find σ such that ``apply_subst(σ, pattern) == target``.

    Differs from :func:`unify` in that PropVars are metavariables **only**
    in ``pattern``. PropVars appearing in ``target`` are treated as ground
    constants — they must align literally with the pattern, never bind to
    something else.

    Used by the subsumption index: ``match(φ, ψ) ≠ None`` iff ``φ ⊒ ψ``
    (φ subsumes ψ).

    Returns:
        Substitution dict ``{var_name → Formula}`` if a match exists,
        otherwise ``None``.

    Caller responsibility:
        Both ``pattern`` and ``target`` are expected to be in structural
        canonical form (see :func:`canonicalize_formula`). Bound first-order
        variables are required to align literally — alpha-equivalence is
        handled by canonicalization, not by this function.
    """
    return _match_rec(pattern, target, {})


def _match_rec(
    pattern: Formula,
    target: Formula,
    sigma: Dict[str, Formula],
) -> Optional[Dict[str, Formula]]:
    tp = type(pattern)
    # ── PropVar on the pattern side: bind or check existing binding ──
    if tp is _PropVar:
        name = pattern.name
        if name in sigma:
            return sigma if sigma[name] == target else None
        new_sigma = dict(sigma)
        new_sigma[name] = target
        return new_sigma

    tt = type(target)
    if tp is not tt:
        return None

    if tp is _Implies:
        sigma = _match_rec(pattern.antecedent, target.antecedent, sigma)
        if sigma is None:
            return None
        return _match_rec(pattern.consequent, target.consequent, sigma)

    if tp in _BINARY_TYPES:
        sigma = _match_rec(pattern.left, target.left, sigma)
        if sigma is None:
            return None
        return _match_rec(pattern.right, target.right, sigma)

    if tp is _Not or tp is _Box:
        return _match_rec(pattern.inner, target.inner, sigma)

    if tp is _Membership or tp is _Equality:
        # No PropVars inside atomic FO formulas — pure ground equality.
        if pattern.left == target.left and pattern.right == target.right:
            return sigma
        return None

    if tp is _ForAll or tp is _Exists:
        # Bound variable names must align literally (caller canonicalized).
        if pattern.variable != target.variable:
            return None
        return _match_rec(pattern.body, target.body, sigma)

    if tp is _PredicateVar:
        return sigma if pattern == target else None

    if tp is _PredApp:
        if (
            pattern.pred_var_name != target.pred_var_name
            or len(pattern.args) != len(target.args)
        ):
            return None
        for a, b in zip(pattern.args, target.args):
            if a != b:
                return None
        return sigma

    if tp is _ForAllPred or tp is _ExistsPred:
        if pattern.pred_var != target.pred_var or pattern.arity != target.arity:
            return None
        return _match_rec(pattern.body, target.body, sigma)

    if tp in _BIG_TYPES:
        if len(pattern.formulas) != len(target.formulas):
            return None
        for a, b in zip(pattern.formulas, target.formulas):
            sigma = _match_rec(a, b, sigma)
            if sigma is None:
                return None
        return sigma

    # Unknown constructor — structural equality only.
    return sigma if pattern == target else None


# ═══════════════════════════════════════════════════════════════════════════
#  Pattern unification (Phase A): predicate metavariables (Miller fragment)
# ═══════════════════════════════════════════════════════════════════════════
#
# A predicate metavariable is a PredApp whose name is FREE (not bound by an
# enclosing ForAllPred/ExistsPred) — the second-order analogue of a PropVar.
# `pattern_match` is one-way (metavars only in the pattern) and handles the
# Miller higher-order-pattern fragment: a metavar applied to DISTINCT first-
# order variables. Non-pattern applications (repeated/constant args) are
# deferred (return None) — sound for subsumption matching. See
# docs/PHASE_A_DESIGN.md §3. Isolated capability: `match`/`unify`/the generator
# are untouched.

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class PredTemplate:
    """A predicate-metavariable binding ``?P ↦ λ(params). body``."""
    params: Tuple[str, ...]
    body: Formula


def _instantiate_template(tmpl: "PredTemplate", args: Tuple[Term, ...]) -> Formula:
    body = tmpl.body
    for pv, arg in zip(tmpl.params, args):
        body = body.substitute(pv, arg)
    return body


def _apply_pred_template_in(name: str, tmpl: "PredTemplate", f: Formula) -> Formula:
    """Replace every free ``PredApp(name, args)`` in f by ``tmpl`` applied to args."""
    t = type(f)
    if t is _PredApp:
        if f.pred_var_name == name:
            return _instantiate_template(tmpl, f.args)
        return f
    if t is _Implies:
        return Implies(_apply_pred_template_in(name, tmpl, f.antecedent),
                       _apply_pred_template_in(name, tmpl, f.consequent))
    if t in _BINARY_TYPES:
        return t(_apply_pred_template_in(name, tmpl, f.left),
                 _apply_pred_template_in(name, tmpl, f.right))
    if t is _Not or t is _Box:
        return t(_apply_pred_template_in(name, tmpl, f.inner))
    if t is _ForAll or t is _Exists:
        return t(f.variable, _apply_pred_template_in(name, tmpl, f.body))
    if t is _ForAllPred or t is _ExistsPred:
        if f.pred_var == name:           # re-binding shadows the metavar
            return f
        return t(f.pred_var, f.arity, _apply_pred_template_in(name, tmpl, f.body))
    if t in _BIG_TYPES:
        return t(tuple(_apply_pred_template_in(name, tmpl, x) for x in f.formulas))
    return f  # atomic (Membership, Equality, PropVar, PredicateVar)


def apply_pattern_subst(sigma: Dict[str, object], formula: Formula) -> Formula:
    """Apply a mixed substitution (PropVar→Formula, predname→PredTemplate)."""
    prop = {k: v for k, v in sigma.items() if isinstance(v, Formula)}
    preds = {k: v for k, v in sigma.items() if isinstance(v, PredTemplate)}
    out = apply_subst(prop, formula) if prop else formula
    for name, tmpl in preds.items():
        out = _apply_pred_template_in(name, tmpl, out)
    return out


def _fresh_params(n: int, avoid: Set[str]) -> Tuple[str, ...]:
    out: List[str] = []
    i = 0
    while len(out) < n:
        name = f"_pp{i}"
        i += 1
        if name not in avoid:
            out.append(name)
    return tuple(out)


def _match_pred_metavar(
    pattern: Formula, target: Formula, sigma: Dict[str, object],
) -> Optional[Dict[str, object]]:
    args = pattern.args
    # Miller condition: applied to DISTINCT first-order variables.
    if not all(a.is_variable for a in args):
        return None
    argnames = [a.name for a in args]
    if len(set(argnames)) != len(argnames):
        return None
    name = pattern.pred_var_name
    if name in sigma:
        prev = sigma[name]
        if not isinstance(prev, PredTemplate):
            return None
        # Consistency: the existing binding applied to THESE args == target.
        return sigma if _instantiate_template(prev, args) == target else None
    avoid = set(argnames) | set(target.free_variables()) | set(target.constants())
    params = _fresh_params(len(args), avoid)
    body = target
    for an, pp in zip(argnames, params):
        body = body.substitute(an, Term(pp, is_variable=True))
    new_sigma = dict(sigma)
    new_sigma[name] = PredTemplate(params, body)
    return new_sigma


def pattern_match(pattern: Formula, target: Formula) -> Optional[Dict[str, object]]:
    """One-way match extended with predicate metavariables (Miller fragment).

    Returns σ with ``apply_pattern_subst(σ, pattern) == target`` (σ mixes
    PropVar→Formula and predname→PredTemplate), or None. Free PredApp names are
    metavariables; bound predicate variables are rigid.
    """
    return _pattern_match_rec(pattern, target, {}, frozenset())


def _pattern_match_rec(
    pattern: Formula, target: Formula,
    sigma: Dict[str, object], bound_preds: frozenset,
) -> Optional[Dict[str, object]]:
    tp = type(pattern)
    if tp is _PropVar:
        name = pattern.name
        if name in sigma:
            return sigma if sigma[name] == target else None
        ns = dict(sigma)
        ns[name] = target
        return ns
    # Free predicate metavariable (not bound by an enclosing ForAllPred).
    if tp is _PredApp and pattern.pred_var_name not in bound_preds:
        return _match_pred_metavar(pattern, target, sigma)

    tt = type(target)
    if tp is not tt:
        return None

    if tp is _Implies:
        sigma = _pattern_match_rec(pattern.antecedent, target.antecedent, sigma, bound_preds)
        if sigma is None:
            return None
        return _pattern_match_rec(pattern.consequent, target.consequent, sigma, bound_preds)
    if tp in _BINARY_TYPES:
        sigma = _pattern_match_rec(pattern.left, target.left, sigma, bound_preds)
        if sigma is None:
            return None
        return _pattern_match_rec(pattern.right, target.right, sigma, bound_preds)
    if tp is _Not or tp is _Box:
        return _pattern_match_rec(pattern.inner, target.inner, sigma, bound_preds)
    if tp is _Membership or tp is _Equality:
        if pattern.left == target.left and pattern.right == target.right:
            return sigma
        return None
    if tp is _ForAll or tp is _Exists:
        if pattern.variable != target.variable:
            return None
        return _pattern_match_rec(pattern.body, target.body, sigma, bound_preds)
    if tp is _PredicateVar:
        return sigma if pattern == target else None
    if tp is _PredApp:  # bound (rigid) predicate application
        if (pattern.pred_var_name != target.pred_var_name
                or len(pattern.args) != len(target.args)):
            return None
        for a, b in zip(pattern.args, target.args):
            if a != b:
                return None
        return sigma
    if tp is _ForAllPred or tp is _ExistsPred:
        if pattern.pred_var != target.pred_var or pattern.arity != target.arity:
            return None
        return _pattern_match_rec(
            pattern.body, target.body, sigma, bound_preds | {pattern.pred_var}
        )
    if tp in _BIG_TYPES:
        if len(pattern.formulas) != len(target.formulas):
            return None
        for a, b in zip(pattern.formulas, target.formulas):
            sigma = _pattern_match_rec(a, b, sigma, bound_preds)
            if sigma is None:
                return None
        return sigma
    return sigma if pattern == target else None


def _collect_pred_names(f: Formula, acc: Set[str]) -> Set[str]:
    """Collect all PredApp names in f (for predicate-metavar occurs-check)."""
    t = type(f)
    if t is _PredApp:
        acc.add(f.pred_var_name)
    elif t is _Implies:
        _collect_pred_names(f.antecedent, acc); _collect_pred_names(f.consequent, acc)
    elif t in _BINARY_TYPES:
        _collect_pred_names(f.left, acc); _collect_pred_names(f.right, acc)
    elif t is _Not or t is _Box:
        _collect_pred_names(f.inner, acc)
    elif t is _ForAll or t is _Exists:
        _collect_pred_names(f.body, acc)
    elif t is _ForAllPred or t is _ExistsPred:
        _collect_pred_names(f.body, acc)
    elif t in _BIG_TYPES:
        for x in f.formulas:
            _collect_pred_names(x, acc)
    return acc


def _full_apply_pat(sigma: Dict[str, object], f: Formula) -> Formula:
    """Recursively resolve a mixed substitution (PropVar chains + PredTemplate
    applications) to a fixpoint. Occurs-checks at bind time prevent cycles.

    Precondition: predicate-metavariable names are disjoint from bound
    predicate-variable names (holds in all configs; metavars come from free
    PredApps in axiom schemas, bound vars from ForAllPred/ExistsPred)."""
    t = type(f)
    if t is _PropVar:
        v = sigma.get(f.name)
        if isinstance(v, Formula):
            return _full_apply_pat(sigma, v)
        return f
    if t is _PredApp:
        v = sigma.get(f.pred_var_name)
        if isinstance(v, PredTemplate):
            return _full_apply_pat(sigma, _instantiate_template(v, f.args))
        return f
    if t is _Implies:
        return Implies(_full_apply_pat(sigma, f.antecedent),
                       _full_apply_pat(sigma, f.consequent))
    if t in _BINARY_TYPES:
        return t(_full_apply_pat(sigma, f.left), _full_apply_pat(sigma, f.right))
    if t is _Not or t is _Box:
        return t(_full_apply_pat(sigma, f.inner))
    if t is _ForAll or t is _Exists:
        return t(f.variable, _full_apply_pat(sigma, f.body))
    if t is _ForAllPred or t is _ExistsPred:
        return t(f.pred_var, f.arity, _full_apply_pat(sigma, f.body))
    if t in _BIG_TYPES:
        return t(tuple(_full_apply_pat(sigma, x) for x in f.formulas))
    return f


def _bind_prop_pat(name: str, term: Formula, sigma: Dict[str, object]) -> Optional[Dict[str, object]]:
    if type(term) is _PropVar and term.name == name:
        return sigma
    resolved = _full_apply_pat(sigma, term)
    if occurs_in(name, resolved):
        return None
    ns = dict(sigma)
    ns[name] = resolved
    return ns


def _bind_pred_pat(predapp: Formula, other: Formula, sigma: Dict[str, object]) -> Optional[Dict[str, object]]:
    args = predapp.args
    if not all(a.is_variable for a in args):          # Miller: distinct vars only
        return None
    argnames = [a.name for a in args]
    if len(set(argnames)) != len(argnames):
        return None
    name = predapp.pred_var_name
    resolved = _full_apply_pat(sigma, other)
    if name in _collect_pred_names(resolved, set()):  # occurs check (cyclic)
        return None
    avoid = set(argnames) | set(resolved.free_variables()) | set(resolved.constants())
    params = _fresh_params(len(args), avoid)
    body = resolved
    for an, pp in zip(argnames, params):
        body = body.substitute(an, Term(pp, is_variable=True))
    ns = dict(sigma)
    ns[name] = PredTemplate(params, body)
    return ns


def pattern_unify(f1: Formula, f2: Formula) -> Optional[Dict[str, object]]:
    """Two-way unification with predicate metavariables (Miller fragment).

    Conservative extension of :func:`unify`: PropVars + structure as before,
    plus flex-rigid binding of free predicate metavariables. Flex-flex (a
    predicate metavar on BOTH sides) is deferred (returns None) — sound but
    incomplete; sufficient for unifying derived schemas against ?P-bearing
    axioms (Separation/Replacement/Comprehension). Returns σ with
    ``apply_pattern_subst(σ, f1) == apply_pattern_subst(σ, f2)`` or None.
    See docs/PHASE_A_DESIGN.md §3."""
    return _pattern_unify_rec(f1, f2, {}, frozenset())


def _pattern_unify_rec(
    f1: Formula, f2: Formula, sigma: Dict[str, object], bp: frozenset,
) -> Optional[Dict[str, object]]:
    f1 = _full_apply_pat(sigma, f1)
    f2 = _full_apply_pat(sigma, f2)
    if f1 == f2:
        return sigma
    t1, t2 = type(f1), type(f2)

    if t1 is _PropVar:
        return _bind_prop_pat(f1.name, f2, sigma)
    if t2 is _PropVar:
        return _bind_prop_pat(f2.name, f1, sigma)

    f1_flex = (t1 is _PredApp and f1.pred_var_name not in bp)
    f2_flex = (t2 is _PredApp and f2.pred_var_name not in bp)
    if f1_flex and f2_flex:
        return None                       # flex-flex deferred (conservative)
    if f1_flex:
        return _bind_pred_pat(f1, f2, sigma)
    if f2_flex:
        return _bind_pred_pat(f2, f1, sigma)

    if t1 is not t2:
        return None

    if t1 is _Implies:
        sigma = _pattern_unify_rec(f1.antecedent, f2.antecedent, sigma, bp)
        if sigma is None:
            return None
        return _pattern_unify_rec(f1.consequent, f2.consequent, sigma, bp)
    if t1 in _BINARY_TYPES:
        sigma = _pattern_unify_rec(f1.left, f2.left, sigma, bp)
        if sigma is None:
            return None
        return _pattern_unify_rec(f1.right, f2.right, sigma, bp)
    if t1 is _Not or t1 is _Box:
        return _pattern_unify_rec(f1.inner, f2.inner, sigma, bp)
    if t1 is _Membership or t1 is _Equality:
        if _terms_equal(f1.left, f2.left) and _terms_equal(f1.right, f2.right):
            return sigma
        return None
    if t1 is _ForAll or t1 is _Exists:
        fresh = _fresh_bound_name(f1, f2)
        lb = f1.body.substitute(f1.variable, Term(fresh, is_variable=True))
        rb = f2.body.substitute(f2.variable, Term(fresh, is_variable=True))
        return _pattern_unify_rec(lb, rb, sigma, bp)
    if t1 is _ForAllPred or t1 is _ExistsPred:
        if f1.arity != f2.arity:
            return None
        fresh_p = _fresh_pred_name(f1, f2)
        lb = _subst_pred_var(f1.pred_var, fresh_p, f1.body)
        rb = _subst_pred_var(f2.pred_var, fresh_p, f2.body)
        return _pattern_unify_rec(lb, rb, sigma, bp | {fresh_p})
    if t1 is _PredApp:                     # both rigid (bound) here
        if f1.pred_var_name != f2.pred_var_name or len(f1.args) != len(f2.args):
            return None
        for a1, a2 in zip(f1.args, f2.args):
            if a1 != a2:
                return None
        return sigma
    if t1 in _BIG_TYPES:
        if len(f1.formulas) != len(f2.formulas):
            return None
        for s1, s2 in zip(f1.formulas, f2.formulas):
            sigma = _pattern_unify_rec(s1, s2, sigma, bp)
            if sigma is None:
                return None
        return sigma
    return sigma if f1 == f2 else None


def _terms_equal(t1: Term, t2: Term) -> bool:
    """Term equality helper (terms are atomic in this project)."""
    return t1 == t2


def _fresh_bound_name(left_q, right_q) -> str:
    """Choose a fresh variable name for quantifier alpha-alignment."""
    used = set(left_q.body.all_variables()) | set(right_q.body.all_variables())
    used.add(left_q.variable)
    used.add(right_q.variable)
    i = 0
    while True:
        name = f"_qb{i}"
        if name not in used:
            return name
        i += 1


def _fresh_pred_name(left_qp, right_qp) -> str:
    """Choose a fresh predicate variable name for second-order quantifier alpha-alignment."""
    i = 0
    while True:
        name = f"_qp{i}"
        if name != left_qp.pred_var and name != right_qp.pred_var:
            return name
        i += 1


def _subst_pred_var(old_name: str, new_name: str, formula: Formula) -> Formula:
    """Rename all PredApp(old_name, ...) to PredApp(new_name, ...) in formula.

    Used for alpha-renaming in second-order quantifier unification.
    """
    t = type(formula)
    if t is _PredApp:
        if formula.pred_var_name == old_name:
            return PredApp(new_name, formula.args)
        return formula
    if t is _Implies:
        return Implies(
            _subst_pred_var(old_name, new_name, formula.antecedent),
            _subst_pred_var(old_name, new_name, formula.consequent),
        )
    if t in (_And, _Or, _Iff):
        return t(
            _subst_pred_var(old_name, new_name, formula.left),
            _subst_pred_var(old_name, new_name, formula.right),
        )
    if t is _Not:
        return Not(_subst_pred_var(old_name, new_name, formula.inner))
    if t is _Box:
        return Box(_subst_pred_var(old_name, new_name, formula.inner))
    if t is _ForAll or t is _Exists:
        return t(formula.variable, _subst_pred_var(old_name, new_name, formula.body))
    if t is _ForAllPred or t is _ExistsPred:
        # Don't rename inside a re-binding of the same name
        if formula.pred_var == old_name:
            return formula
        return t(formula.pred_var, formula.arity,
                 _subst_pred_var(old_name, new_name, formula.body))
    if t in (_BigAnd, _BigOr):
        return t(tuple(_subst_pred_var(old_name, new_name, f) for f in formula.formulas))
    return formula


def _walk(formula: Formula, subst: Dict[str, Formula]) -> Formula:
    """Recursively apply substitution until we reach a non-variable or unbound variable."""
    while type(formula) is _PropVar and formula.name in subst:
        formula = subst[formula.name]
    return formula


def _bind(
    var_name: str, term: Formula, subst: Dict[str, Formula]
) -> Optional[Dict[str, Formula]]:
    """Bind var to term, checking for occurs."""
    if type(term) is _PropVar and term.name == var_name:
        return subst  # x = x, no-op
    # Apply subst to term to get the most resolved form
    resolved = _full_apply(subst, term)
    if occurs_in(var_name, resolved):
        return None  # occurs check failure
    new_subst = dict(subst)
    new_subst[var_name] = resolved
    return new_subst


def _full_apply(subst: Dict[str, Formula], formula: Formula) -> Formula:
    """Fully apply substitution (resolving chains)."""
    t = type(formula)
    if t is _PropVar:
        if formula.name in subst:
            return _full_apply(subst, subst[formula.name])
        return formula
    if t is _Implies:
        return Implies(
            _full_apply(subst, formula.antecedent),
            _full_apply(subst, formula.consequent),
        )
    if t in (_And, _Or, _Iff):
        return t(
            _full_apply(subst, formula.left),
            _full_apply(subst, formula.right),
        )
    if t is _Not:
        return Not(_full_apply(subst, formula.inner))
    if t is _Box:
        return Box(_full_apply(subst, formula.inner))
    if t is _ForAll or t is _Exists:
        return t(formula.variable, _full_apply(subst, formula.body))
    if t is _ForAllPred or t is _ExistsPred:
        return t(formula.pred_var, formula.arity, _full_apply(subst, formula.body))
    if t in (_BigAnd, _BigOr):
        return t(tuple(_full_apply(subst, f) for f in formula.formulas))
    return formula


# ═══════════════════════════════════════════════════════════════════════════
#  Schema MP (with pre-filter)
# ═══════════════════════════════════════════════════════════════════════════

def _top_tag(f: Formula) -> int:
    """Return a cheap integer tag for the top-level constructor.

    Used for fast pre-filtering of MP candidates: if the antecedent's
    top constructor is not a PropVar and differs from the minor's top
    constructor, unification is guaranteed to fail.

    Returns type id (using id(type)) for fast comparison.
    """
    return id(type(f))


# Pre-compute PropVar tag for the filter check.
_PROPVAR_TAG = id(_PropVar)


def detect_implicit_dvar_constraints(
    major: Formula,
) -> List[Tuple[str, str]]:
    """Detect implicit distinct-variable ($d) constraints from schema shape.

    For every PropVar metavariable ``p`` that appears in ``major.antecedent``
    (i.e., is unifiable at MP time) and also appears under a first-order
    quantifier scope ``∀v`` / ``∃v`` in ``major.consequent``, the schema
    requires ``v ∉ FV(σ(p))`` for any sound MP application — otherwise
    the substitution σ(p) may be captured by ``v`` in the result.

    Returns a list of ``(propvar_name, bound_var_name)`` constraints.

    This catches the metamath-style ax-5 pattern ``Implies(p, ForAll(v, p))``
    and any derived schema sharing the shape, without requiring per-axiom
    annotations.
    """
    if type(major) is not _Implies:
        return []

    ant = major.antecedent
    cons = major.consequent

    ant_propvars = collect_vars(ant)
    if not ant_propvars:
        return []

    seen: Set[Tuple[str, str]] = set()
    constraints: List[Tuple[str, str]] = []
    for p_name in ant_propvars:
        for binder_var in _find_scope_binders_over_propvar(p_name, cons, ()):
            key = (p_name, binder_var)
            if key not in seen:
                seen.add(key)
                constraints.append(key)
    return constraints


def _find_scope_binders_over_propvar(
    p_name: str,
    formula: Formula,
    scope: Tuple[str, ...],
) -> List[str]:
    """Return list of FO bound-var names whose scope contains a free
    occurrence of PropVar(p_name) in formula. Each binder is reported once
    per occurrence; callers should deduplicate."""
    t = type(formula)
    if t is _PropVar:
        if formula.name == p_name:
            return list(scope)
        return []
    if t is _ForAll or t is _Exists:
        return _find_scope_binders_over_propvar(
            p_name, formula.body, scope + (formula.variable,)
        )
    if t is _Implies:
        return (
            _find_scope_binders_over_propvar(p_name, formula.antecedent, scope)
            + _find_scope_binders_over_propvar(p_name, formula.consequent, scope)
        )
    if t in (_And, _Or, _Iff):
        return (
            _find_scope_binders_over_propvar(p_name, formula.left, scope)
            + _find_scope_binders_over_propvar(p_name, formula.right, scope)
        )
    if t is _Not or t is _Box:
        return _find_scope_binders_over_propvar(p_name, formula.inner, scope)
    if t is _ForAllPred or t is _ExistsPred:
        # Predicate binder — does not contribute FO scope.
        return _find_scope_binders_over_propvar(p_name, formula.body, scope)
    if t in (_BigAnd, _BigOr):
        out: List[str] = []
        for sub in formula.formulas:
            out.extend(_find_scope_binders_over_propvar(p_name, sub, scope))
        return out
    return []


def schema_mp(
    minor: Formula,
    major: Formula,
    *,
    major_vars: Optional[Set[str]] = None,
    antecedent_tag: Optional[int] = None,
    antecedent_is_wildcard: Optional[bool] = None,
    inputs_canonical: bool = False,
    dvar_constraints: Optional[List[Tuple[str, str]]] = None,
    canonicalize_output: bool = True,
    use_pattern_unify: bool = False,
) -> Optional[Formula]:
    """Apply Modus Ponens at schema level.

    major must be Implies(antecedent, consequent).
    Tries to unify minor with antecedent.
    If successful, returns ``apply(σ, consequent)`` — canonicalized if
    ``canonicalize_output`` (default True). Callers that will canonicalize
    the result themselves (e.g. MP workers which run
    ``canonicalize_with_hash`` for dedup) pass ``canonicalize_output=False``
    to skip the redundant traversal.

    Soundness guard: any $d-constraint of the major is checked against the
    unifier σ. If σ binds metavariable ``p`` to a formula whose free
    variables include the constrained binder ``v``, MP is rejected. This
    blocks unsound applications of metamath-style ax-5 (`p → ∀x.p` with
    `$d x p`) and related schemas. Constraints are auto-detected from the
    major's structure unless explicitly provided.
    """
    if not inputs_canonical:
        minor = canonicalize_formula(minor)
        major = canonicalize_formula(major)

    if type(major) is not _Implies:
        return None

    ant = major.antecedent

    # ── Pre-filter: skip if top constructors are incompatible ──
    # If antecedent is not a PropVar (which unifies with anything),
    # and the minor's top constructor differs, unification will fail.
    minor_tag = id(type(minor))
    ant_tag = antecedent_tag if antecedent_tag is not None else id(type(ant))
    ant_is_wildcard = (
        antecedent_is_wildcard
        if antecedent_is_wildcard is not None
        else ant_tag == _PROPVAR_TAG
    )
    # Pattern mode: a free predicate-metavar antecedent ?P(...) is flexible too
    # (it unifies with many constructors), so the rigid pre-filter must not
    # reject it.
    if use_pattern_unify and type(ant) is _PredApp:
        ant_is_wildcard = True
    if not ant_is_wildcard and ant_tag != minor_tag:
        # Neither side is a PropVar and constructors differ → no match
        if minor_tag != _PROPVAR_TAG:
            return None

    cons = major.consequent

    # Rename minor's variables to avoid clash with major
    if major_vars is None:
        major_vars = collect_vars(major)
    minor_renamed, _ = rename_apart(minor, major_vars)

    σ = (pattern_unify(minor_renamed, ant) if use_pattern_unify
         else unify(minor_renamed, ant))
    if σ is None:
        return None

    # Resolver handles mixed (PropVar + PredTemplate) σ in pattern mode;
    # plain _full_apply suffices for PropVar-only σ otherwise.
    resolver = _full_apply_pat if use_pattern_unify else _full_apply

    # ── Soundness guard: enforce $d-constraints ──
    if dvar_constraints is None:
        dvar_constraints = detect_implicit_dvar_constraints(major)
    if dvar_constraints:
        for p_name, forbidden_var in dvar_constraints:
            bound_value = resolver(σ, _PropVar(p_name))
            if bound_value is not None and forbidden_var in bound_value.free_variables():
                return None  # $d-constraint violated → reject MP

    result = resolver(σ, cons)
    if canonicalize_output:
        return canonicalize_formula(result)
    return result
