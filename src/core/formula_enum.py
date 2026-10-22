"""Bounded formula enumeration (Option B).

Enumerates ALL predicates φ of complexity ≤ k over the primitive signature
{∈, =, ¬, →, ∀} with the Separation variable `z` free, deduped up to
parameter/bound renaming with z PINNED (z is distinguished, so `z∈a` ≠ `a∈z`,
but `z∈a` = `z∈b` up to the universally-quantified parameter).

Uniform, convergent replacement for the hand-picked 12 Separation instances:
k → ∞ exhausts every φ (the full ZFC Separation schema). Soundness is automatic
— `separation_instance(φ)` is a valid axiom for ANY φ.

`complexity` is a self-consistent node count (NOT the formula's own `.size()`,
which is inconsistent across node types).
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .formula import (
    Formula, Membership, Equality, Not, Implies, ForAll,
    make_variable, make_constant,
)
from .schema import canonicalize_formula, schema_hash

_BOUND = "w"
_Z_PIN = make_constant("__Z_SEP__")


def complexity(f: Formula) -> int:
    t = type(f)
    if t is Membership or t is Equality:
        return 1
    if t is Not:
        return 1 + complexity(f.inner)
    if t is ForAll:
        return 1 + complexity(f.body)
    if t is Implies:
        return 1 + complexity(f.antecedent) + complexity(f.consequent)
    return 1


def _pin_key(f: Formula) -> str:
    """Dedup key: canonical form with z pinned (params/bound normalized)."""
    return schema_hash(canonicalize_formula(f.substitute("z", _Z_PIN)))


def _enumerate_by_complexity(max_k: int, terms: List[str]) -> List[Formula]:
    """All formulas with complexity ≤ max_k over {∈,=,¬,→,∀}.

    This is exact-complexity dynamic programming. It avoids the old saturation
    loop over the growing global pool, which repeatedly reconsidered the same
    lower-complexity pairs and became the practical bottleneck at larger k.
    """
    if max_k < 1:
        return []

    by_complexity: Dict[int, List[Formula]] = {i: [] for i in range(1, max_k + 1)}
    seen: Set[Formula] = set()

    def _add(f: Formula, c: int) -> None:
        # Structural dedup ONLY (z, params, bound all fixed) — params must stay
        # DISTINCT building blocks so predicates over two params (z∈a→z∈b) can
        # be built. Param-renaming collapse happens once, in the final output.
        if c > max_k:
            return
        if f in seen:
            return
        seen.add(f)
        by_complexity[c].append(f)

    pool = terms + [_BOUND]
    for a in pool:
        for b in pool:
            _add(Membership(make_variable(a), make_variable(b)), 1)
            _add(Equality(make_variable(a), make_variable(b)), 1)

    for c in range(2, max_k + 1):
        prev = by_complexity[c - 1]
        for f in prev:
            _add(Not(f), c)
            if _BOUND in f.free_variables():
                _add(ForAll(_BOUND, f), c)

        for left_c in range(1, c - 1):
            right_c = c - 1 - left_c
            for f in by_complexity[left_c]:
                for g in by_complexity[right_c]:
                    _add(Implies(f, g), c)

    out: List[Formula] = []
    for c in range(1, max_k + 1):
        out.extend(by_complexity[c])
    return out


def enumerate_predicates(max_size: int, params: Tuple[str, ...] = ("a", "b")) -> List[Formula]:
    """All φ (complexity ≤ max_size, signature {∈,=,¬,→,∀}) with z free, deduped
    up to parameter/bound renaming (z pinned)."""
    terms = ["z"] + list(params)
    out: List[Formula] = []
    seen: set = set()
    for f in _enumerate_by_complexity(max_size, terms):
        fv = f.free_variables()
        if "z" not in fv or _BOUND in fv:
            continue
        k = _pin_key(f)
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out
