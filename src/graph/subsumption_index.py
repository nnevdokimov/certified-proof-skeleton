"""Discrimination-tree subsumption index for the proof graph.

Replaces the earlier size + connective-signature pre-filter with a
**discrimination tree** (term index / flatterm trie), the standard ATP
technique (E, Vampire) for retrieving generalizations and instances of a
term in time proportional to the query depth times the number of
*surviving* candidates — not the whole population.  This removes the
O(n²) subsumption scan that blocked ``full``-mode generation at d ≥ 3
(see docs/PHASE_A_COMPLETION.md, docs/CLEAN_GRAPH_PLAN.md §2.4).

Soundness contract is unchanged: the tree only *narrows* candidates; the
exact :func:`core.schema.match` is still the final arbiter.  The tree
never drops a true subsumer/subsumed (no false negatives); it may surface
spurious candidates (e.g. non-linear ``P→P``), which ``match`` rejects.

Key insight — ``match`` (core/schema.py) has exactly **one** wildcard:
a ``PropVar`` on the *pattern* side.  Every other constructor must align
literally: bound-variable names, predicate names/arities, ``BigAnd``
length, and whole atomic FO/SO formulas (Membership, Equality, PredApp,
PredicateVar — no PropVars inside).  So:

  * stored PropVars collapse to a single ``STAR`` symbol in the tree;
  * atoms are opaque leaf tokens compared by structural ``==``;
  * ``find_subsumer``  = retrieve **generalizations** of the target
    (target PropVars are ground constants, only a tree STAR covers them);
  * ``find_subsumed_by`` = retrieve **instances** of the pattern
    (pattern PropVars are wildcards that skip a whole subterm in the tree).

The two retrievals over one shared tree are the classic discrimination-tree
generalization / instance algorithms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

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
)
from core.schema import match


# Cached types for fast `is`-checks.
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


# ── Flatterm tokens ──────────────────────────────────────────────────────
#
# A formula is flattened to a pre-order list of hashable tokens.  Each token
# carries a fixed arity = number of child subterms that immediately follow
# it in the flatterm.  ``STAR`` (arity 0) abstracts a stored PropVar; a
# concrete atom token (arity 0) carries the atom itself for exact ``==``.

Token = tuple
_STAR: Token = ("*",)  # sentinel: a stored PropVar / a query wildcard


# arity by tag — anything not listed is a leaf (arity 0).
_ARITY_1 = frozenset({"Not", "Box", "ForAll", "Exists", "ForAllPred", "ExistsPred"})
_ARITY_2 = frozenset({"->", "&", "|", "<>"})


def _arity(tok: Token) -> int:
    tag = tok[0]
    if tag in _ARITY_2:
        return 2
    if tag in _ARITY_1:
        return 1
    if tag == "BigAnd" or tag == "BigOr":
        return tok[1]
    # "*", "pv", "atom" → 0
    return 0


def _flatten(f: Formula, out: List[Token], star_for_propvar: bool) -> None:
    """Append the pre-order flatterm of ``f`` to ``out``.

    ``star_for_propvar`` controls how PropVars are emitted:
      * True  — as ``STAR`` (a wildcard / collapsed stored variable).  Used
        for insertion and for ``find_subsumed_by`` queries (pattern role).
      * False — as a ground ``("pv", name)`` constant.  Used for
        ``find_subsumer`` queries (target role), where a PropVar is literal.
    """
    t = type(f)
    if t is _PropVar:
        out.append(_STAR if star_for_propvar else ("pv", f.name))
        return
    if t is _Implies:
        out.append(("->",))
        _flatten(f.antecedent, out, star_for_propvar)
        _flatten(f.consequent, out, star_for_propvar)
        return
    if t is _And:
        out.append(("&",))
        _flatten(f.left, out, star_for_propvar)
        _flatten(f.right, out, star_for_propvar)
        return
    if t is _Or:
        out.append(("|",))
        _flatten(f.left, out, star_for_propvar)
        _flatten(f.right, out, star_for_propvar)
        return
    if t is _Iff:
        out.append(("<>",))
        _flatten(f.left, out, star_for_propvar)
        _flatten(f.right, out, star_for_propvar)
        return
    if t is _Not:
        out.append(("Not",))
        _flatten(f.inner, out, star_for_propvar)
        return
    if t is _Box:
        out.append(("Box",))
        _flatten(f.inner, out, star_for_propvar)
        return
    if t is _ForAll:
        out.append(("ForAll", f.variable))
        _flatten(f.body, out, star_for_propvar)
        return
    if t is _Exists:
        out.append(("Exists", f.variable))
        _flatten(f.body, out, star_for_propvar)
        return
    if t is _ForAllPred:
        out.append(("ForAllPred", f.pred_var, f.arity))
        _flatten(f.body, out, star_for_propvar)
        return
    if t is _ExistsPred:
        out.append(("ExistsPred", f.pred_var, f.arity))
        _flatten(f.body, out, star_for_propvar)
        return
    if t is _BigAnd:
        out.append(("BigAnd", len(f.formulas)))
        for sub in f.formulas:
            _flatten(sub, out, star_for_propvar)
        return
    if t is _BigOr:
        out.append(("BigOr", len(f.formulas)))
        for sub in f.formulas:
            _flatten(sub, out, star_for_propvar)
        return
    # Atomic leaves (Membership, Equality, PredApp, PredicateVar): no PropVars
    # inside, so ``match`` compares them via structural ``==``.  One opaque
    # token carrying the formula reproduces that exactly.
    out.append(("atom", f))


def _skip_subterm(flat: List[Token], i: int) -> int:
    """Return the index one past the complete subterm starting at ``flat[i]``."""
    remaining = 1
    while remaining:
        remaining += _arity(flat[i]) - 1
        i += 1
    return i


class _TrieNode:
    __slots__ = ("children", "ids", "max_size")

    def __init__(self) -> None:
        self.children: Dict[Token, "_TrieNode"] = {}
        self.ids: Set[int] = set()
        # Largest formula size anywhere in this subtree. Used to prune
        # ``find_subsumed_by`` branches that cannot hold a candidate ≥ the
        # pattern's size (φ ⊒ ψ ⟹ size(φ) ≤ size(ψ)). Maintained as a
        # monotone upper bound: bumped on insert, never lowered on remove,
        # so it stays sound (an over-estimate only costs a little pruning).
        self.max_size = 0


class _Entry:
    __slots__ = ("node_id", "formula", "size")

    def __init__(self, node_id: int, formula: Formula) -> None:
        self.node_id = node_id
        self.formula = formula
        self.size = formula.size()


class SubsumptionIndex:
    """Discrimination-tree index over canonical schemas.

    All queries assume their inputs are structurally canonical
    (see :func:`core.schema.canonicalize_formula`); the index does not
    re-canonicalize.  Callers that store non-canonical formulas will get
    spurious negatives.

    Public API (unchanged from the previous size+signature index):
    ``insert``, ``remove``, ``find_subsumer``, ``find_subsumed_by``,
    ``__len__``, ``__contains__``, ``all_node_ids``.
    """

    def __init__(self) -> None:
        self._entries: Dict[int, _Entry] = {}
        self._root = _TrieNode()

    # ── mutation ──────────────────────────────────────────────────────

    def insert(self, node_id: int, formula: Formula) -> None:
        if node_id in self._entries:
            return
        entry = _Entry(node_id, formula)
        self._entries[node_id] = entry
        size = entry.size
        flat: List[Token] = []
        _flatten(formula, flat, star_for_propvar=True)
        node = self._root
        if size > node.max_size:
            node.max_size = size
        for tok in flat:
            nxt = node.children.get(tok)
            if nxt is None:
                nxt = _TrieNode()
                node.children[tok] = nxt
            node = nxt
            if size > node.max_size:
                node.max_size = size
        node.ids.add(node_id)

    def remove(self, node_id: int) -> None:
        entry = self._entries.pop(node_id, None)
        if entry is None:
            return
        flat: List[Token] = []
        _flatten(entry.formula, flat, star_for_propvar=True)
        # Walk to the terminal, remembering the path to prune empty nodes.
        path: List[Tuple[_TrieNode, Token]] = []
        node = self._root
        for tok in flat:
            nxt = node.children.get(tok)
            if nxt is None:
                return  # should not happen; defensive
            path.append((node, tok))
            node = nxt
        node.ids.discard(node_id)
        # Prune now-empty subtrees bottom-up.
        if not node.ids and not node.children:
            for parent, tok in reversed(path):
                child = parent.children.get(tok)
                if child is not None and not child.ids and not child.children:
                    del parent.children[tok]
                else:
                    break

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, node_id: int) -> bool:
        return node_id in self._entries

    # ── retrieval primitives over the discrimination tree ─────────────

    def _retrieve_generalizations(self, qflat: List[Token]) -> List[int]:
        """Candidate ids of stored formulas that subsume the query.

        Stored STAR edges cover an entire query subterm; exact edges match
        one query token.  Query PropVars appear as ground ``("pv", …)``
        tokens — only a tree STAR can cover them, never a concrete edge.
        """
        out: List[int] = []
        n = len(qflat)
        root = self._root

        def rec(node: _TrieNode, i: int) -> None:
            if i == n:
                if node.ids:
                    out.extend(node.ids)
                return
            star = node.children.get(_STAR)
            if star is not None:
                rec(star, _skip_subterm(qflat, i))
            child = node.children.get(qflat[i])
            if child is not None:
                rec(child, i + 1)

        rec(root, 0)
        return out

    def _retrieve_instances(self, qflat: List[Token], min_size: int) -> List[int]:
        """Candidate ids of stored formulas subsumed by the query.

        A query STAR (pattern wildcard) matches any one subterm of a stored
        formula — we advance the tree past one complete subterm.  A concrete
        query token must follow the exact tree edge.  Branches whose subtree
        max size < ``min_size`` are pruned (no candidate there can be ≥ the
        pattern's size, so none can be subsumed by it).
        """
        out: List[int] = []
        n = len(qflat)

        def skip_tree(node: _TrieNode, remaining: int) -> List[_TrieNode]:
            # Nodes reached after consuming `remaining` complete subterms,
            # pruning subtrees that cannot hold a large-enough candidate.
            frontier = [(node, remaining)]
            done: List[_TrieNode] = []
            while frontier:
                nd, rem = frontier.pop()
                if rem == 0:
                    done.append(nd)
                    continue
                for tok, child in nd.children.items():
                    if child.max_size >= min_size:
                        frontier.append((child, rem - 1 + _arity(tok)))
            return done

        def rec(node: _TrieNode, i: int) -> None:
            if i == n:
                if node.ids:
                    out.extend(node.ids)
                return
            tok = qflat[i]
            if tok == _STAR:
                for endnode in skip_tree(node, 1):
                    rec(endnode, i + 1)
            else:
                child = node.children.get(tok)
                if child is not None and child.max_size >= min_size:
                    rec(child, i + 1)

        if self._root.max_size >= min_size:
            rec(self._root, 0)
        return out

    # ── queries ───────────────────────────────────────────────────────

    def find_subsumer(self, target: Formula) -> Optional[int]:
        """Return a node id whose formula subsumes ``target`` (or None).

        Deterministic: among confirmed subsumers, returns the minimal by
        ``(size, node_id)`` — matching the previous size-ascending scan, so
        the canonical ``full``-factor object is unchanged.
        """
        qflat: List[Token] = []
        _flatten(target, qflat, star_for_propvar=False)
        best_key: Optional[Tuple[int, int]] = None
        best_id: Optional[int] = None
        seen: Set[int] = set()
        for nid in self._retrieve_generalizations(qflat):
            if nid in seen:
                continue
            seen.add(nid)
            entry = self._entries.get(nid)
            if entry is None:
                continue
            if match(entry.formula, target) is not None:
                key = (entry.size, nid)
                if best_key is None or key < best_key:
                    best_key = key
                    best_id = nid
        return best_id

    def find_subsumed_by(self, pattern: Formula) -> List[int]:
        """Return all node ids whose formula is subsumed by ``pattern``.

        Sorted by ``(size, node_id)`` for deterministic retirement order.
        Entries strictly smaller than ``pattern`` cannot be instances and
        are skipped (size lower bound: φ ⊒ ψ ⟹ size(φ) ≤ size(ψ)).
        """
        pat_size = pattern.size()
        qflat: List[Token] = []
        _flatten(pattern, qflat, star_for_propvar=True)
        out: List[int] = []
        seen: Set[int] = set()
        for nid in self._retrieve_instances(qflat, pat_size):
            if nid in seen:
                continue
            seen.add(nid)
            entry = self._entries.get(nid)
            if entry is None or entry.size < pat_size:
                continue
            if match(pattern, entry.formula) is not None:
                out.append(nid)
        out.sort(key=lambda x: (self._entries[x].size, x))
        return out

    # ── introspection (for tests / metrics) ───────────────────────────

    def all_node_ids(self) -> Set[int]:
        return set(self._entries.keys())
