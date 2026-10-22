"""
Representation of first-order logic formulas.

Syntax of formulas:
- Atomic: p, t₁ ∈ t₂, t₁ = t₂
- Logical connectives: ¬φ, φ ∧ ψ, φ ∨ ψ, φ → ψ, φ ↔ ψ
- Modal operator: □φ
- Quantifiers: ∀x.φ, ∃x.φ

This module provides immutable, hashable formula classes with:
- Substitution (with proper variable capture avoidance)
- Free/bound variable computation
- Canonical hashing for α-equivalence
- Size computation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Set, FrozenSet, Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import logging

logger = logging.getLogger(__name__)


class FormulaType(Enum):
    """Types of formulas."""
    PROP_VAR = "prop"           # Propositional variable p, q, r...
    MEMBERSHIP = "mem"          # t₁ ∈ t₂
    EQUALITY = "eq"             # t₁ = t₂
    NOT = "not"                 # ¬φ
    AND = "and"                 # φ ∧ ψ
    OR = "or"                   # φ ∨ ψ
    IMPLIES = "imp"             # φ → ψ
    IFF = "iff"                 # φ ↔ ψ
    BOX = "box"                 # □φ
    FORALL = "forall"           # ∀x.φ
    EXISTS = "exists"           # ∃x.φ
    # Second-order logic (SOL)
    PRED_VAR = "pred_var"       # Second-order metavariable P, Q, F
    PRED_APP = "pred_app"       # Predicate application P(x), F(x,y)
    FORALL_PRED = "forall_pred" # ∀P.φ  (second-order universal)
    EXISTS_PRED = "exists_pred" # ∃P.φ  (second-order existential)
    # Infinitary logic L_{ω₁,ω}
    BIG_AND = "big_and"         # ⋀{φ₁, …, φₙ}
    BIG_OR = "big_or"           # ⋁{φ₁, …, φₙ}


@dataclass(frozen=True, slots=True)
class Term:
    """
    A term (object) in set theory.

    Terms can be:
    - Variable: x, y, z, ...
    - Constant: ∅ (empty set)
    - Skolem constant: sk0, sk1, ...

    Attributes:
        name: The name/identifier of the term
        is_variable: True if this is a variable, False if constant
    """
    name: str
    is_variable: bool = True

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.is_variable:
            return f"Var({self.name})"
        return f"Const({self.name})"

    def __hash__(self) -> int:
        return hash((self.name, self.is_variable))

    def substitute(self, var: str, replacement: 'Term') -> 'Term':
        """Substitute variable `var` with `replacement`."""
        if self.is_variable and self.name == var:
            return replacement
        return self

    def substitute_term(self, old_term: 'Term', new_term: 'Term') -> 'Term':
        """
        Substitute any term (variable or constant) with another term.

        Unlike substitute(), this works with constants too.

        Args:
            old_term: Term to replace (can be variable or constant)
            new_term: Replacement term

        Returns:
            New term with substitution applied
        """
        if self == old_term:
            return new_term
        return self

    def free_variables(self) -> Set[str]:
        """Get the set of free variables in this term."""
        if self.is_variable:
            return {self.name}
        return set()

    def constants(self) -> Set[str]:
        """Get the set of constants in this term."""
        if not self.is_variable:
            return {self.name}
        return set()

    def contains_term(self, term: 'Term') -> bool:
        """Check if this term equals or contains the given term."""
        return self == term


# Common constants
EMPTY_SET = Term("∅", is_variable=False)


def make_variable(name: str) -> Term:
    """Create a variable term."""
    return Term(name, is_variable=True)


def make_constant(name: str) -> Term:
    """Create a constant term."""
    return Term(name, is_variable=False)


class Formula(ABC):
    """
    Abstract base class for first-order logic formulas.

    All formula classes are immutable (frozen dataclasses) and hashable.
    """
    __slots__ = ('_size_cache', '_depth_cache')

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the formula."""
        pass

    @abstractmethod
    def substitute(self, var: str, term: Term) -> 'Formula':
        """
        Substitute a free variable with a term.

        Handles variable capture avoidance automatically.

        Args:
            var: Name of the variable to replace
            term: Term to substitute

        Returns:
            New formula with substitution applied
        """
        pass

    @abstractmethod
    def substitute_term(self, old_term: Term, new_term: Term) -> 'Formula':
        """
        Substitute any term (variable or constant) with another term.

        Unlike substitute(), this works with constants and any terms,
        not just variables by name.

        Args:
            old_term: Term to replace (can be variable or constant)
            new_term: Replacement term

        Returns:
            New formula with substitution applied
        """
        pass

    @abstractmethod
    def free_variables(self) -> FrozenSet[str]:
        """Get the set of free variables in this formula."""
        pass

    @abstractmethod
    def bound_variables(self) -> FrozenSet[str]:
        """Get the set of bound variables in this formula."""
        pass

    @abstractmethod
    def constants(self) -> FrozenSet[str]:
        """
        Get the set of constant names appearing in this formula.

        This includes Skolem constants (sk0, sk1, ...) and other constants.
        Used to check if a formula depends on Skolem constants.
        """
        pass

    def all_variables(self) -> FrozenSet[str]:
        """Get all variables (free and bound) in this formula."""
        return self.free_variables() | self.bound_variables()

    def is_closed(self) -> bool:
        """Check if this formula is closed (no free variables)."""
        return len(self.free_variables()) == 0

    def canonical_hash(self) -> str:
        """
        Compute a canonical hash for α-equivalence comparison.

        Two formulas have the same canonical hash iff they are
        α-equivalent (equal up to renaming of bound variables).

        Returns:
            MD5 hash of canonical string representation
        """
        canonical = self._to_canonical_string(counter=[0], env={})
        return hashlib.md5(canonical.encode()).hexdigest()

    @abstractmethod
    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        """
        Convert to canonical string for α-equivalence.

        Args:
            counter: Mutable counter for generating fresh variable names
            env: Mapping from original bound variable names to canonical names

        Returns:
            Canonical string representation
        """
        pass

    @property
    @abstractmethod
    def formula_type(self) -> FormulaType:
        """Get the type of this formula."""
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Compute the size of this formula (number of AST nodes).

        Used for filtering overly complex formulas.
        """
        pass

    @abstractmethod
    def depth(self) -> int:
        """
        Compute the depth of this formula (longest path in AST).
        """
        pass

    @staticmethod
    def _fresh_variable(used: Set[str], prefix: str = "_v") -> str:
        """
        Generate a fresh variable name not in `used`.

        Args:
            used: Set of variable names to avoid
            prefix: Prefix for the new variable name

        Returns:
            Fresh variable name
        """
        i = 0
        while f"{prefix}{i}" in used:
            i += 1
        return f"{prefix}{i}"


@dataclass(frozen=True, slots=True)
class Membership(Formula):
    """
    Membership formula: t₁ ∈ t₂

    The fundamental atomic formula in set theory.
    """
    left: Term   # Element
    right: Term  # Set

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.MEMBERSHIP

    def __str__(self) -> str:
        return f"({self.left} ∈ {self.right})"

    def substitute(self, var: str, term: Term) -> Formula:
        return Membership(
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Membership(
            self.left.substitute_term(old_term, new_term),
            self.right.substitute_term(old_term, new_term)
        )

    def free_variables(self) -> FrozenSet[str]:
        return frozenset(self.left.free_variables() | self.right.free_variables())

    def bound_variables(self) -> FrozenSet[str]:
        return frozenset()

    def constants(self) -> FrozenSet[str]:
        return frozenset(self.left.constants() | self.right.constants())

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        left_name = env.get(self.left.name, self.left.name) if self.left.is_variable else self.left.name
        right_name = env.get(self.right.name, self.right.name) if self.right.is_variable else self.right.name
        return f"MEM({left_name},{right_name})"

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class Equality(Formula):
    """
    Equality formula: t₁ = t₂
    """
    left: Term
    right: Term

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.EQUALITY

    def __str__(self) -> str:
        return f"({self.left} = {self.right})"

    def substitute(self, var: str, term: Term) -> Formula:
        return Equality(
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Equality(
            self.left.substitute_term(old_term, new_term),
            self.right.substitute_term(old_term, new_term)
        )

    def free_variables(self) -> FrozenSet[str]:
        return frozenset(self.left.free_variables() | self.right.free_variables())

    def bound_variables(self) -> FrozenSet[str]:
        return frozenset()

    def constants(self) -> FrozenSet[str]:
        return frozenset(self.left.constants() | self.right.constants())

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        left_name = env.get(self.left.name, self.left.name) if self.left.is_variable else self.left.name
        right_name = env.get(self.right.name, self.right.name) if self.right.is_variable else self.right.name
        return f"EQ({left_name},{right_name})"

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class PropVar(Formula):
    """
    Propositional variable atom: p, q, r, ...

    This is a term-free atomic formula used by propositional-only runs.
    """
    name: str

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.PROP_VAR

    def __str__(self) -> str:
        return self.name

    def substitute(self, var: str, term: Term) -> Formula:
        # Propositional variables do not contain first-order terms.
        return self

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        # Propositional variables do not contain first-order terms.
        return self

    def free_variables(self) -> FrozenSet[str]:
        return frozenset()

    def bound_variables(self) -> FrozenSet[str]:
        return frozenset()

    def constants(self) -> FrozenSet[str]:
        return frozenset()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        return f"PROP({self.name})"

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class Not(Formula):
    """
    Negation: ¬φ
    """
    inner: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.NOT

    def __str__(self) -> str:
        return f"¬{self.inner}"

    def substitute(self, var: str, term: Term) -> Formula:
        return Not(self.inner.substitute(var, term))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Not(self.inner.substitute_term(old_term, new_term))

    def free_variables(self) -> FrozenSet[str]:
        return self.inner.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.inner.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.inner.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        return f"NOT({self.inner._to_canonical_string(counter, env)})"

    def size(self) -> int:
        return 1 + self.inner.size()

    def depth(self) -> int:
        return 1 + self.inner.depth()


@dataclass(frozen=True, slots=True)
class Box(Formula):
    """
    Modal necessity operator: □φ
    """
    inner: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.BOX

    def __str__(self) -> str:
        return f"□{self.inner}"

    def substitute(self, var: str, term: Term) -> Formula:
        return Box(self.inner.substitute(var, term))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Box(self.inner.substitute_term(old_term, new_term))

    def free_variables(self) -> FrozenSet[str]:
        return self.inner.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.inner.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.inner.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        return f"BOX({self.inner._to_canonical_string(counter, env)})"

    def size(self) -> int:
        return 1 + self.inner.size()

    def depth(self) -> int:
        return 1 + self.inner.depth()


@dataclass(frozen=True, slots=True)
class And(Formula):
    """
    Conjunction: φ ∧ ψ
    """
    left: Formula
    right: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.AND

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def substitute(self, var: str, term: Term) -> Formula:
        return And(
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return And(
            self.left.substitute_term(old_term, new_term),
            self.right.substitute_term(old_term, new_term)
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.left.free_variables() | self.right.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.left.bound_variables() | self.right.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.left.constants() | self.right.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        left_str = self.left._to_canonical_string(counter, env)
        right_str = self.right._to_canonical_string(counter, env)
        return f"AND({left_str},{right_str})"

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


@dataclass(frozen=True, slots=True)
class Or(Formula):
    """
    Disjunction: φ ∨ ψ
    """
    left: Formula
    right: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.OR

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def substitute(self, var: str, term: Term) -> Formula:
        return Or(
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Or(
            self.left.substitute_term(old_term, new_term),
            self.right.substitute_term(old_term, new_term)
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.left.free_variables() | self.right.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.left.bound_variables() | self.right.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.left.constants() | self.right.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        left_str = self.left._to_canonical_string(counter, env)
        right_str = self.right._to_canonical_string(counter, env)
        return f"OR({left_str},{right_str})"

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


@dataclass(frozen=True, slots=True)
class Implies(Formula):
    """
    Implication: φ → ψ
    """
    antecedent: Formula
    consequent: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.IMPLIES

    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"

    def substitute(self, var: str, term: Term) -> Formula:
        return Implies(
            self.antecedent.substitute(var, term),
            self.consequent.substitute(var, term)
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Implies(
            self.antecedent.substitute_term(old_term, new_term),
            self.consequent.substitute_term(old_term, new_term)
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.antecedent.free_variables() | self.consequent.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.antecedent.bound_variables() | self.consequent.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.antecedent.constants() | self.consequent.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        ant_str = self.antecedent._to_canonical_string(counter, env)
        con_str = self.consequent._to_canonical_string(counter, env)
        # Implication is NOT commutative - order matters
        return f"IMP({ant_str},{con_str})"

    def size(self) -> int:
        return 1 + self.antecedent.size() + self.consequent.size()

    def depth(self) -> int:
        return 1 + max(self.antecedent.depth(), self.consequent.depth())


@dataclass(frozen=True, slots=True)
class Iff(Formula):
    """
    Biconditional: φ ↔ ψ
    """
    left: Formula
    right: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.IFF

    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"

    def substitute(self, var: str, term: Term) -> Formula:
        return Iff(
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return Iff(
            self.left.substitute_term(old_term, new_term),
            self.right.substitute_term(old_term, new_term)
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.left.free_variables() | self.right.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.left.bound_variables() | self.right.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.left.constants() | self.right.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        left_str = self.left._to_canonical_string(counter, env)
        right_str = self.right._to_canonical_string(counter, env)
        return f"IFF({left_str},{right_str})"

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


@dataclass(frozen=True, slots=True)
class ForAll(Formula):
    """
    Universal quantifier: ∀x.φ
    """
    variable: str
    body: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.FORALL

    def __str__(self) -> str:
        return f"∀{self.variable}.{self.body}"

    def substitute(self, var: str, term: Term) -> Formula:
        if var == self.variable:
            # Variable is bound, don't substitute
            return self

        if self.variable in term.free_variables():
            # Need to rename bound variable to avoid capture
            all_vars = (
                self.body.free_variables() |
                term.free_variables() |
                {self.variable, var}
            )
            new_var = Formula._fresh_variable(all_vars)
            renamed_body = self.body.substitute(self.variable, Term(new_var))
            return ForAll(new_var, renamed_body.substitute(var, term))

        return ForAll(self.variable, self.body.substitute(var, term))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        # If old_term is this bound variable, don't substitute inside
        if old_term.is_variable and old_term.name == self.variable:
            return self

        # Check for variable capture: if new_term contains the bound variable
        if self.variable in new_term.free_variables():
            # Need to rename bound variable to avoid capture
            all_vars = (
                self.body.free_variables() |
                new_term.free_variables() |
                {self.variable}
            )
            fresh_var = Formula._fresh_variable(all_vars)
            renamed_body = self.body.substitute(self.variable, Term(fresh_var))
            return ForAll(fresh_var, renamed_body.substitute_term(old_term, new_term))

        return ForAll(self.variable, self.body.substitute_term(old_term, new_term))

    def free_variables(self) -> FrozenSet[str]:
        return self.body.free_variables() - {self.variable}

    def bound_variables(self) -> FrozenSet[str]:
        return self.body.bound_variables() | {self.variable}

    def constants(self) -> FrozenSet[str]:
        return self.body.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        # Assign canonical name to bound variable
        canonical_name = f"v{counter[0]}"
        counter[0] += 1
        new_env = {**env, self.variable: canonical_name}
        body_str = self.body._to_canonical_string(counter, new_env)
        return f"FORALL({canonical_name},{body_str})"

    def size(self) -> int:
        return 1 + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()


@dataclass(frozen=True, slots=True)
class Exists(Formula):
    """
    Existential quantifier: ∃x.φ
    """
    variable: str
    body: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.EXISTS

    def __str__(self) -> str:
        return f"∃{self.variable}.{self.body}"

    def substitute(self, var: str, term: Term) -> Formula:
        if var == self.variable:
            # Variable is bound, don't substitute
            return self

        if self.variable in term.free_variables():
            # Need to rename bound variable to avoid capture
            all_vars = (
                self.body.free_variables() |
                term.free_variables() |
                {self.variable, var}
            )
            new_var = Formula._fresh_variable(all_vars)
            renamed_body = self.body.substitute(self.variable, Term(new_var))
            return Exists(new_var, renamed_body.substitute(var, term))

        return Exists(self.variable, self.body.substitute(var, term))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        # If old_term is this bound variable, don't substitute inside
        if old_term.is_variable and old_term.name == self.variable:
            return self

        # Check for variable capture: if new_term contains the bound variable
        if self.variable in new_term.free_variables():
            # Need to rename bound variable to avoid capture
            all_vars = (
                self.body.free_variables() |
                new_term.free_variables() |
                {self.variable}
            )
            fresh_var = Formula._fresh_variable(all_vars)
            renamed_body = self.body.substitute(self.variable, Term(fresh_var))
            return Exists(fresh_var, renamed_body.substitute_term(old_term, new_term))

        return Exists(self.variable, self.body.substitute_term(old_term, new_term))

    def free_variables(self) -> FrozenSet[str]:
        return self.body.free_variables() - {self.variable}

    def bound_variables(self) -> FrozenSet[str]:
        return self.body.bound_variables() | {self.variable}

    def constants(self) -> FrozenSet[str]:
        return self.body.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        # Assign canonical name to bound variable
        canonical_name = f"v{counter[0]}"
        counter[0] += 1
        new_env = {**env, self.variable: canonical_name}
        body_str = self.body._to_canonical_string(counter, new_env)
        return f"EXISTS({canonical_name},{body_str})"

    def size(self) -> int:
        return 1 + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()


# ─────────────────────────────────────────────────────────────────────────────
#  Second-order logic formula types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PredicateVar(Formula):
    """
    Second-order metavariable: P, Q, F.

    Used as a schema-level metavariable analogous to PropVar but at second
    order.  In concrete axioms it is always bound under ForAllPred/ExistsPred.
    When free, it acts as a unification variable in the schema engine.

    Attributes:
        name:  Identifier (e.g. "P", "F", "P0").
        arity: Number of first-order arguments (1 = unary, 2 = binary).
    """
    name: str
    arity: int

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.PRED_VAR

    def __str__(self) -> str:
        return self.name

    def substitute(self, var: str, term: Term) -> Formula:
        return self  # no first-order terms inside

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return self

    def free_variables(self) -> FrozenSet[str]:
        return frozenset()

    def bound_variables(self) -> FrozenSet[str]:
        return frozenset()

    def constants(self) -> FrozenSet[str]:
        return frozenset()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        return f"PVAR({self.name},{self.arity})"

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class PredApp(Formula):
    """
    Predicate application: P(t₁, …, tₙ).

    Represents a second-order predicate variable applied to first-order terms.
    `pred_var_name` is the name of the predicate variable (string); the
    variable may be bound by a surrounding ForAllPred/ExistsPred.

    Attributes:
        pred_var_name: Name of the predicate variable (e.g. "P").
        args:          Tuple of first-order Term arguments.
    """
    pred_var_name: str
    args: Tuple[Term, ...]

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.PRED_APP

    def __str__(self) -> str:
        arg_str = ", ".join(str(a) for a in self.args)
        return f"{self.pred_var_name}({arg_str})"

    def substitute(self, var: str, term: Term) -> Formula:
        return PredApp(
            self.pred_var_name,
            tuple(a.substitute(var, term) for a in self.args),
        )

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return PredApp(
            self.pred_var_name,
            tuple(a.substitute_term(old_term, new_term) for a in self.args),
        )

    def free_variables(self) -> FrozenSet[str]:
        fv: Set[str] = set()
        for a in self.args:
            fv.update(a.free_variables())
        return frozenset(fv)

    def bound_variables(self) -> FrozenSet[str]:
        return frozenset()

    def constants(self) -> FrozenSet[str]:
        cs: Set[str] = set()
        for a in self.args:
            cs.update(a.constants())
        return frozenset(cs)

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        arg_strs = ",".join(
            env.get(a.name, a.name) if a.is_variable else a.name
            for a in self.args
        )
        return f"PAPP({self.pred_var_name},{arg_strs})"

    def size(self) -> int:
        return 1 + len(self.args)

    def depth(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class ForAllPred(Formula):
    """
    Second-order universal quantifier: ∀P.φ

    Binds predicate variable `pred_var` of given `arity` in `body`.

    Attributes:
        pred_var: Name of the bound predicate variable (e.g. "P").
        arity:    Arity of P in the body.
        body:     Formula in which P is bound.
    """
    pred_var: str
    arity: int
    body: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.FORALL_PRED

    def __str__(self) -> str:
        return f"∀{self.pred_var}^{self.arity}.{self.body}"

    # First-order substitution: pred_var is second-order, no conflict.
    def substitute(self, var: str, term: Term) -> Formula:
        return ForAllPred(self.pred_var, self.arity, self.body.substitute(var, term))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return ForAllPred(self.pred_var, self.arity, self.body.substitute_term(old_term, new_term))

    def free_variables(self) -> FrozenSet[str]:
        return self.body.free_variables()  # pred_var is second-order, not in FV

    def bound_variables(self) -> FrozenSet[str]:
        return self.body.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.body.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        canonical_name = f"P{counter[0]}"
        counter[0] += 1
        body_str = self.body._to_canonical_string(counter, env)
        return f"FALLP({canonical_name},{self.arity},{body_str})"

    def size(self) -> int:
        return 1 + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()


@dataclass(frozen=True, slots=True)
class ExistsPred(Formula):
    """
    Second-order existential quantifier: ∃P.φ

    Attributes:
        pred_var: Name of the bound predicate variable.
        arity:    Arity of P in the body.
        body:     Formula in which P is bound.
    """
    pred_var: str
    arity: int
    body: Formula

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.EXISTS_PRED

    def __str__(self) -> str:
        return f"∃{self.pred_var}^{self.arity}.{self.body}"

    def substitute(self, var: str, term: Term) -> Formula:
        return ExistsPred(self.pred_var, self.arity, self.body.substitute(var, term))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return ExistsPred(self.pred_var, self.arity, self.body.substitute_term(old_term, new_term))

    def free_variables(self) -> FrozenSet[str]:
        return self.body.free_variables()

    def bound_variables(self) -> FrozenSet[str]:
        return self.body.bound_variables()

    def constants(self) -> FrozenSet[str]:
        return self.body.constants()

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        canonical_name = f"P{counter[0]}"
        counter[0] += 1
        body_str = self.body._to_canonical_string(counter, env)
        return f"EXISTSP({canonical_name},{self.arity},{body_str})"

    def size(self) -> int:
        return 1 + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()


# ─────────────────────────────────────────────────────────────────────────────
#  Infinitary logic formula types  (L_{ω₁,ω})
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BigAnd(Formula):
    """
    Countable (finite approximation) conjunction: ⋀{φ₁, …, φₙ}

    Used to represent L_{ω₁,ω} conjunctions.
    In practice len(formulas) is bounded (≤ 10) for AST size control.

    Attributes:
        formulas: Non-empty tuple of conjuncts.
    """
    formulas: Tuple[Formula, ...]

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.BIG_AND

    def __str__(self) -> str:
        return "⋀(" + ", ".join(str(f) for f in self.formulas) + ")"

    def substitute(self, var: str, term: Term) -> Formula:
        return BigAnd(tuple(f.substitute(var, term) for f in self.formulas))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return BigAnd(tuple(f.substitute_term(old_term, new_term) for f in self.formulas))

    def free_variables(self) -> FrozenSet[str]:
        fv: Set[str] = set()
        for f in self.formulas:
            fv.update(f.free_variables())
        return frozenset(fv)

    def bound_variables(self) -> FrozenSet[str]:
        bv: Set[str] = set()
        for f in self.formulas:
            bv.update(f.bound_variables())
        return frozenset(bv)

    def constants(self) -> FrozenSet[str]:
        cs: Set[str] = set()
        for f in self.formulas:
            cs.update(f.constants())
        return frozenset(cs)

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        parts = [f._to_canonical_string(counter, env) for f in self.formulas]
        return "BIGAND(" + ",".join(parts) + ")"

    def size(self) -> int:
        return 1 + sum(f.size() for f in self.formulas)

    def depth(self) -> int:
        return 1 + max((f.depth() for f in self.formulas), default=0)


@dataclass(frozen=True, slots=True)
class BigOr(Formula):
    """
    Countable (finite approximation) disjunction: ⋁{φ₁, …, φₙ}

    Attributes:
        formulas: Non-empty tuple of disjuncts.
    """
    formulas: Tuple[Formula, ...]

    @property
    def formula_type(self) -> FormulaType:
        return FormulaType.BIG_OR

    def __str__(self) -> str:
        return "⋁(" + ", ".join(str(f) for f in self.formulas) + ")"

    def substitute(self, var: str, term: Term) -> Formula:
        return BigOr(tuple(f.substitute(var, term) for f in self.formulas))

    def substitute_term(self, old_term: Term, new_term: Term) -> Formula:
        return BigOr(tuple(f.substitute_term(old_term, new_term) for f in self.formulas))

    def free_variables(self) -> FrozenSet[str]:
        fv: Set[str] = set()
        for f in self.formulas:
            fv.update(f.free_variables())
        return frozenset(fv)

    def bound_variables(self) -> FrozenSet[str]:
        bv: Set[str] = set()
        for f in self.formulas:
            bv.update(f.bound_variables())
        return frozenset(bv)

    def constants(self) -> FrozenSet[str]:
        cs: Set[str] = set()
        for f in self.formulas:
            cs.update(f.constants())
        return frozenset(cs)

    def _to_canonical_string(self, counter: List[int], env: Dict[str, str]) -> str:
        parts = [f._to_canonical_string(counter, env) for f in self.formulas]
        return "BIGOR(" + ",".join(parts) + ")"

    def size(self) -> int:
        return 1 + sum(f.size() for f in self.formulas)

    def depth(self) -> int:
        return 1 + max((f.depth() for f in self.formulas), default=0)


# Type alias for any formula
FormulaUnion = Union[
    PropVar,
    Membership,
    Equality,
    Not,
    Box,
    And,
    Or,
    Implies,
    Iff,
    ForAll,
    Exists,
    PredicateVar,
    PredApp,
    ForAllPred,
    ExistsPred,
    BigAnd,
    BigOr,
]


# ─────────────────────────────────────────────────────────────────────────
# Lazy size()/depth() caching.
#
# Each Formula instance reserves two slots (_size_cache, _depth_cache) via
# Formula.__slots__. The wrappers below replace the per-subclass size/depth
# methods with cached variants that compute lazily on first access. Cache
# is per-instance — id-stable for the formula's lifetime, no external dict.
# ─────────────────────────────────────────────────────────────────────────

def _install_size_depth_cache(cls):
    raw_size = cls.size
    def cached_size(self, _raw=raw_size):
        try:
            return self._size_cache
        except AttributeError:
            val = _raw(self)
            object.__setattr__(self, '_size_cache', val)
            return val
    cls.size = cached_size

    raw_depth = cls.depth
    def cached_depth(self, _raw=raw_depth):
        try:
            return self._depth_cache
        except AttributeError:
            val = _raw(self)
            object.__setattr__(self, '_depth_cache', val)
            return val
    cls.depth = cached_depth
    return cls


for _cls in (
    PropVar,
    Membership,
    Equality,
    Not,
    Box,
    And,
    Or,
    Implies,
    Iff,
    ForAll,
    Exists,
    PredicateVar,
    PredApp,
    ForAllPred,
    ExistsPred,
    BigAnd,
    BigOr,
):
    _install_size_depth_cache(_cls)
