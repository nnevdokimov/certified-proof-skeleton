"""
Conversion of formulas to primitive form.

The primitive signature is {→, ¬, ∀, =, ∈}.
All other connectives are abbreviations:

    ∃x.φ   :=  ¬∀x.¬φ
    φ ∧ ψ  :=  ¬(φ → ¬ψ)
    φ ∨ ψ  :=  ¬φ → ψ
    φ ↔ ψ  :=  ¬((φ → ψ) → ¬(ψ → φ))

These match the definitions used in Metamath set.mm (df-an, df-or, df-bi, df-ex).
"""

from .formula import (
    Formula, Term, PropVar, Membership, Equality,
    Not, Box, And, Or, Implies, Iff, ForAll, Exists,
    PredicateVar, PredApp, ForAllPred, ExistsPred,
    BigAnd, BigOr,
)


def to_primitive_form(formula: Formula) -> Formula:
    """Recursively expand all abbreviations to primitive connectives {→, ¬, ∀, =, ∈}.

    Handles: And, Or, Iff, Exists → expanded into Implies, Not, ForAll.
    Atomic formulas (Membership, Equality, PropVar) and already-primitive
    connectives (Not, Implies, ForAll) are preserved (but recursed into).

    Args:
        formula: Any Formula object.

    Returns:
        Equivalent formula using only {Implies, Not, ForAll, Membership, Equality, PropVar}.
    """
    if isinstance(formula, (Membership, Equality, PropVar)):
        return formula

    if isinstance(formula, Not):
        return Not(to_primitive_form(formula.inner))

    if isinstance(formula, Implies):
        return Implies(
            to_primitive_form(formula.antecedent),
            to_primitive_form(formula.consequent),
        )

    if isinstance(formula, ForAll):
        return ForAll(formula.variable, to_primitive_form(formula.body))

    # ∃x.φ  :=  ¬∀x.¬φ
    if isinstance(formula, Exists):
        return Not(ForAll(formula.variable, Not(to_primitive_form(formula.body))))

    # φ ∧ ψ  :=  ¬(φ → ¬ψ)
    if isinstance(formula, And):
        left = to_primitive_form(formula.left)
        right = to_primitive_form(formula.right)
        return Not(Implies(left, Not(right)))

    # φ ∨ ψ  :=  ¬φ → ψ
    if isinstance(formula, Or):
        left = to_primitive_form(formula.left)
        right = to_primitive_form(formula.right)
        return Implies(Not(left), right)

    # φ ↔ ψ  :=  ¬((φ → ψ) → ¬(ψ → φ))
    if isinstance(formula, Iff):
        left = to_primitive_form(formula.left)
        right = to_primitive_form(formula.right)
        return Not(Implies(Implies(left, right), Not(Implies(right, left))))

    # Box (modal) — not part of ZFC, but recurse for completeness
    if isinstance(formula, Box):
        return Box(to_primitive_form(formula.inner))

    # Second-order quantifiers — recurse into body, preserve SOL structure
    if isinstance(formula, ForAllPred):
        return ForAllPred(formula.pred_var, formula.arity,
                          to_primitive_form(formula.body))

    if isinstance(formula, ExistsPred):
        return ExistsPred(formula.pred_var, formula.arity,
                          to_primitive_form(formula.body))

    # PredApp and PredicateVar are atomic — no connectives inside
    if isinstance(formula, (PredApp, PredicateVar)):
        return formula

    # Infinitary connectives — recurse into each sub-formula
    if isinstance(formula, BigAnd):
        return BigAnd(tuple(to_primitive_form(f) for f in formula.formulas))

    if isinstance(formula, BigOr):
        return BigOr(tuple(to_primitive_form(f) for f in formula.formulas))

    return formula


def is_primitive(formula: Formula) -> bool:
    """Check that a formula contains only primitive connectives.

    Primitive: Membership, Equality, PropVar, Not, Implies, ForAll.
    Extended primitive: ForAllPred, ExistsPred, PredApp, PredicateVar, BigAnd, BigOr.
    Non-primitive: And, Or, Iff, Exists.
    """
    if isinstance(formula, (Membership, Equality, PropVar, PredicateVar)):
        return True

    if isinstance(formula, Not):
        return is_primitive(formula.inner)

    if isinstance(formula, Implies):
        return is_primitive(formula.antecedent) and is_primitive(formula.consequent)

    if isinstance(formula, ForAll):
        return is_primitive(formula.body)

    if isinstance(formula, Box):
        return is_primitive(formula.inner)

    if isinstance(formula, (ForAllPred, ExistsPred)):
        return is_primitive(formula.body)

    if isinstance(formula, PredApp):
        return True  # atomic at FO level

    if isinstance(formula, (BigAnd, BigOr)):
        return all(is_primitive(f) for f in formula.formulas)

    # And, Or, Iff, Exists are non-primitive
    return False
