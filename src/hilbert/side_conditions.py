"""
Side conditions for Hilbert calculus axiom schemas.

Implements the crucial checks for:
- A4: "t is free for x in φ" (term substitutability)
- A5: "x is not free in φ" (variable restriction)
"""

import logging
from typing import Set, FrozenSet

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.formula import (
    Formula, Term, PropVar, Membership, Equality,
    Not, Box, And, Or, Implies, Iff, ForAll, Exists,
    PredicateVar, PredApp, ForAllPred, ExistsPred,
    BigAnd, BigOr,
)

logger = logging.getLogger(__name__)


def is_free_for(term: Term, var: str, formula: Formula) -> bool:
    """
    Check if a term t is free for substitution of variable x in formula φ.

    A term t is FREE FOR x in φ if and only if no free variable in t
    becomes bound when t is substituted for x in φ.

    Formally: t is free for x in φ iff for every variable y occurring in t,
    no subformula of φ of the form ∀y.ψ or ∃y.ψ contains a free occurrence of x.

    This is the side condition for axiom schema A4: ∀x.φ(x) → φ(t)

    Args:
        term: The term t to substitute
        var: The variable x to be replaced
        formula: The formula φ

    Returns:
        True if t is free for x in φ

    Examples:
        >>> x, y, z = make_variable('x'), make_variable('y'), make_variable('z')
        >>> # y is free for x in (x ∈ z)
        >>> is_free_for(y, 'x', Membership(x, z))
        True
        >>> # y is NOT free for x in ∀y.(x ∈ y)
        >>> is_free_for(y, 'x', ForAll('y', Membership(x, y)))
        False
    """
    # If x doesn't occur free in φ, any term is free for x
    if var not in formula.free_variables():
        return True

    # Get free variables of the term
    if term.is_variable:
        term_free_vars = frozenset([term.name])
    else:
        # Constants have no free variables
        term_free_vars = frozenset()

    # If term has no free variables (is a constant or ground term),
    # it's always free for any variable
    if not term_free_vars:
        return True

    # Check if any variable in term would be captured
    return not _would_capture_variable(term_free_vars, var, formula)


def _would_capture_variable(term_vars: FrozenSet[str], var: str, formula: Formula) -> bool:
    """
    Check if substituting a term with given free variables for var in formula
    would result in variable capture.

    Args:
        term_vars: Free variables in the term being substituted
        var: Variable being replaced
        formula: Formula to substitute into

    Returns:
        True if capture would occur, False otherwise
    """
    if isinstance(formula, (PropVar, Membership, Equality, PredicateVar, PredApp)):
        # Atomic / metavariable forms: no FO quantifiers inside, no capture.
        # PropVar/PredicateVar are metavariables with no FO content.
        # PredApp has FO term arguments but no quantifier scope.
        return False

    elif isinstance(formula, Not):
        return _would_capture_variable(term_vars, var, formula.inner)

    elif isinstance(formula, Box):
        return _would_capture_variable(term_vars, var, formula.inner)

    elif isinstance(formula, (And, Or, Iff)):
        return (_would_capture_variable(term_vars, var, formula.left) or
                _would_capture_variable(term_vars, var, formula.right))

    elif isinstance(formula, Implies):
        return (_would_capture_variable(term_vars, var, formula.antecedent) or
                _would_capture_variable(term_vars, var, formula.consequent))

    elif isinstance(formula, (ForAll, Exists)):
        bound_var = formula.variable
        body = formula.body

        # If var is the bound variable, no free occurrences below
        if var == bound_var:
            return False

        # If var occurs free in the body
        if var in body.free_variables():
            # And the bound variable is in our term's free variables
            if bound_var in term_vars:
                # CAPTURE! The term's variable would become bound
                logger.debug(
                    f"Variable capture detected: {bound_var} in term "
                    f"would be captured by quantifier ∀{bound_var} or ∃{bound_var}"
                )
                return True

        # Recurse into body
        return _would_capture_variable(term_vars, var, body)

    elif isinstance(formula, (ForAllPred, ExistsPred)):
        # Second-order quantifiers bind a *predicate* variable, not a
        # first-order variable. Therefore they cannot capture a FO term variable.
        # Recurse into the body.
        return _would_capture_variable(term_vars, var, formula.body)

    elif isinstance(formula, (BigAnd, BigOr)):
        # Infinitary connectives: capture if any subformula captures.
        return any(
            _would_capture_variable(term_vars, var, sub)
            for sub in formula.formulas
        )

    else:
        logger.warning(f"Unknown formula type: {type(formula).__name__}")
        # Fail-safe: assume capture is possible for unknown types so callers
        # err on the conservative side rather than admit unsound substitutions.
        return True


def is_not_free_in(var: str, formula: Formula) -> bool:
    """
    Check if variable x is NOT free in formula φ.

    This is the side condition for axiom schema A5:
    ∀x.(φ → ψ) → (φ → ∀x.ψ)  requires x not free in φ

    Args:
        var: The variable x
        formula: The formula φ

    Returns:
        True if x is not free in φ

    Examples:
        >>> x, y = make_variable('x'), make_variable('y')
        >>> # x is not free in (y ∈ y)
        >>> is_not_free_in('x', Membership(y, y))
        True
        >>> # x IS free in (x ∈ y)
        >>> is_not_free_in('x', Membership(x, y))
        False
    """
    return var not in formula.free_variables()


def get_binding_context(formula: Formula, var: str) -> Set[str]:
    """
    Get the set of variables that bind var in the given formula.

    This helps understand the binding structure when analyzing
    substitution correctness.

    Args:
        formula: The formula to analyze
        var: The variable to track

    Returns:
        Set of quantified variables that have var free in their scope
    """
    result = set()
    _collect_binding_context(formula, var, result)
    return result


def _collect_binding_context(formula: Formula, var: str, bindings: Set[str]) -> None:
    """Helper to collect binding context."""
    if isinstance(formula, (PropVar, Membership, Equality)):
        return

    elif isinstance(formula, Not):
        _collect_binding_context(formula.inner, var, bindings)

    elif isinstance(formula, Box):
        _collect_binding_context(formula.inner, var, bindings)

    elif isinstance(formula, (And, Or, Iff)):
        _collect_binding_context(formula.left, var, bindings)
        _collect_binding_context(formula.right, var, bindings)

    elif isinstance(formula, Implies):
        _collect_binding_context(formula.antecedent, var, bindings)
        _collect_binding_context(formula.consequent, var, bindings)

    elif isinstance(formula, (ForAll, Exists)):
        bound_var = formula.variable
        body = formula.body

        if var != bound_var and var in body.free_variables():
            # var is free under this quantifier
            bindings.add(bound_var)

        if var != bound_var:
            _collect_binding_context(body, var, bindings)


def check_substitution_safety(term: Term, var: str, formula: Formula) -> dict:
    """
    Detailed check of substitution safety with diagnostic information.

    Args:
        term: Term to substitute
        var: Variable to replace
        formula: Target formula

    Returns:
        Dictionary with:
        - 'safe': bool - whether substitution is safe
        - 'var_free_in_formula': bool - whether var is free in formula
        - 'term_free_vars': set - free variables in term
        - 'binding_context': set - variables that bind var
        - 'conflicts': set - variables that would be captured
    """
    var_free = var in formula.free_variables()

    if term.is_variable:
        term_free_vars = {term.name}
    else:
        term_free_vars = set()

    binding_ctx = get_binding_context(formula, var)
    conflicts = term_free_vars & binding_ctx

    return {
        'safe': len(conflicts) == 0 or not var_free,
        'var_free_in_formula': var_free,
        'term_free_vars': term_free_vars,
        'binding_context': binding_ctx,
        'conflicts': conflicts,
    }
