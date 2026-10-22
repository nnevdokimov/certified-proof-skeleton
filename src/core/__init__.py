"""
Core module for ZFC Proof Graph.

Contains base data structures:
- Formula representations (first-order logic)
- ZFC axioms
- Schema operations
"""

from .formula import (
    FormulaType,
    Term,
    Formula,
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
    EMPTY_SET,
    make_variable,
    make_constant,
)

from .axioms import ZFCAxioms

from .schema import (
    normalize_schema,
    schema_hash,
    schema_mp,
)

from .primitives import (
    to_primitive_form,
    is_primitive,
)

__all__ = [
    # Formula types
    'FormulaType',
    'Term',
    'Formula',
    'PropVar',
    'Membership',
    'Equality',
    'Not',
    'Box',
    'And',
    'Or',
    'Implies',
    'Iff',
    'ForAll',
    'Exists',
    'EMPTY_SET',
    'make_variable',
    'make_constant',
    # Axioms
    'ZFCAxioms',
    # Schema
    'normalize_schema',
    'schema_hash',
    'schema_mp',
    # Primitives
    'to_primitive_form',
    'is_primitive',
]
