"""
Hilbert-style proof calculus module.

Implements schema-level Hilbert calculus for first-order logic:
- Axiom schemas: H1-H6 (propositional) + Q1, Q2 (quantifiers)
- Inference rules: Modus Ponens (MP) and Generalization (Gen)
- Schema-level BFS generation with checkpointing

Usage:
    from hilbert.schema_generator import SchemaGenerator

    gen = SchemaGenerator(logic="zfc", max_formula_size=15)
    graph = gen.generate()
"""

from .schema_generator import CheckpointChildError, SchemaGenerator
from .side_conditions import is_free_for, is_not_free_in

__all__ = [
    'SchemaGenerator',
    'CheckpointChildError',
    'is_free_for',
    'is_not_free_in',
]
