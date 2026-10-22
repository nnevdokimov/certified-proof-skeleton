"""
Intuitionistic Zermelo-Fraenkel set theory (IZF).

IZF is ZF without the Axiom of Choice, over an intuitionistic (H1-H5) propositional
base. Separation is restricted to Δ₀-formulas (bounded quantifiers only) to keep
the theory constructively acceptable. Replacement is kept in full.

References:
    Myhill 1973; Aczel 1978; Troelstra, van Dalen 1988.
"""

from typing import Dict

from .formula import (
    Formula, Membership, Equality, Not, And, Or,
    Implies, Iff, ForAll, Exists, make_variable
)
from .axioms import ZFCAxioms


class IZFAxioms(ZFCAxioms):
    """
    IZF = ZF (without Choice) over intuitionistic logic.

    Differences from full ZFC:
      - No Axiom of Choice.
      - Separation is restricted to Δ₀ (bounded) predicates only.
        get_common_schema_instances() returns only ax-sep-mem and ax-sep-eq.
      - All other ZF axioms are unchanged.
    """

    @classmethod
    def get_all_axioms(cls, include_choice: bool = False,   # Choice excluded by default
                       include_schemas: bool = False,
                       use_schema_instances: bool = False) -> Dict[str, Formula]:
        """
        Get IZF axioms.  Choice is excluded regardless of include_choice.
        """
        axioms: Dict[str, Formula] = {
            'ax-ext':   cls.extensionality(),
            'ax-empty': cls.empty_set(),
            'ax-pair':  cls.pairing(),
            'ax-union': cls.union(),
            'ax-pow':   cls.power_set(),
            'ax-inf':   cls.infinity(),
            'ax-reg':   cls.foundation(),
        }
        # Choice is NOT part of IZF — ignore include_choice

        if include_schemas:
            axioms.update(cls.get_common_schema_instances())

        return axioms

    @classmethod
    def get_common_schema_instances(cls) -> Dict[str, Formula]:
        """
        Δ₀-safe instances of Separation only (no Replacement instances,
        no non-Δ₀ predicates like ∃w(z∈w ∧ w∈a)).

        Returned instances:
          ax-sep-mem : {z ∈ x : z ∈ a}  — bounded membership predicate (Δ₀)
          ax-sep-eq  : {z ∈ x : z = a}  — bounded equality predicate (Δ₀)
        """
        z_var = make_variable('z')
        a_var = make_variable('a')

        phi_membership = Membership(z_var, a_var)
        phi_equality   = Equality(z_var, a_var)

        return {
            'ax-sep-mem': cls.separation_instance(phi_membership, 'z'),
            'ax-sep-eq':  cls.separation_instance(phi_equality,   'z'),
        }
