"""
Kripke-Platek set theory (KP).

KP is a weak fragment of ZF over classical FOL.  It is the minimal set theory
sufficient for admissible set recursion theory.

Key differences from ZFC:
  - No Power Set axiom.
  - No Axiom of Choice.
  - Separation restricted to Δ₀-formulas (bounded quantifiers only).
  - Foundation replaced by ∈-Induction (schematically).
    Here we use the concrete first-order Foundation axiom as a proxy, since
    ∈-induction is a schema (like Separation/Replacement) and would need
    the same finite-instance treatment.

References:
    Kripke 1964; Platek 1966; Barwise 1975.
"""

from typing import Dict

from .formula import (
    Formula, Membership, Equality, Not, And, Or,
    Implies, Iff, ForAll, Exists, make_variable
)
from .axioms import ZFCAxioms


class KPAxioms(ZFCAxioms):
    """
    KP = Kripke-Platek set theory over classical FOL.

    Axioms included:
      ax-ext   : Extensionality
      ax-pair  : Pairing
      ax-union : Union
      ax-inf   : Infinity
      ax-reg   : Foundation (used as a proxy for ∈-Induction)

    Axioms excluded:
      ax-empty  : (derivable from Separation + Foundation in KP; omitted to
                   keep the axiom set minimal)
      ax-pow    : Power Set — NOT part of KP
      ax-choice : Axiom of Choice — NOT part of KP

    Schema instances (Δ₀ only):
      ax-sep-mem : {z ∈ x : z ∈ a}
      ax-sep-eq  : {z ∈ x : z = a}
    """

    @classmethod
    def get_all_axioms(cls, include_choice: bool = False,
                       include_schemas: bool = False,
                       use_schema_instances: bool = False) -> Dict[str, Formula]:
        """
        Get KP axioms.  Power Set and Choice are excluded regardless of flags.
        """
        axioms: Dict[str, Formula] = {
            'ax-ext':   cls.extensionality(),
            'ax-pair':  cls.pairing(),
            'ax-union': cls.union(),
            'ax-inf':   cls.infinity(),
            'ax-reg':   cls.foundation(),   # proxy for ∈-Induction schema
        }
        # ax-pow excluded: KP has no power set
        # ax-choice excluded: KP has no choice
        # ax-empty excluded: derivable in KP from Δ₀-Separation

        if include_schemas:
            axioms.update(cls.get_common_schema_instances())

        return axioms

    @classmethod
    def get_common_schema_instances(cls) -> Dict[str, Formula]:
        """
        Δ₀-safe instances of Separation only.

        Returned instances:
          ax-sep-mem : {z ∈ x : z ∈ a}
          ax-sep-eq  : {z ∈ x : z = a}
        """
        z_var = make_variable('z')
        a_var = make_variable('a')

        phi_membership = Membership(z_var, a_var)
        phi_equality   = Equality(z_var, a_var)

        return {
            'ax-sep-mem': cls.separation_instance(phi_membership, 'z'),
            'ax-sep-eq':  cls.separation_instance(phi_equality,   'z'),
        }
