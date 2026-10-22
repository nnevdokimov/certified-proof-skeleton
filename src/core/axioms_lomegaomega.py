"""
ZFC in the infinitary logic L_{ω₁,ω}.

In L_{ω₁,ω} one allows countable conjunctions ⋀_{i<ω} φᵢ and disjunctions
⋁_{i<ω} φᵢ.  The key effect on ZFC is that the axiom *schemas* (Separation,
Replacement) become single *infinitary axioms*: we bind a finite approximation
of all schema instances into one BigAnd node.

In practice we cap the number of conjuncts at N_MAX = 6 for AST size control,
using the same concrete instance predicates as in ZFC.  This gives:

  ax-sep-bigand : ⋀(ax-sep-mem, ax-sep-eq, ax-sep-notmem, ax-sep-union)
  ax-rep-bigand : ⋀(ax-rep-id, ax-rep-const)

All other ZFC axioms are kept unchanged.

References:
    Scott 1965; Barwise 1975; Marker 2002.
"""

from typing import Dict

from .formula import Formula, BigAnd, BigOr
from .axioms import ZFCAxioms


class LOmegaOmegaAxioms(ZFCAxioms):
    """
    ZFC axioms adapted for L_{ω₁,ω}.

    Separation and Replacement schemas are replaced by BigAnd conjunctions of
    their concrete instances (finite approximation of the infinitary axiom).
    """

    @classmethod
    def get_all_axioms(cls, include_choice: bool = True,
                       include_schemas: bool = False,
                       use_schema_instances: bool = False) -> Dict[str, Formula]:
        """
        Get L_{ω₁,ω}+ZFC axioms.

        Separation and Replacement become BigAnd axioms.
        include_schemas is ignored (schemas are always folded into BigAnd).
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
        if include_choice:
            axioms['ax-choice'] = cls.choice()

        # Separation schema → single BigAnd infinitary axiom
        axioms['ax-sep-bigand'] = cls.separation_bigand()

        # Replacement schema → single BigAnd infinitary axiom
        axioms['ax-rep-bigand'] = cls.replacement_bigand()

        return axioms

    @classmethod
    def separation_bigand(cls) -> Formula:
        """
        L_{ω₁,ω} Separation: ⋀{sep_instance | φ ∈ {mem, eq, notmem, union}}
        """
        instances = cls._separation_instances()
        return BigAnd(tuple(instances.values()))

    @classmethod
    def replacement_bigand(cls) -> Formula:
        """
        L_{ω₁,ω} Replacement: ⋀{rep_instance | φ ∈ {id, const}}
        """
        instances = cls._replacement_instances()
        return BigAnd(tuple(instances.values()))

    @classmethod
    def _separation_instances(cls) -> Dict[str, Formula]:
        """All separation instances used in the BigAnd."""
        from .formula import Membership, Equality, Not, And, Exists, make_variable
        z_var = make_variable('z')
        a_var = make_variable('a')
        w_var = make_variable('w')

        return {
            'ax-sep-mem':    cls.separation_instance(Membership(z_var, a_var), 'z'),
            'ax-sep-eq':     cls.separation_instance(Equality(z_var, a_var), 'z'),
            'ax-sep-notmem': cls.separation_instance(Not(Membership(z_var, a_var)), 'z'),
            'ax-sep-union':  cls.separation_instance(
                Exists('w', And(Membership(z_var, w_var), Membership(w_var, a_var))), 'z'
            ),
        }

    @classmethod
    def _replacement_instances(cls) -> Dict[str, Formula]:
        """All replacement instances used in the BigAnd."""
        from .formula import Equality, make_variable
        y_var = make_variable('y')
        z_var = make_variable('z')
        w_var = make_variable('w')
        a_var = make_variable('a')

        phi_id_yz    = Equality(y_var, z_var)
        phi_id_yw    = Equality(y_var, w_var)
        phi_const_yz = Equality(z_var, a_var)
        phi_const_yw = Equality(w_var, a_var)

        return {
            'ax-rep-id':    cls.replacement_instance(phi_id_yz, phi_id_yw,
                                                      input_var='y', output_var='z',
                                                      alt_output_var='w'),
            'ax-rep-const': cls.replacement_instance(phi_const_yz, phi_const_yw,
                                                      input_var='y', output_var='z',
                                                      alt_output_var='w'),
        }

    @classmethod
    def get_common_schema_instances(cls) -> Dict[str, Formula]:
        """No individual schema instances — they're bundled into BigAnd axioms."""
        return {}
