"""
Second-order ZFC (ZFC₂).

ZFC₂ uses the full (standard) semantics of second-order logic (SOL) over the
signature {∈}.  The key feature is that Separation and Replacement become
single second-order axioms rather than first-order axiom *schemas*:

  Separation  : ∀P^1 ∀a ∃b ∀z (z∈b ↔ (z∈a ∧ P(z)))
  Replacement : ∀F^2 ∀a ∃b ∀z (z∈b ↔ ∃w(w∈a ∧ z = F(w)))

Together with the standard ZFC individual axioms (Extensionality, Empty Set,
Pairing, Union, Power Set, Infinity, Foundation, Choice) this gives exactly
**10 axioms and 0 schemas**.

ZFC₂ under full semantics is categorical up to Vκ for inaccessible κ.

For MSOL (Monadic SOL) only Separation has a single second-order axiom;
Replacement remains a schema because MSOL lacks binary function variables.

References:
    Zermelo 1930; Shapiro 1991; Väänänen 2001.
"""

from typing import Dict

from .formula import (
    Formula, Term, Membership, Equality, Not, And, Or,
    Implies, Iff, ForAll, Exists,
    PredApp, ForAllPred, ExistsPred,
    make_variable,
)
from .axioms import ZFCAxioms


# Standard variables
x = make_variable('x')
y = make_variable('y')
z = make_variable('z')
w = make_variable('w')
a = make_variable('a')
b = make_variable('b')


class ZFC2Axioms(ZFCAxioms):
    """
    ZFC₂ — second-order ZFC with full (standard) semantics.

    10 axioms, 0 schemas:
      ax-ext, ax-empty, ax-pair, ax-union, ax-pow, ax-inf, ax-reg, ax-choice
      + ax-sep-sol (∀P^1 ∀a ∃b ∀z (z∈b ↔ (z∈a ∧ P(z))))
      + ax-rep-sol (∀F^2 ∀a ∃b ∀z (z∈b ↔ ∃w(w∈a ∧ z=F(w))))
    """

    @staticmethod
    def separation_sol() -> Formula:
        """
        Second-order Separation axiom (single axiom, no schema):
        ∀P^1.∀a.∃b.∀z.(z∈b ↔ (z∈a ∧ P(z)))
        """
        z_in_b = Membership(z, b)
        z_in_a = Membership(z, a)
        p_of_z = PredApp("P", (z,))
        body = ForAll('a', Exists('b', ForAll('z',
            Iff(z_in_b, And(z_in_a, p_of_z)))))
        return ForAllPred("P", 1, body)

    @staticmethod
    def replacement_sol() -> Formula:
        """
        Second-order Replacement axiom (single axiom, no schema):
        ∀F^2.∀a.∃b.∀z.(z∈b ↔ ∃w(w∈a ∧ z=F(w)))
        """
        z_in_b = Membership(z, b)
        w_in_a = Membership(w, a)
        z_eq_fw = Equality(z, make_variable('F_placeholder'))  # placeholder; replaced below
        # F(w) is represented as PredApp("F", (w,)) — a binary pred applied to one arg
        # actually F is a function: F : domain → domain, so we use F as a 1-ary pred var
        # (output is determined by input: z = F(w) encoded as PredApp("F", (w, z)) = True)
        # Simpler: use F as a 2-ary predicate for the graph of F: F(w,z) ≡ "F maps w to z"
        f_wz = PredApp("F", (w, z))
        inner = Exists('w', And(w_in_a, f_wz))
        body = ForAll('a', Exists('b', ForAll('z', Iff(z_in_b, inner))))
        return ForAllPred("F", 2, body)

    @classmethod
    def get_all_axioms(cls, include_choice: bool = True,
                       include_schemas: bool = False,
                       use_schema_instances: bool = False) -> Dict[str, Formula]:
        """
        Get ZFC₂ axioms: 8 individual axioms + 2 SOL axioms = 10 total.
        include_schemas is ignored (ZFC₂ has no schemas).
        """
        axioms: Dict[str, Formula] = {
            'ax-ext':    cls.extensionality(),
            'ax-empty':  cls.empty_set(),
            'ax-pair':   cls.pairing(),
            'ax-union':  cls.union(),
            'ax-pow':    cls.power_set(),
            'ax-inf':    cls.infinity(),
            'ax-reg':    cls.foundation(),
            'ax-sep-sol': cls.separation_sol(),
            'ax-rep-sol': cls.replacement_sol(),
        }
        if include_choice:
            axioms['ax-choice'] = cls.choice()
        return axioms

    @classmethod
    def get_common_schema_instances(cls) -> Dict[str, Formula]:
        """ZFC₂ has no schema instances — Separation and Replacement are single axioms."""
        return {}


class HenkinSOLAxioms(ZFC2Axioms):
    """
    Henkin second-order logic + ZFC₂ (henkin_sol configuration).

    The axiom *schemas* are identical to ZFC2Axioms (ax-sep-sol, ax-rep-sol
    plus the 8 individual ZFC axioms).  The semantic difference — quantifiers
    range over a fixed "general model" rather than all relations — does not
    affect which formulas are written as axioms but does change which
    comprehension principles hold.

    Syntactic distinction from sol_zfc2
    ─────────────────────────────────────
    In full (standard) SOL, comprehension ∃P ∀z(P(z) ↔ φ(z)) is a semantic
    theorem: every definable class is a relation in the full structure, so no
    extra axiom is needed.  In Henkin semantics the general model may omit
    some relations, so comprehension must be *stipulated* for each template
    family that the proof graph uses.  We add one comprehension axiom per
    pred_instantiation_v1 template (5 axioms total), matching the templates
    used by Pred-Inst so that every Henkin comprehension fact has a
    corresponding graph vertex.

    Comprehension axioms added (ax-henkin-comp-*):
        Arity-1: ∃P^1 ∀z (P(z) ↔ z∈a)        — comp-1-mem
                 ∃P^1 ∀z (P(z) ↔ z=a)         — comp-1-eq
                 ∃P^1 ∀z (P(z) ↔ ¬(z∈a))      — comp-1-notmem
        Arity-2: ∃F^2 ∀w∀z (F(w,z) ↔ w=z)     — comp-2-id
                 ∃F^2 ∀w∀z (F(w,z) ↔ z=a)     — comp-2-const

    These five axioms create cross-edges in the proof graph between the
    Henkin axiom layer and Pred-Inst derivations, which is absent in sol_zfc2
    and yields a structurally different degree distribution.

    Policy: henkin_comprehension_v1.  Any extension of the template set
    (pred_instantiation_v1) must be accompanied by a matching extension here
    and a policy version bump.
    """

    @staticmethod
    def _comprehension_axioms() -> Dict[str, "Formula"]:
        """Return Henkin comprehension axioms for all pred_instantiation_v1 templates."""
        z = make_variable('z')
        w = make_variable('w')
        a = make_variable('a')

        # Arity-1: ∃P^1 ∀z (P(z) ↔ φ(z))
        def comp1(phi_z: "Formula", name: str) -> tuple:
            body = ForAll('z', Iff(PredApp("P", (z,)), phi_z))
            return (name, ExistsPred("P", 1, body))

        # Arity-2: ∃F^2 ∀w∀z (F(w,z) ↔ φ(w,z))
        def comp2(phi_wz: "Formula", name: str) -> tuple:
            body = ForAll('w', ForAll('z', Iff(PredApp("F", (w, z)), phi_wz)))
            return (name, ExistsPred("F", 2, body))

        pairs = [
            comp1(Membership(z, a),        'ax-henkin-comp-1-mem'),
            comp1(Equality(z, a),          'ax-henkin-comp-1-eq'),
            comp1(Not(Membership(z, a)),   'ax-henkin-comp-1-notmem'),
            comp2(Equality(w, z),          'ax-henkin-comp-2-id'),
            comp2(Equality(z, a),          'ax-henkin-comp-2-const'),
        ]
        return dict(pairs)

    @classmethod
    def get_all_axioms(cls, include_choice: bool = True,
                       include_schemas: bool = False,
                       use_schema_instances: bool = False,
                       sep_instance_limit=None,
                       rep_instance_limit=None) -> Dict[str, "Formula"]:
        """ZFC₂ axioms + Henkin comprehension axioms (5 additional)."""
        axioms = super().get_all_axioms(
            include_choice=include_choice,
            include_schemas=include_schemas,
            use_schema_instances=use_schema_instances,
        )
        axioms.update(cls._comprehension_axioms())
        return axioms


class MSOLAxioms(ZFC2Axioms):
    """
    Monadic Second-Order Logic + ZFC.

    MSOL allows quantification only over *unary* predicates (P^1).
    Therefore:
      - Separation becomes a single MSOL axiom  (ax-sep-sol, same as ZFC₂).
      - Replacement CANNOT be a single axiom (no binary function variables).
        It falls back to the first-order schema approach with concrete instances.

    Axioms: 8 individual + 1 SOL-sep + N schema instances for replacement.
    """

    @classmethod
    def get_all_axioms(cls, include_choice: bool = True,
                       include_schemas: bool = False,
                       use_schema_instances: bool = False) -> Dict[str, Formula]:
        """
        Get MSOL+ZFC axioms.
        """
        axioms: Dict[str, Formula] = {
            'ax-ext':    cls.extensionality(),
            'ax-empty':  cls.empty_set(),
            'ax-pair':   cls.pairing(),
            'ax-union':  cls.union(),
            'ax-pow':    cls.power_set(),
            'ax-inf':    cls.infinity(),
            'ax-reg':    cls.foundation(),
            'ax-sep-sol': cls.separation_sol(),   # Single MSOL axiom for Separation
        }
        if include_choice:
            axioms['ax-choice'] = cls.choice()

        if include_schemas:
            # Replacement remains a schema — include concrete instances
            axioms.update(cls._get_replacement_instances())

        return axioms

    @classmethod
    def _get_replacement_instances(cls) -> Dict[str, Formula]:
        """Concrete Replacement instances (fallback since MSOL lacks binary F)."""
        y_var = make_variable('y')
        z_var = make_variable('z')
        w_var = make_variable('w')
        a_var = make_variable('a')

        phi_id_yz = Equality(y_var, z_var)
        phi_id_yw = Equality(y_var, w_var)
        phi_const_yz = Equality(z_var, a_var)
        phi_const_yw = Equality(w_var, a_var)

        return {
            'ax-rep-id':    cls.replacement_instance(
                phi_id_yz, phi_id_yw, input_var='y', output_var='z', alt_output_var='w'
            ),
            'ax-rep-const': cls.replacement_instance(
                phi_const_yz, phi_const_yw, input_var='y', output_var='z', alt_output_var='w'
            ),
        }

    @classmethod
    def get_common_schema_instances(cls) -> Dict[str, Formula]:
        """MSOL: only Replacement schema instances (Separation is an axiom)."""
        return cls._get_replacement_instances()
