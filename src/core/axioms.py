"""
Zermelo-Fraenkel set theory with Choice, represented in first-order syntax.

Ordinary ZF axioms are provided as closed first-order formulas.
Separation and Replacement are provided as schema-instance constructors plus a
small finite "common instances" policy for experiments that need a concrete
finite approximation.
"""

from typing import List, Dict, Optional
import logging

from .formula import (
    Formula, Term, Membership, Equality, Not, And, Or,
    Implies, Iff, ForAll, Exists, PredApp, make_variable, make_constant
)

logger = logging.getLogger(__name__)


# Standard variables used in axioms
x = make_variable('x')
y = make_variable('y')
z = make_variable('z')
w = make_variable('w')
u = make_variable('u')
v = make_variable('v')
a = make_variable('a')
b = make_variable('b')
c = make_variable('c')
e = make_variable('e')
s = make_variable('s')
f = make_variable('f')


class ZFCAxioms:
    """
    ZFC Axioms of set theory.

    Provides static methods to construct each axiom as a Formula object.
    """

    @staticmethod
    def _universal_closure(formula: Formula, bound_vars: Optional[set[str]] = None) -> Formula:
        """Universally close the remaining free variables of a schema instance."""
        bound_vars = bound_vars or set()
        free_vars = sorted(v for v in formula.free_variables() if v not in bound_vars)
        closed = formula
        for var in reversed(free_vars):
            closed = ForAll(var, closed)
        return closed

    @staticmethod
    def extensionality() -> Formula:
        """
        Axiom of Extensionality:
        Two sets are equal iff they have the same elements.

        ∀x∀y(∀z(z∈x ↔ z∈y) → x=y)

        Returns:
            The extensionality axiom as a Formula
        """
        z_in_x = Membership(z, x)
        z_in_y = Membership(z, y)
        same_elements = ForAll('z', Iff(z_in_x, z_in_y))
        x_eq_y = Equality(x, y)

        return ForAll('x', ForAll('y', Implies(same_elements, x_eq_y)))

    @staticmethod
    def empty_set() -> Formula:
        """
        Axiom of Empty Set (Null Set):
        There exists a set with no elements.

        ∃x∀y(y∉x)

        Equivalently: ∃x∀y¬(y∈x)

        Returns:
            The empty set axiom as a Formula
        """
        y_not_in_x = Not(Membership(y, x))
        return Exists('x', ForAll('y', y_not_in_x))

    @staticmethod
    def pairing() -> Formula:
        """
        Axiom of Pairing:
        For any x, y, there exists a set {x, y}.

        ∀x∀y∃z∀w(w∈z ↔ (w=x ∨ w=y))

        Returns:
            The pairing axiom as a Formula
        """
        w_in_z = Membership(w, z)
        w_eq_x = Equality(w, x)
        w_eq_y = Equality(w, y)
        condition = Iff(w_in_z, Or(w_eq_x, w_eq_y))

        return ForAll('x', ForAll('y', Exists('z', ForAll('w', condition))))

    @staticmethod
    def union() -> Formula:
        """
        Axiom of Union:
        For any set, its union exists.

        ∀x∃y∀z(z∈y ↔ ∃w(z∈w ∧ w∈x))

        The union ∪x contains all elements of elements of x.

        Returns:
            The union axiom as a Formula
        """
        z_in_y = Membership(z, y)
        z_in_w = Membership(z, w)
        w_in_x = Membership(w, x)
        exists_w = Exists('w', And(z_in_w, w_in_x))
        condition = Iff(z_in_y, exists_w)

        return ForAll('x', Exists('y', ForAll('z', condition)))

    @staticmethod
    def power_set() -> Formula:
        """
        Axiom of Power Set:
        For any set, the set of all its subsets exists.

        ∀x∃y∀z(z∈y ↔ ∀w(w∈z → w∈x))

        where ∀w(w∈z → w∈x) means z⊆x.

        Returns:
            The power set axiom as a Formula
        """
        z_in_y = Membership(z, y)
        w_in_z = Membership(w, z)
        w_in_x = Membership(w, x)
        z_subset_x = ForAll('w', Implies(w_in_z, w_in_x))
        condition = Iff(z_in_y, z_subset_x)

        return ForAll('x', Exists('y', ForAll('z', condition)))

    @staticmethod
    def infinity() -> Formula:
        """
        Axiom of Infinity:
        An infinite set exists (containing ∅ and closed under successor).

        ∃x(∃e(∀z(z∉e) ∧ e∈x) ∧ ∀y(y∈x → ∃s(s∈x ∧ ∀w(w∈s ↔ (w∈y ∨ w=y)))))

        This asserts existence of a set containing:
        - The empty set ∅
        - For each element y, also y ∪ {y} (the successor)

        Returns:
            The infinity axiom as a Formula
        """
        # ∅ exists and belongs to x
        z_not_in_e = Not(Membership(z, e))
        e_is_empty = ForAll('z', z_not_in_e)
        e_in_x = Membership(e, x)
        empty_in_x = Exists('e', And(e_is_empty, e_in_x))

        # For each y∈x, successor s = y ∪ {y} exists in x
        y_in_x = Membership(y, x)
        s_in_x = Membership(s, x)
        w_in_s = Membership(w, s)
        w_in_y = Membership(w, y)
        w_eq_y = Equality(w, y)
        s_is_successor = ForAll('w', Iff(w_in_s, Or(w_in_y, w_eq_y)))
        successor_exists = Exists('s', And(s_in_x, s_is_successor))
        closure = ForAll('y', Implies(y_in_x, successor_exists))

        return Exists('x', And(empty_in_x, closure))

    @staticmethod
    def separation(predicate_name: str = "φ") -> Formula:
        """
        Axiom Schema of Separation (Specification/Comprehension):
        For any set x and property φ, the subset {z∈x : φ(z)} exists.

        ∀x∃y∀z(z∈y ↔ (z∈x ∧ φ(z)))

        This placeholder form is kept only as a legacy helper.
        Actual experiments should use concrete instances generated by
        `separation_instance()`.

        Args:
            predicate_name: Name for the predicate placeholder

        Returns:
            The separation schema as a Formula (placeholder form)
        """
        z_in_y = Membership(z, y)
        z_in_x = Membership(z, x)
        # φ(z) represented as z ∈ φ (placeholder)
        phi_z = Membership(z, make_constant(predicate_name))
        condition = Iff(z_in_y, And(z_in_x, phi_z))

        return ForAll('x', Exists('y', ForAll('z', condition)))

    @staticmethod
    def separation_instance(phi: Formula, bound_var: str = 'z') -> Formula:
        """
        Generate a concrete instance of the Separation schema.

        ∀x∃y∀z(z∈y ↔ (z∈x ∧ φ(z)))

        where φ is a concrete formula with 'z' as its free variable.

        Args:
            phi: A formula with bound_var as a free variable (the property)
            bound_var: The variable used in φ (default 'z')

        Returns:
            A concrete instance of the Separation axiom
        """
        # Ensure we use fresh variables if phi uses x or y
        all_vars = phi.free_variables() | phi.bound_variables()
        set_var = 'x' if 'x' not in all_vars else Formula._fresh_variable(all_vars, 'x')
        subset_var = 'y' if 'y' not in (all_vars | {set_var}) else Formula._fresh_variable(all_vars | {set_var}, 'y')

        x_term = make_variable(set_var)
        y_term = make_variable(subset_var)
        z_term = make_variable(bound_var)

        z_in_y = Membership(z_term, y_term)
        z_in_x = Membership(z_term, x_term)
        # φ(z) is the actual formula
        condition = Iff(z_in_y, And(z_in_x, phi))

        instance = ForAll(set_var, Exists(subset_var, ForAll(bound_var, condition)))
        return ZFCAxioms._universal_closure(instance, {set_var, subset_var, bound_var})

    @staticmethod
    def replacement(function_name: str = "F") -> Formula:
        """
        Axiom Schema of Replacement:
        The image of a set under a functional relation is a set.

        ∀x(∀y∀z∀w((φ(y,z) ∧ φ(y,w)) → z=w) →
           ∃v∀z(z∈v ↔ ∃y(y∈x ∧ φ(y,z))))

        This placeholder form is kept only as a legacy helper.
        Actual experiments should use concrete instances generated by
        `replacement_instance()`.

        Args:
            function_name: Name for the function placeholder

        Returns:
            The replacement schema as a Formula (placeholder form)
        """
        # Placeholders for φ(y,z) and φ(y,w)
        phi_yz = Membership(y, make_constant(f"{function_name}_range"))
        phi_yw = Membership(y, make_constant(f"{function_name}_range2"))

        # Functionality: φ(y,z) ∧ φ(y,w) → z=w
        z_eq_w = Equality(z, w)
        functional_condition = Implies(And(phi_yz, phi_yw), z_eq_w)
        functionality = ForAll('y', ForAll('z', ForAll('w', functional_condition)))

        # Image exists
        z_in_v = Membership(z, v)
        y_in_x = Membership(y, x)
        image_condition = Iff(z_in_v, Exists('y', And(y_in_x, phi_yz)))
        image_exists = Exists('v', ForAll('z', image_condition))

        return ForAll('x', Implies(functionality, image_exists))

    @staticmethod
    def replacement_instance(phi_yz: Formula, phi_yw: Formula,
                            input_var: str = 'y', output_var: str = 'z',
                            alt_output_var: str = 'w') -> Formula:
        """
        Generate a concrete instance of the Replacement schema.

        ∀x(∀y∀z∀w((φ(y,z) ∧ φ(y,w)) → z=w) →
           ∃v∀z(z∈v ↔ ∃y(y∈x ∧ φ(y,z))))

        where φ(y,z) is a concrete binary relation formula.

        Args:
            phi_yz: A formula representing φ(y,z) - the functional relation
            phi_yw: Same formula but with output_var replaced by alt_output_var
            input_var: The input variable of the relation (default 'y')
            output_var: The output variable of the relation (default 'z')
            alt_output_var: Alternative output variable for uniqueness (default 'w')

        Returns:
            A concrete instance of the Replacement axiom
        """
        # Collect all variables used
        all_vars = (phi_yz.free_variables() | phi_yz.bound_variables() |
                   phi_yw.free_variables() | phi_yw.bound_variables())

        # Choose fresh variables for x and v if needed
        set_var = 'x' if 'x' not in all_vars else Formula._fresh_variable(all_vars, 'x')
        image_var = 'v' if 'v' not in (all_vars | {set_var}) else Formula._fresh_variable(all_vars | {set_var}, 'v')

        x_term = make_variable(set_var)
        v_term = make_variable(image_var)
        y_term = make_variable(input_var)
        z_term = make_variable(output_var)
        w_term = make_variable(alt_output_var)

        # Functionality: φ(y,z) ∧ φ(y,w) → z=w
        z_eq_w = Equality(z_term, w_term)
        functional_condition = Implies(And(phi_yz, phi_yw), z_eq_w)
        functionality = ForAll(input_var, ForAll(output_var, ForAll(alt_output_var, functional_condition)))

        # Image exists
        z_in_v = Membership(z_term, v_term)
        y_in_x = Membership(y_term, x_term)
        image_condition = Iff(z_in_v, Exists(input_var, And(y_in_x, phi_yz)))
        image_exists = Exists(image_var, ForAll(output_var, image_condition))

        instance = ForAll(set_var, Implies(functionality, image_exists))
        return ZFCAxioms._universal_closure(
            instance,
            {set_var, image_var, input_var, output_var, alt_output_var},
        )

    @staticmethod
    def foundation() -> Formula:
        """
        Axiom of Foundation (Regularity):
        Every non-empty set has an ∈-minimal element.

        ∀x(∃w(w∈x) → ∃y(y∈x ∧ ∀z(z∈y → z∉x)))

        Equivalently: no infinite descending ∈-chains exist.

        Returns:
            The foundation axiom as a Formula
        """
        # x is non-empty
        x_nonempty = Exists('w', Membership(w, x))

        # y is an ∈-minimal element of x
        y_in_x = Membership(y, x)
        z_in_y = Membership(z, y)
        z_not_in_x = Not(Membership(z, x))
        y_minimal = ForAll('z', Implies(z_in_y, z_not_in_x))
        minimal_exists = Exists('y', And(y_in_x, y_minimal))

        return ForAll('x', Implies(x_nonempty, minimal_exists))

    @staticmethod
    def choice() -> Formula:
        """
        Axiom of Choice in "choice set" form.

        We use a standard equivalent formulation in the pure language {∈, =}:
        every family of pairwise disjoint non-empty sets has a choice set.

        ∀x(
          (all members of x are non-empty and pairwise disjoint)
          →
          ∃c ∀y(
            y∈x →
            ∃z(
              z∈y ∧ z∈c ∧ ∀w(((w∈y ∧ w∈c) → w=z))
            )
          )
        )

        Returns:
            The choice axiom as a Formula
        """
        y_in_x = Membership(y, x)
        y_nonempty = Exists('u', Membership(u, y))
        all_nonempty = ForAll('y', Implies(y_in_x, y_nonempty))

        z_in_x = Membership(z, x)
        yz_distinct = Not(Equality(y, z))
        u_in_y = Membership(u, y)
        u_in_z = Membership(u, z)
        overlap = Exists('u', And(u_in_y, u_in_z))
        pairwise_disjoint = ForAll(
            'y',
            ForAll(
                'z',
                Implies(And(y_in_x, z_in_x), Implies(yz_distinct, Not(overlap))),
            ),
        )

        c_term = c
        z_in_y = Membership(z, y)
        z_in_c = Membership(z, c_term)
        w_in_y = Membership(w, y)
        w_in_c = Membership(w, c_term)
        unique_in_section = ForAll('w', Implies(And(w_in_y, w_in_c), Equality(w, z)))
        choice_for_y = Exists('z', And(And(z_in_y, z_in_c), unique_in_section))
        choice_set = Exists('c', ForAll('y', Implies(y_in_x, choice_for_y)))

        return ForAll('x', Implies(And(all_nonempty, pairwise_disjoint), choice_set))

    @classmethod
    def get_all_axioms(cls, include_choice: bool = True,
                       include_schemas: bool = False,
                       use_schema_instances: bool = False,
                       sep_instance_limit: Optional[int] = None,
                       rep_instance_limit: Optional[int] = None) -> Dict[str, Formula]:
        """
        Get all ZFC axioms as a dictionary.

        Args:
            include_choice: Whether to include Axiom of Choice
            include_schemas: Whether to include a finite schema-instance policy
            use_schema_instances: Deprecated compatibility flag. Placeholder
                                 schema formulas are no longer returned.
            sep_instance_limit: Max number of Separation instances to include
                (None = all 12). Instances are ordered by complexity (atomic →
                boolean → quantified) per schema_instance_budget_v1 policy.
            rep_instance_limit: Max number of Replacement instances to include
                (None = all 4). Instances ordered simplest first.

        Returns:
            Dictionary mapping axiom names to Formula objects
        """
        axioms = {
            'ax-ext': cls.extensionality(),
            'ax-empty': cls.empty_set(),
            'ax-pair': cls.pairing(),
            'ax-union': cls.union(),
            'ax-pow': cls.power_set(),
            'ax-inf': cls.infinity(),
            'ax-reg': cls.foundation(),
        }

        if include_choice:
            axioms['ax-choice'] = cls.choice()

        if include_schemas:
            if not use_schema_instances:
                logger.warning(
                    "Placeholder schema axioms are disabled; using common concrete "
                    "schema instances instead."
                )
            axioms.update(cls.get_common_schema_instances(
                sep_limit=sep_instance_limit,
                rep_limit=rep_instance_limit,
            ))

        logger.debug(f"Created {len(axioms)} axioms")
        return axioms

    @classmethod
    def get_axiom_names(cls, include_choice: bool = True,
                        include_schemas: bool = True) -> List[str]:
        """
        Get list of all axiom names.

        Args:
            include_choice: Include Axiom of Choice
            include_schemas: Include schema axioms

        Returns:
            List of axiom names
        """
        names = [
            'ax-ext', 'ax-empty', 'ax-pair', 'ax-union',
            'ax-pow', 'ax-inf', 'ax-reg'
        ]

        if include_choice:
            names.append('ax-choice')

        if include_schemas:
            names.extend(sorted(cls.get_common_schema_instances().keys()))

        return names

    @classmethod
    def get_axiom_by_name(cls, name: str) -> Optional[Formula]:
        """
        Get a specific axiom by name.

        Args:
            name: Axiom name (e.g., 'ax-ext', 'ax-pair')

        Returns:
            The axiom Formula, or None if not found
        """
        axiom_map = {
            'ax-ext': cls.extensionality,
            'ax-empty': cls.empty_set,
            'ax-pair': cls.pairing,
            'ax-union': cls.union,
            'ax-pow': cls.power_set,
            'ax-inf': cls.infinity,
            'ax-reg': cls.foundation,
            'ax-choice': cls.choice,
        }

        if name in axiom_map:
            return axiom_map[name]()
        common_instances = cls.get_common_schema_instances()
        if name in common_instances:
            return common_instances[name]
        return None

    @classmethod
    def get_axiom_description(cls, name: str) -> str:
        """
        Get a human-readable description of an axiom.

        Args:
            name: Axiom name

        Returns:
            Description string
        """
        descriptions = {
            'ax-ext': "Extensionality: Sets with same elements are equal",
            'ax-empty': "Empty Set: There exists an empty set",
            'ax-pair': "Pairing: For any x,y, set {x,y} exists",
            'ax-union': "Union: For any set, its union exists",
            'ax-pow': "Power Set: For any set, its power set exists",
            'ax-inf': "Infinity: An infinite set exists",
            'ax-reg': "Foundation: No infinite ∈-chains",
            'ax-choice': "Choice: every pairwise disjoint non-empty family has a choice set",
        }
        if name in cls.get_common_schema_instances():
            return f"Concrete schema instance: {name}"
        return descriptions.get(name, f"Unknown axiom: {name}")

    @staticmethod
    def schematic_separation() -> Formula:
        """I7 faithful Separation: ∀x∃y∀z(z∈y ↔ z∈x ∧ ?P(z)) — free predicate
        metavariable ?P, parameters FREE (schema-with-parameters; no universal
        closure). One ⊒-maximal node that subsumes every concrete instance;
        ?P is instantiated on demand by pattern-MP. See
        tests/test_i7_schematic_coverage.py / docs/PHASE_A_COMPLETION.md."""
        zt = make_variable("z")
        return ForAll("x", Exists("y", ForAll("z",
            Iff(Membership(zt, make_variable("y")),
                And(Membership(zt, make_variable("x")), PredApp("P", (zt,)))))))

    @staticmethod
    def get_separation_instances_upto_k(k: int, params=("a", "b"),
                                        max_instances: Optional[int] = None) -> Dict[str, Formula]:
        """Option B: principled Separation instances = ALL φ of complexity ≤ k
        (signature {∈,=,¬,→,∀}, z free). Uniform, convergent (k→∞ → full ZFC
        Separation schema), free of the hand-picked-12 artifact.

        `max_instances` is an explicit caller guard only. If it would bind, fail
        rather than silently changing the mathematical object.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1 for Separation enumeration, got {k!r}")
        if max_instances is not None and max_instances < 0:
            raise ValueError(f"max_instances must be non-negative or None, got {max_instances!r}")
        from .formula_enum import enumerate_predicates
        preds = enumerate_predicates(k, tuple(params))
        if max_instances is not None and len(preds) > max_instances:
            raise ValueError(
                f"Separation enumeration produced {len(preds)} instances for k={k}, "
                f"exceeding explicit max_instances={max_instances}; refusing to truncate"
            )
        return {f"ax-sep-k{k}-{i}": ZFCAxioms.separation_instance(phi, "z")
                for i, phi in enumerate(preds)}

    @staticmethod
    def schematic_replacement() -> Formula:
        """I7 faithful Replacement: functional relation as a binary predicate
        metavariable ?F(y,z). One ⊒-maximal node subsuming all concrete
        Replacement instances; ?F bound on demand by pattern-MP."""
        yt, zt, wt = make_variable("y"), make_variable("z"), make_variable("w")
        return ZFCAxioms.replacement_instance(
            PredApp("F", (yt, zt)), PredApp("F", (yt, wt)),
            input_var="y", output_var="z", alt_output_var="w",
        )

    @classmethod
    def get_common_schema_instances(
        cls,
        sep_limit: Optional[int] = None,
        rep_limit: Optional[int] = None,
    ) -> Dict[str, Formula]:
        """
        Generate concrete instances of Separation and Replacement.

        Instances are ordered by predicate complexity (schema_instance_budget_v1
        policy): atomic → boolean combinations → quantified for Separation;
        simplest functional relation first for Replacement.  This ordering is
        the canonical basis for sensitivity analysis: budget b means "take the
        first b entries from each category list".

        Separation tiers (12 total):
          Tier 1 — atomic (4):   mem, eq, notmem, noteq
          Tier 2 — boolean (5):  and-mem-mem, or-mem-mem, diff, or-eq-eq, impl-mem
          Tier 3 — quantified (3): union, subset, exists-elt

        Replacement order (4 total): id, const, into-a, elem-of

        Args:
            sep_limit: Keep only the first N Separation instances (None = all 12).
            rep_limit: Keep only the first N Replacement instances (None = all 4).

        Returns:
            Ordered dict of axiom instance names → Formulas.
        """
        z_var = make_variable('z')
        a_var = make_variable('a')
        b_var = make_variable('b')
        y_var = make_variable('y')
        w_var = make_variable('w')

        # ── Separation: ordered from simplest to most complex ─────────
        sep_ordered: list = []

        # Tier 1: atomic predicates (4 instances)
        sep_ordered.append(('ax-sep-mem', cls.separation_instance(
            Membership(z_var, a_var), 'z')))                      # z ∈ a
        sep_ordered.append(('ax-sep-eq', cls.separation_instance(
            Equality(z_var, a_var), 'z')))                        # z = a
        sep_ordered.append(('ax-sep-notmem', cls.separation_instance(
            Not(Membership(z_var, a_var)), 'z')))                 # z ∉ a
        sep_ordered.append(('ax-sep-noteq', cls.separation_instance(
            Not(Equality(z_var, a_var)), 'z')))                   # z ≠ a

        # Tier 2: boolean combinations (5 instances)
        sep_ordered.append(('ax-sep-and-mem-mem', cls.separation_instance(
            And(Membership(z_var, a_var), Membership(z_var, b_var)), 'z')))
        sep_ordered.append(('ax-sep-or-mem-mem', cls.separation_instance(
            Or(Membership(z_var, a_var), Membership(z_var, b_var)), 'z')))
        sep_ordered.append(('ax-sep-diff', cls.separation_instance(
            And(Membership(z_var, a_var), Not(Membership(z_var, b_var))), 'z')))
        sep_ordered.append(('ax-sep-or-eq-eq', cls.separation_instance(
            Or(Equality(z_var, a_var), Equality(z_var, b_var)), 'z')))
        sep_ordered.append(('ax-sep-impl-mem', cls.separation_instance(
            Implies(Membership(z_var, a_var), Membership(z_var, b_var)), 'z')))

        # Tier 3: quantified predicates (3 instances)
        sep_ordered.append(('ax-sep-union', cls.separation_instance(
            Exists('w', And(Membership(z_var, w_var),
                            Membership(w_var, a_var))), 'z')))
        sep_ordered.append(('ax-sep-subset', cls.separation_instance(
            ForAll('w', Implies(Membership(w_var, z_var),
                                Membership(w_var, a_var))), 'z')))
        sep_ordered.append(('ax-sep-exists-elt', cls.separation_instance(
            Exists('w', Membership(z_var, w_var)), 'z')))

        # ── Replacement: ordered simplest first (4 instances) ─────────
        rep_ordered: list = []

        rep_ordered.append(('ax-rep-id', cls.replacement_instance(
            Equality(y_var, z_var), Equality(y_var, w_var),
            input_var='y', output_var='z', alt_output_var='w')))
        rep_ordered.append(('ax-rep-const', cls.replacement_instance(
            Equality(z_var, a_var), Equality(w_var, a_var),
            input_var='y', output_var='z', alt_output_var='w')))
        rep_ordered.append(('ax-rep-into-a', cls.replacement_instance(
            Membership(z_var, a_var), Membership(w_var, a_var),
            input_var='y', output_var='z', alt_output_var='w')))
        rep_ordered.append(('ax-rep-elem-of', cls.replacement_instance(
            Membership(z_var, y_var), Membership(w_var, y_var),
            input_var='y', output_var='z', alt_output_var='w')))

        # Apply limits
        if sep_limit is not None:
            sep_ordered = sep_ordered[:sep_limit]
        if rep_limit is not None:
            rep_ordered = rep_ordered[:rep_limit]

        instances: Dict[str, Formula] = dict(sep_ordered + rep_ordered)
        logger.debug(f"Generated {len(instances)} schema instances "
                     f"(sep={len(sep_ordered)}, rep={len(rep_ordered)})")
        return instances
