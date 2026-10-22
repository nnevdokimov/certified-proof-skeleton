"""Small independent replay kernel for propositional schemas and normal modal K.

No generator, canonicalizer, unifier, or axiom factory is imported. The contract
is one first derivation per node, not completeness or all alternative proofs.
Terms: (v, name), (i, antecedent, consequent), (n, term), (b, term).
"""
from collections import Counter


class InvalidProof(ValueError):
    pass


def term(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise InvalidProof('Malformed formula')
    tag = value[0]
    if tag == 'v' and len(value) == 2 and isinstance(value[1], (str, int)):
        return ('v', str(value[1]))
    if tag in ('n', 'b') and len(value) == 2:
        return (tag, term(value[1]))
    if tag == 'i' and len(value) == 3:
        return ('i', term(value[1]), term(value[2]))
    raise InvalidProof('Unsupported formula constructor')


def canonical(t):
    names = {}
    def visit(x):
        if x[0] == 'v':
            return ('v', names.setdefault(x[1], len(names)))
        return (x[0], *(visit(y) for y in x[1:]))
    return visit(t)


def scoped(t, side):
    if t[0] == 'v':
        return ('v', (side, t[1]))
    return (t[0], *(scoped(y, side) for y in t[1:]))


def mp(minor, major):
    if major[0] != 'i':
        raise InvalidProof('MP major is not an implication')
    minor, major = scoped(minor, 0), scoped(major, 1)
    subst = {}
    def resolve(x):
        if x[0] == 'v':
            return resolve(subst[x[1]]) if x[1] in subst else x
        return (x[0], *(resolve(y) for y in x[1:]))
    def occurs(name, x):
        return x[1] == name if x[0] == 'v' else any(occurs(name, y) for y in x[1:])
    pending = [(minor, major[1])]
    while pending:
        a, b = (resolve(x) for x in pending.pop())
        if a == b:
            continue
        if a[0] != 'v' and b[0] == 'v':
            a, b = b, a
        if a[0] == 'v':
            if occurs(a[1], b):
                raise InvalidProof('Occurs check failed')
            subst[a[1]] = b
        elif a[0] != b[0]:
            raise InvalidProof('MP premises do not unify')
        else:
            pending.extend(zip(a[1:], b[1:]))
    return canonical(resolve(major[2]))


def axioms(substrate):
    p, q, r = ('v', 'p'), ('v', 'q'), ('v', 'r')
    I = lambda a, b: ('i', a, b)
    N = lambda a: ('n', a)
    B = lambda a: ('b', a)
    forms = [I(p, I(q, p)), I(I(p, I(q, r)), I(I(p, q), I(p, r))),
             I(I(N(p), N(q)), I(q, p))]
    if substrate == 'prop':
        return dict(zip(('ax-1', 'ax-2', 'ax-3'), map(canonical, forms)))
    if substrate == 'modal':
        return dict(zip(('A1', 'A2', 'A3', 'K'),
                        map(canonical, forms + [I(B(I(p, q)), I(B(p), B(q)))])))
    raise InvalidProof('Unsupported substrate')


class Checker:
    def __init__(self, substrate):
        self.substrate = substrate
        self.axioms = axioms(substrate)
        self.nodes = {}
        self.heights = {}
        self.rules = Counter()

    def accept(self, node_id, formula, rule, parents):
        if not isinstance(node_id, int) or node_id in self.nodes:
            raise InvalidProof('Duplicate or invalid node id')
        f = canonical(term(formula))
        if self.substrate == 'prop':
            def has_box(x):
                return x[0] == 'b' or (x[0] != 'v' and any(has_box(y) for y in x[1:]))
            if has_box(f):
                raise InvalidProof('Modal formula in propositional archive')
        if any(not isinstance(p, int) or p not in self.nodes or p >= node_id for p in parents):
            raise InvalidProof('Missing, cyclic, or non-earlier parent')
        if rule in self.axioms:
            if parents or f != self.axioms[rule]:
                raise InvalidProof('Invalid axiom or axiom label')
        elif rule == 'MP':
            if len(parents) != 2 or mp(*(self.nodes[p] for p in parents)) != f:
                raise InvalidProof('Invalid MP conclusion')
        elif rule == 'Nec' and self.substrate == 'modal':
            if len(parents) != 1 or canonical(('b', self.nodes[parents[0]])) != f:
                raise InvalidProof('Invalid necessitation')
        else:
            raise InvalidProof(f'Unsupported rule: {rule}')
        self.nodes[node_id] = f
        self.heights[node_id] = 1 + max((self.heights[p] for p in parents), default=-1)
        self.rules[rule] += 1

    def summary(self):
        return dict(status='verified', nodes=len(self.nodes), rules=dict(self.rules),
                    max_proof_height=max(self.heights.values(), default=0),
                    scope='First-derivation replay: named axioms, schema MP and modal K necessitation; no completeness claim.')


def encode_formula(f):
    """Thin AST adapter only; the kernel above operates exclusively on tuples."""
    tag = type(f).__name__
    if tag == 'PropVar':
        return ('v', f.name)
    if tag in ('Not', 'Box'):
        return ('n' if tag == 'Not' else 'b', encode_formula(f.inner))
    if tag == 'Implies':
        return ('i', encode_formula(f.antecedent), encode_formula(f.consequent))
    raise InvalidProof(f'Unsupported AST class {tag}')
