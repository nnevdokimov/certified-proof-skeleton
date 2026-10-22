"""Sealed, target-blind structural selection and POI holdout utilities.

This module does not train on human targets. A finite, common rule catalogue
comes from MP trees, equality partitions, and the checked deduction compiler.
Only a source proof DAG determines adaptive ranks. A held-out score cannot
alter these ranks. Statistical units are source RNG states, not goals or views.
"""
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
import math
import random
import re
import tarfile

from metamath_native.proofs import Library, freeze
from metamath_native.terms import (canonicalise, replace, size, syntax,
                                   variable, variables)
from native_rule_synthesis import (ProofFactory, ClosedCompiler,
    abstract_programme, boundary_partitions, boundary_count, internal_nodes)


METHODS = ('core', 'fixed', 'random', 'adaptive', 'fixed_all')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


@lru_cache(None)
def grammar(n):
    """All ordered binary MP trees with exactly n internal nodes."""
    if n == 0:
        return ('H',)
    return tuple((a, b) for k in range(n)
                 for a in grammar(k) for b in grammar(n-1-k))


def profile_key(shape, partition):
    return json.dumps([shape, partition], separators=(',', ':'))


def structural_profile(lib, maximum_mp=3, start=0):
    """Count actual instantiated boundary equalities, weighted by past reuse.

    Compose parent substitutions down the tree. Equal node IDs alone do NOT
    imply equal boundary formulae. Gen and all non-MP nodes stop an open tree.
    Future test events and human formulae never enter this function.
    """
    uses = Counter(p for n in lib.nodes for p, _ in n.parents)
    counts = Counter()

    def leaves(nid, shape, environment):
        n = lib.nodes[nid]
        if shape == 'H':
            return [replace(n.formula, environment)]
        if n.assertion != 'ax-mp':
            raise ValueError('MP tree crosses a non-MP inference')
        result = []
        for child_shape, (parent, mapping) in zip(shape, n.parents):
            child = lib.nodes[parent]
            env = {v: replace(mapping[v[2]], environment)
                   for v in variables(child.formula)}
            result.extend(leaves(parent, child_shape, env))
        return result

    for n in lib.nodes[start:]:
        if n.assertion != 'ax-mp':
            continue
        for k in range(1, maximum_mp+1):
            for shape in grammar(k):
                try:
                    boundary = leaves(n.id, shape, {})
                except ValueError:
                    continue
                ids = {}; partition = []
                for term in boundary:
                    partition.append(ids.setdefault(term, len(ids)))
                counts[profile_key(shape, tuple(partition))] += 1+uses[n.id]
    return dict(sorted(counts.items()))


def make_catalogue(basis, maximum_mp=3, formula_size=160, proof_height=256):
    """Compile a source-independent grammar once, including a complete short set.

    Deduplicate ACTUAL delivered schemas, not requested instances of a more
    general stored schema. Candidate quotas later count distinct delivered
    rules. Certificate support never automatically receives search rights.
    """
    if maximum_mp not in (2, 3):
        raise ValueError('The frozen grammar supports two or three MP nodes')
    lib = Library(basis); lib.initialise()
    factory = ProofFactory(); compiler = ClosedCompiler(lib, formula_size, proof_height)
    candidates = {}; invalid = 0; bounded = 0
    for k in range(maximum_mp+1):
        for shape in grammar(k):
            for partition in boundary_partitions(boundary_count(shape)):
                try:
                    proof, holes = abstract_programme(shape, partition, factory)
                except ValueError:
                    invalid += 1; continue
                for order in permutations(sorted(holes)):
                    closed = proof
                    for j in order:
                        closed = factory.discharge(closed, j, holes[j])
                    try:
                        nid = compiler.compile(closed)
                    except ValueError:
                        bounded += 1; continue
                    if not lib.nodes[nid].parents:
                        continue
                    rule = candidates.setdefault(nid, dict(node=nid,
                        formula=lib.nodes[nid].formula, minimum_mp=k, features=[]))
                    key = profile_key(shape, partition)
                    if key not in rule['features']:
                        rule['features'].append(key)
                    rule['minimum_mp'] = min(k, rule['minimum_mp'])
    rows = []
    for r in candidates.values():
        r['features'].sort(); r['id'] = digest(r['formula'])
        r['formula_size'] = size(r['formula'])
        rows.append(r)
    rows.sort(key=lambda r: (r['minimum_mp'], r['formula_size'], r['id']))
    return lib, dict(schema='source-independent-native-catalogue/v1',
        maximum_mp=maximum_mp, rules=rows,
        short_ids=[r['id'] for r in rows if r['minimum_mp'] <= 2],
        invalid_partitions=invalid, bounded_requests=bounded,
        scope='Finite checked MP/deduction grammar, not all useful structural laws')


def rank_rules(catalogue, profile):
    """A frozen intrinsic heuristic, not an optimal or goal-trained policy."""
    scores = {}
    for r in catalogue['rules']:
        value = 0.
        for key in r['features']:
            shape, _ = json.loads(key)
            # Repeated future-use fragments in the SOURCE's past DAG amortise
            # a rule body; the formula length charges a simple description cost.
            value += internal_nodes(freeze(shape))*math.log1p(profile.get(key, 0))
        scores[r['id']] = value/max(1, r['formula_size'])
    return sorted(scores, key=lambda i: (-scores[i], i)), scores


def select_rules(catalogue, profile, fixed_ids, method, quota, seed):
    if method not in METHODS or quota < 1:
        raise ValueError('Invalid frozen method or resource quota')
    common = set(catalogue['short_ids'])
    extra = sorted(r['id'] for r in catalogue['rules'] if r['id'] not in common)
    if len(extra) < quota:
        raise ValueError('Catalogue cannot supply the predeclared quota')
    scores = {}
    if method == 'core':
        selected = []
    elif method == 'fixed_all':
        selected = extra
    elif method == 'fixed':
        selected = list(fixed_ids)
        if len(selected) != quota or len(set(selected)) != quota or not set(selected) <= set(extra):
            raise ValueError('Invalid locked structural rule set')
    elif method == 'random':
        selected = random.Random(seed).sample(extra, quota)
    else:
        order, scores = rank_rules(catalogue, profile)
        selected = [i for i in order if i in set(extra)][:quota]
    return sorted(common|set(selected)), dict(method=method,
        extra_ids=sorted(selected), extra_quota=quota, scores=scores,
        selection_sha256=digest(sorted(selected)))


def merge_rules(lib, rule_lib, roots):
    """Import only selected certificates; return their actual search IDs."""
    projected, allowed, _ = rule_lib.project(sorted(roots))
    mapping = {}
    for n in projected.nodes:
        parents = [(mapping[p], args) for p, args in n.parents]
        # A canonical duplicate keeps the destination's existing first proof.
        # Its axiom origins can differ from the alternative catalogue proof.
        # Descendants must describe the proof they ACTUALLY cite after mapping,
        # not copy metadata from the abandoned alternative. Keep kernel checks.
        origins = (set().union(*(set(lib.nodes[p].origins) for p, _ in parents))
                   if parents else {n.assertion})
        mapping[n.id], _ = lib.add(n.formula, n.proof_disjoint, n.assertion,
                                   n.arguments, parents, sorted(origins))
    return [mapping[i] for i in allowed]


def unfold_prop(t):
    """Exact primitive expansion of df-bi/df-an/df-or; no logical quotient."""
    tag = t[0]
    if tag == 'v':
        if t[1] != 'wff':
            raise ValueError('Not propositional')
        return t
    a = [unfold_prop(x) for x in t[1:]]
    if tag in ('wi', 'wn'):
        return syntax(tag, *a)
    if tag == 'wa':
        return syntax('wn', syntax('wi', a[0], syntax('wn', a[1])))
    if tag == 'wo':
        return syntax('wi', syntax('wn', a[0]), a[1])
    if tag == 'wb':
        return unfold_prop(syntax('wa', syntax('wi', *a), syntax('wi', a[1], a[0])))
    raise ValueError('No frozen propositional expansion for '+tag)


def parse_poi(text):
    """Strict POI TPTP subset: universally quantified i/n terms only.

    Reject unknown functions, includes, extra conjectures and free variables.
    Do not execute TPTP or Prolog source and do not extract tar members to disk.
    """
    text = re.sub(r'%[^\n]*', '', text)
    if re.search(r'\binclude\s*\(', text):
        raise ValueError('External TPTP include forbidden')
    starts = list(re.finditer(r'fof\([^,]+,\s*conjecture\s*,', text))
    if len(starts) != 1:
        raise ValueError('Expected exactly one POI conjecture')
    tail = text[starts[0].end():]
    match = re.fullmatch(r'\s*\(\s*!\s*\[([^\]]*)\]\s*:\s*is_theorem\((.*)\)\s*\)\s*\)\s*\.\s*', tail, re.S)
    if not match:
        raise ValueError('Unexpected POI conjecture syntax')
    names = [x.strip() for x in match[1].split(',') if x.strip()]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Z][A-Za-z0-9_]*', n) for n in names):
        raise ValueError('Invalid universally quantified variables')
    raw = match[2]; tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]*|[(),]', raw)
    if ''.join(tokens) != re.sub(r'\s+', '', raw):
        raise ValueError('Unsupported POI term characters')
    pos = 0
    def term():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError('Truncated POI term')
        token = tokens[pos]; pos += 1
        if token in names:
            return variable('wff', token)
        if token not in ('i', 'n') or pos >= len(tokens) or tokens[pos] != '(':
            raise ValueError('Unknown POI function or free variable')
        pos += 1; args = [term()]
        if token == 'i':
            if pos >= len(tokens) or tokens[pos] != ',':
                raise ValueError('Missing POI argument')
            pos += 1; args.append(term())
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError('Unclosed POI term')
        pos += 1
        return syntax('wi' if token == 'i' else 'wn', *args)
    result = term()
    if pos != len(tokens):
        raise ValueError('Trailing POI term tokens')
    return canonicalise(result)[0]


def poi_holdout(archive, exposed, maximum_size=160):
    old = set(); unexpanded = 0
    for row in exposed:
        try:
            old.add(canonicalise(unfold_prop(freeze(row['body'])))[0])
        except ValueError:
            unexpanded += 1
    groups = {}; excluded = Counter(); members = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if (not member.isfile() or not re.fullmatch(r'poi_tptp/poi_[0-9]+_[A-Za-z0-9_.-]+\.p', member.name)
                    or member.size > 1_000_000):
                raise ValueError('Unexpected or unsafe POI archive member')
            members += 1
            text = tar.extractfile(member).read().decode('utf-8')
            # Nine author-supplied POI files retain uninterpreted truth or
            # falsehood constants. They do NOT belong to our declared i/n/wff
            # language. Do not silently turn constants into metavariables or
            # assume a definition absent from the TPTP axiom set.
            clean = re.sub(r'%[^\n]*', '', text)
            goal = clean[clean.index('conjecture'):]
            if re.search(r'\b(truth|falsehood)\b', goal):
                excluded['named_constants_outside_frozen_language'] += 1
                continue
            body = parse_poi(text)
            if body in old:
                excluded['previous_exact_after_definitions'] += 1; continue
            if size(body) > maximum_size:
                excluded['predeclared_size_bound'] += 1; continue
            key = digest(body)
            row = groups.setdefault(key, dict(body=body, disjoint=(), labels=[],
                family='prop', corpus_id=key))
            row['labels'].append(member.name.rsplit('/', 1)[-1][:-2])
    if not groups:
        raise ValueError('Empty decontaminated holdout')
    for row in groups.values():
        row['labels'].sort()
    return sorted(groups.values(), key=lambda r: r['corpus_id']), dict(
        archive_members=members, retained_unique=len(groups), excluded=dict(excluded),
        exposed_unique_expanded=len(old), exposed_not_pure_prop=unexpanded,
        language='Pure i/n with universally quantified wff variables; author files containing uninterpreted truth/falsehood constants are explicitly excluded, not reinterpreted',
        decontamination='Alpha-exact after df-bi/df-an/df-or expansion against ALL supplied previous scored targets; not semantic equivalence or all one-way instances',
        independence='Previously unscored external POI targets from the SAME Metamath lineage. Not an independent mathematical source or unseen-to-authors guarantee.')


def paired_inference(differences):
    """Exact one-sided paired sign-flip test; source blocks are the units.

    This needs exchangeability of signs under the null; RNG source replicates
    on one basis do not establish generality over independent axiom systems.
    """
    values = list(differences)
    if not values or len(values) > 20 or any(not math.isfinite(x) for x in values):
        raise ValueError('Invalid paired source differences')
    observed = sum(values); tail = 0
    for signs in product((-1, 1), repeat=len(values)):
        tail += sum(x*s for x, s in zip(values, signs)) >= observed-1e-12
    mean = observed/len(values)
    return dict(n=len(values), mean=mean, p_one_sided=tail/(2**len(values)),
        wins=sum(x > 0 for x in values), ties=sum(x == 0 for x in values),
        losses=sum(x < 0 for x in values), differences=values)


def holm(pvalues):
    result = {}; previous = 0.
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    for j, key in enumerate(order):
        previous = max(previous, min(1., (len(order)-j)*pvalues[key]))
        result[key] = previous
    return result
