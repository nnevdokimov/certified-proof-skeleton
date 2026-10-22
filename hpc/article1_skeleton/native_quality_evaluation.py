"""Evaluate frozen proof interfaces, not their builders' immediate output.

Reuse OnlineGrowth.choose_pair and the native kernel; reset policy/queue/reuse.
Certificates may verify a frozen anchor but may NEVER supply search premises,
subsumers, novelty labels or a duplicate oracle. An inaccessible stored theorem
becomes available only after a new, explicitly checked derivation from available
parents. Its new witness joins the standalone Metamath export.
"""
from collections import Counter
from dataclasses import asdict, replace as dc_replace
import math
import time

from metamath_native.index import PatternIndex
from metamath_native.online_skeleton import OnlineGrowth, OnlineConfig
from metamath_native.proofs import Node, Library, ordered_vars, proof_text, SYNTAX_FORMALS
from metamath_native.saturation import statement, instance_witness
from metamath_native.terms import (size, variables, variable, canonicalise,
                                   replace, render)
from metamath_native.growth import family


def normalised_witness(lib, proposal):
    """Same normalisation as Library.add; check a duplicate's NEW proof too."""
    f, dv, assertion, arguments, parents, origins = proposal
    expressions = list(arguments.values()) + [t for _, args in parents for t in args.values()]
    conclusion, mapping = canonicalise(f, expressions)
    if any(v not in mapping for pair in dv for v in pair):
        raise ValueError('Unaccounted proof variable')
    proof_dv = tuple(sorted({tuple(sorted((mapping[a], mapping[b]))) for a, b in dv}))
    mandatory = variables(conclusion)
    frame = tuple(pair for pair in proof_dv if set(pair) <= mandatory)
    node = Node(len(lib.nodes), conclusion, frame, proof_dv, assertion,
        {k: replace(t, mapping) for k, t in arguments.items()},
        tuple((i, {k: replace(t, mapping) for k, t in args.items()}) for i, args in parents),
        1 + max((lib.nodes[i].height for i, _ in parents), default=-1),
        tuple(sorted(set(origins))))
    lib.check(node)
    return node


class FrozenContinuation(OnlineGrowth):
    """Access-control adapter around the existing sampler and native rules."""
    def __init__(self, lib, allowed, seed, *, formula_size=160, proof_height=256,
                 max_events=6000):
        cfg = OnlineConfig(attempts=10**9, max_nodes=len(lib.nodes)+max_events+1,
                           formula_size=formula_size, proof_height=proof_height)
        super().__init__(lib, cfg, seed)
        self.allowed = set(allowed)
        self.roots = {n.id for n in lib.nodes if not n.parents}
        if not self.roots <= self.allowed or not self.allowed <= set(range(len(lib.nodes))):
            raise ValueError('Frozen view must contain the original basis and valid IDs')
        self.initial_allowed = sorted(self.allowed)
        self.anchors = sorted(self.allowed-self.roots)
        self.search_known = {lib.nodes[i].key: i for i in self.allowed}
        self.bodies = {lib.nodes[i].formula for i in self.allowed}
        self.index = PatternIndex()
        for i in sorted(self.allowed):
            self.index.add(lib.nodes[i].formula, i)
        # Inherited hidden construction history cannot bias the neutral sampler.
        lib.reuse.clear()
        self.masks = {i: 1 << j for j, i in enumerate(self.anchors)}
        self.masks.update({i: 0 for i in self.roots})
        self.depths = {i: 0 for i in self.allowed}
        self.steps = {i: 1 for i in self.allowed}
        self.events, self.witnesses = [], []
        self.max_events = max_events
        self.growth_cpu = 0.

    def available_ids(self):
        return sorted(self.allowed)

    def offer(self, proposal):
        """No archive lookup can reject an otherwise new SEARCH statement."""
        parents = tuple(p for p, _ in proposal[4])
        if not set(parents) <= self.allowed:
            raise AssertionError('Hidden certificate supplied a search premise')
        if size(proposal[0]) > self.config.formula_size:
            self.counts['size_rejected'] += 1; return None
        if 1+max(self.lib.nodes[p].height for p in parents) > self.config.proof_height:
            self.counts['height_rejected'] += 1; return None
        body, dv = statement(proposal); key = body, dv
        if key in self.search_known:
            self.counts['search_known'] += 1; return None
        for i in self.index.candidates(body):
            self.counts['subsumption_checks'] += 1
            n = self.lib.nodes[i]
            if instance_witness(n.formula, n.disjoint, body, dv) is not None:
                self.counts['subsumed_by_available'] += 1; return None
        promotion = None
        if key in self.lib.known:
            # Validate the new proof BEFORE using the old theorem's statement.
            witness = normalised_witness(self.lib, proposal)
            nid = self.lib.known[key]
            promotion = len(self.witnesses)
            self.witnesses.append(witness)
            self.counts['support_rediscovered'] += 1
            self.lib.reuse.update(set(parents))
            fresh = False
        else:
            nid, fresh = self.lib.accept(proposal)
            if not fresh:
                raise AssertionError('Native key and certificate key differ')
            self._register(self.lib.nodes[nid])
            self.counts['fresh_certificates'] += 1
        new_body = body not in self.bodies
        self.allowed.add(nid); self.search_known[key] = nid; self.bodies.add(body)
        if not fresh:
            self.index.add(body, nid)
        mask = 0
        for p in set(parents):
            mask |= self.masks[p]
        self.masks[nid] = mask
        self.depths[nid] = 1+max(self.depths[p] for p in parents)
        self.steps[nid] = 1+sum(self.steps[p] for p in parents)
        event = dict(node=nid, parents=list(parents), assertion=proposal[2],
            promotion=promotion, new_body=new_body, depth=self.depths[nid],
            anchor_mask=mask, proof_tree_reference_steps=self.steps[nid],
            family=family(body, self.lib.nodes[nid].origins))
        self.events.append(event); self.counts['available_admissions'] += 1
        return event

    def step(self, policy):
        if policy not in ('uniform', 'mixed'):
            raise ValueError('Unknown frozen continuation policy')
        self.counts['opportunities'] += 1
        try:
            if self.rng.random() < self.config.gen_probability:
                ids = self.available_ids()
                parent = ids[int(self.rng.integers(len(ids)))]
                n = self.lib.nodes[parent]
                choices = sorted(v for v in variables(n.formula) if v[1] == 'setvar')
                choices.append(variable('setvar', 'fresh.generalisation'))
                x = choices[int(self.rng.integers(len(choices)))]
                proposal = (('wal', x, n.formula), set(n.disjoint), 'ax-gen',
                    {'ph': n.formula, 'x': x},
                    [(parent, {v[2]: v for v in variables(n.formula)})], n.origins)
                self.counts['gen_calls'] += 1
            else:
                modes = ('uniform', 'bridge', 'contract', 'frontier')
                mode = 'uniform' if policy == 'uniform' else modes[((self.counts['opportunities']-1)//16) % len(modes)]
                pair = self.choose_pair(mode)
                if pair is None:
                    self.counts['empty_pool'] += 1; return None
                self.seen_pairs.add(pair); self.counts['mp_calls'] += 1
                proposal = self.lib.propose_mp(*pair)
            self.counts['valid_proposals'] += 1
            return self.offer(proposal)
        except ValueError:
            self.counts['invalid_or_disjoint'] += 1
            return None

    def export_library(self):
        """Unique main archive + separately labelled rediscovery witnesses."""
        lib = Library(self.lib.basis)
        lib.nodes = list(self.lib.nodes)
        for witness in self.witnesses:
            lib.nodes.append(dc_replace(witness, id=len(lib.nodes)))
        for n in lib.nodes:
            lib.check(n)
        return lib


class CoverageMonitor:
    """Only an observer: never passed to sampler, features, reward or admission."""
    def __init__(self, targets):
        self.targets = targets
        self.first_exact, self.first_cover, self.covering = {}, {}, {}
        self.mappings = {}
        self.by_root = {}
        self.exact_keys = {}
        for j, t in enumerate(targets):
            self.by_root.setdefault(t['body'][0], []).append(j)
            self.exact_keys[t['body'], t['disjoint']] = j
        self.cpu = 0.

    def observe(self, lib, nid, at):
        begin = time.process_time(); n = lib.nodes[nid]
        exact = self.exact_keys.get(n.key)
        if exact is not None and exact not in self.first_exact:
            self.first_exact[exact] = at
        ids = range(len(self.targets)) if n.formula[0] == 'v' else self.by_root.get(n.formula[0], [])
        for j in ids:
            if j in self.first_cover:
                continue
            t = self.targets[j]
            witness = instance_witness(n.formula, n.disjoint, t['body'], t['disjoint'])
            if witness is not None:
                self.first_cover[j] = at; self.covering[j] = nid
                self.mappings[j] = witness
        self.cpu += time.process_time()-begin

    def readout(self, horizon, basis_covered):
        exact = sorted(i for i, t in self.first_exact.items() if t <= horizon)
        covered = sorted(i for i, t in self.first_cover.items() if t <= horizon)
        nontrivial = set(range(len(self.targets)))-set(basis_covered)
        reached = sorted(set(covered)&nontrivial)
        initial = {i for i, t in self.first_cover.items() if t == 0}
        censored_mean = sum(min(self.first_cover.get(i, horizon), horizon) for i in nontrivial)/max(1, len(nontrivial))
        return dict(cpu_horizon=horizon, exact=len(exact), covered=len(covered),
            beyond_basis=len(reached), gained_beyond_initial=len(set(reached)-initial),
            beyond_basis_ids=reached, exact_ids=exact, covered_ids=covered,
            restricted_mean_first_cover_cpu=censored_mean,
            normalised_coverage_auc=1-censored_mean/horizon if horizon else 0.)


def continuation_quality(engine, horizon, source_union):
    events = [e for e in engine.events if e['cpu'] <= horizon]
    novel, multistep, complementary, used = set(), set(), set(), 0
    families = Counter(); old_uses = Counter()
    anchor_set = set(engine.anchors)
    for e in events:
        for p in set(e['parents']) & anchor_set:
            old_uses[p] += 1
        n = engine.lib.nodes[e['node']]
        if e['assertion'] != 'ax-mp' or n.formula in source_union:
            continue
        novel.add(n.formula)
        if e['depth'] >= 2:
            multistep.add(n.formula); used |= e['anchor_mask']
            families[e['family']] += 1
            if e['anchor_mask'].bit_count() >= 2:
                complementary.add(n.formula)
    return dict(future_mp_bodies_outside_source_union=len(novel),
        multistep_mp_bodies_outside_source_union=len(multistep),
        complementary_multistep_mp_bodies=len(complementary),
        old_anchors_in_multistep_outside_union=used.bit_count(),
        old_anchors_directly_used=len(old_uses), old_anchors_directly_used_twice=sum(v>=2 for v in old_uses.values()),
        maximum_continuation_depth=max((e['depth'] for e in events), default=0),
        multistep_event_families=dict(families), available_admissions=len(events))


def target_proof_text(lib, monitor):
    """Compile every claimed covered target, including one-way $d witnesses."""
    text, _ = proof_text(lib)
    names = {}
    for j in monitor.covering:
        for v in variables(monitor.targets[j]['body']):
            names[v] = variable(v[1], 'quality.'+v[2])
    ordered = sorted(set(names.values()), key=lambda v: (v[1], v[2]))
    lines = [text]
    if ordered:
        lines.append('$v '+' '.join(v[2] for v in ordered)+' $.')
    declarations = {v: 'quality.f.'+v[2] for v in ordered}
    for v in ordered:
        lines.append(declarations[v]+' $f '+v[1]+' '+v[2]+' $.')
    def syntax(t):
        if t[0] == 'v':
            return [declarations[t]]
        a = lib.by_label[t[0]]
        mapping = dict(zip(SYNTAX_FORMALS[t[0]], t[1:]))
        proof = []
        for _, (_, name) in a.floats:
            proof.extend(syntax(mapping[name]))
        substituted = tuple(y for token in a.expression[1:] for y in
                            (render(mapping[token]) if token in mapping else (token,)))
        if substituted != render(t):
            raise ValueError('Syntax/source mismatch in target compiler')
        return proof+[a.label]
    for j, nid in sorted(monitor.covering.items()):
        target = monitor.targets[j]; node = lib.nodes[nid]
        proof = []
        for v in ordered_vars(node.formula):
            proof.extend(syntax(replace(monitor.mappings[j][v], names)))
        proof.append(node.label)
        lines.append('${')
        for a, b in target['disjoint']:
            lines.append('$d '+names[a][2]+' '+names[b][2]+' $.')
        lines.append('quality.target.'+str(j)+' $p |- '+' '.join(render(replace(target['body'], names)))+
                     ' $= '+' '.join(proof)+' $.')
        lines.append('$}')
    return '\n'.join(lines)+'\n'
