"""Typed adapter for the accumulated single-search mechanisms, NOT a new prover.

Reuse unchanged: OpportunityQueue, exact closed-fragment coding/selection,
delayed Portfolio, and the existing learned-half gate on its supported syntax.
Native MP, Gen, types, canonicalisation and $d stay in the existing native kernel.

The frozen propositional model sees only pure implication/negation/wff terms;
predicate/ZFC pairs use declared intrinsic fallback features. No claim of
validated transfer of the old model to ZFC follows from this adapter.
"""
from collections import Counter
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'src')]
from deferred_gate import OpportunityQueue
from fragment_policy import select_fragments, future_savings
from adaptive_policy import Portfolio, ACTIONS
from skeleton_gate import describe, pair_features, score, choose_indices
from metamath_native.online_skeleton import OnlineGrowth
from metamath_native.online_skeleton import profile_loss


def propositional_term(t):
    """Syntax-only lossless adapter on the common supported sublanguage."""
    if t[0] == 'v':
        return ('v', t[2]) if t[1] == 'wff' else None
    tags = {'wi': 'i', 'wn': 'n'}
    if t[0] not in tags:
        return None
    children = [propositional_term(x) for x in t[1:]]
    return None if any(x is None for x in children) else (tags[t[0]], *children)


def fragment_records(lib):
    # Existing cost helpers inspect graph parents, never formula syntax.
    return {n.id: (n.id, n.formula, n.assertion, tuple(p for p, _ in n.parents))
            for n in lib.nodes}


class CumulativeGrowth(OnlineGrowth):
    def __init__(self, lib, config, seed, *, model, fragments=True,
                 controller='half_replay', active_cap=128, queue_cap=32768,
                 rot_patience=8, fragment_slots=4, gate=True):
        if min(active_cap, queue_cap, rot_patience) < 1 or fragment_slots > active_cap or fragment_slots < 0:
            raise ValueError('Invalid accumulated-mechanism bounds')
        super().__init__(lib, config, seed)
        self.roots = {n.id for n in lib.nodes if not n.parents}
        self.live = set(range(len(lib.nodes)))
        self.active_cap, self.rot_patience = active_cap, rot_patience
        self.fragment_slots, self.fragments_enabled = fragment_slots, fragments
        self.gate_enabled = gate
        self.model = model
        self.queue = OpportunityQueue(queue_cap)
        self.portfolio = Portfolio(controller, seed, block=8, delay=32)
        self.gate_rng = random.Random(seed + 99173)
        self.block_id = 0; self.retired = {}; self.last_use = Counter()
        self.activated = Counter({i: 0 for i in self.live})
        self.fragment_library = []; self.fragment_gains = {}; self.leased_until = {}
        self.description_cache = {}; self.lifecycle = Counter(); self.owners = {}
        self.fragment_reports = []; self.future_reports = []; self.snapshots = []
        self.current_action = 2
        # A retired certificate may support proofs, but may not silently act
        # as a live subsumer/premise. Filter candidates through the live set.
        underlying = self.index
        owner = self
        class LiveIndex:
            def add(self, formula, nid):
                underlying.add(formula, nid)
            def candidates(self, formula):
                return [i for i in underlying.candidates(formula) if i in owner.live]
        self.index = LiveIndex()
        self._enforce_cap()

    def available_ids(self):
        return sorted(self.live)

    def _enforce_cap(self):
        excess = len(self.live - self.roots) - self.active_cap
        if excess <= 0:
            return
        victims = sorted(self.live - self.roots, key=lambda i: (
            self.block_id < self.leased_until.get(i, 0),
            self.activated[i] == self.block_id,
            self.last_use[i], self.lib.reuse[i], i))[:excess]
        for i in victims:
            self.live.remove(i); self.retired[i] = self.block_id
        self.lifecycle['active_cap_retired'] += len(victims)

    def _descriptor(self, i):
        if i not in self.description_cache:
            t = propositional_term(self.lib.nodes[i].formula)
            self.description_cache[i] = None if t is None else describe(t, self.lib.nodes[i].height)
        return self.description_cache[i]

    def _pair_score(self, minor, major):
        a, b = self._descriptor(minor), self._descriptor(major)
        if self.model is not None and a is not None and b is not None:
            self.lifecycle['frozen_prop_model_scores'] += 1
            value = score(self.model, pair_features(a, b))
        else:
            self.lifecycle['native_fallback_scores'] += 1
            gain = sum(math.log1p(max(0, self.fragment_gains.get(i, 0))) for i in (minor, major))
            value = math.log1p(self.lib.reuse[minor] + self.lib.reuse[major]) + .25 * gain - .1 * math.log1p(self.sizes[major])
        if self.config.topology_weight:
            d = [self.lib.reuse[i] for i in range(len(self.lib.nodes))]
            old = profile_loss(d, alpha=self.config.target_alpha, xmin=self.config.target_xmin)
            for p in set((minor, major)):
                d[p] += 1
            predicted = profile_loss(d, alpha=self.config.target_alpha, xmin=self.config.target_xmin)
            value += self.config.topology_weight * (old - predicted)
            self.lifecycle['topology_prediction_scores'] += 1
        return value

    def choose_pair(self, mode):
        gate, fraction, quota = ACTIONS[self.current_action]
        if quota and self.counts['opportunities'] % 4 == 0:
            for pair in self.queue.pop_ready(self.live, 1, 64):
                if pair not in self.seen_pairs:
                    self.lifecycle['replay_attempts'] += 1
                    return pair
                self.lifecycle['cached_replay_skipped'] += 1
        active = sorted(self.live)
        majors = [i for i in active if self.lib.nodes[i].formula[0] == 'wi']
        if not majors:
            return None
        pairs = []
        for _ in range(self.config.pair_pool):
            p = active[int(self.rng.integers(len(active)))], majors[int(self.rng.integers(len(majors)))]
            self.counts['cheap_pairs_scored'] += 1
            if p in self.seen_pairs or p in pairs:
                continue
            minor, major = p; a, b = self.lib.nodes[minor].formula, self.lib.nodes[major].formula[1]
            if a[0] != 'v' and b[0] != 'v' and a[0] != b[0]:
                self.counts['rigid_root_mismatch'] += 1
                continue
            pairs.append(p)
        if not pairs:
            return None
        scores = [self._pair_score(*p) for p in pairs]
        if not self.gate_enabled:
            fraction = 1.
        kept = choose_indices(scores, fraction, self.gate_rng, exploration=.125)
        selected = kept[int(self.rng.integers(len(kept)))]
        for i, pair in enumerate(pairs):
            if i != selected:
                self.queue.add(pair)
        self.lifecycle['pre_inference_gate_offers'] += len(pairs)
        self.lifecycle['pre_inference_gate_kept'] += len(kept)
        return pairs[selected]

    def before_block(self, block):
        self.block_id = block
        work = self.counts['mp_calls'] + self.counts['gen_calls']
        self.current_action = self.portfolio.before(block, work)
        # Rot only deactivates; all certificates and obligations remain.
        victims = [i for i in self.live - self.roots
                   if block - max(self.activated[i], self.last_use[i]) >= self.rot_patience
                   and self.lib.reuse[i] == 0 and block >= self.leased_until.get(i, 0)]
        for i in victims:
            self.live.remove(i); self.retired[i] = block
        self.lifecycle['rot_retired'] += len(victims)
        _, _, quota = ACTIONS[self.current_action]
        reserved = self.queue.reservations(self.live, self.retired, block, 2, 4) if quota else []
        shared = list(dict.fromkeys(sorted(self.fragment_library) + reserved))[:self.fragment_slots]
        for i in shared:
            if i in self.live or block - self.retired.get(i, -100000) < 2:
                continue
            self.live.add(i); self.activated[i] = block
            self.leased_until[i] = block + 6
            self.lifecycle['reactivated'] += 1
        self._enforce_cap()
        self.frozen_fragments = list(self.fragment_library)

    def accepted(self, nid, parents):
        if not set(parents) <= self.live:
            raise AssertionError('An inactive certificate silently supplied a search premise')
        self.live.add(nid); self.activated[nid] = self.block_id
        for p in parents:
            self.last_use[p] = self.block_id
            if p in self.owners:
                owner, when = self.owners[p]
                self.portfolio.reuse(owner, when, self.block_id, min(10 ** 6, self.tree_steps[p]))
        self.owners[nid] = self.portfolio.created(), self.block_id
        self._enforce_cap()

    def after_block(self, block, first_new):
        records = fragment_records(self.lib)
        future = future_savings(records, range(first_new, len(self.lib.nodes)), self.frozen_fragments)
        self.future_reports.append(dict(block=block, **future))
        if self.fragments_enabled and block % 4 == 0:
            roots = [n.id for n in self.lib.nodes if n.parents][-128:]
            chosen, report = select_fragments(records, roots, self.fragment_slots, 16)
            self.fragment_library = chosen
            self.fragment_gains = {x['root']: x['marginal_saved_symbols'] for x in report['trace']}
            for i in chosen:
                self.leased_until[i] = block + 6
            self.fragment_reports.append(dict(block=block, chosen=chosen, report=report))
        self.portfolio.after(block, self.counts['mp_calls'] + self.counts['gen_calls'])
        self.snapshots.append(dict(block=block, live=sorted(self.live),
                                   fragments=list(self.fragment_library), action=self.current_action,
                                   pending=len(self.queue.pairs), nodes=len(self.lib.nodes)))

    def run(self, **kwargs):
        result, trajectory = super().run(**kwargs)
        result['accumulated'] = dict(
            lifecycle=dict(self.lifecycle), deferred=self.queue.summary(),
            controller=self.portfolio.export(), active_cap=self.active_cap,
            active_final=sorted(self.live), fragment_library=self.fragment_library,
            fragment_reports=self.fragment_reports, future_reports=self.future_reports,
            snapshots=self.snapshots,
            reused_unchanged=['deferred_gate.OpportunityQueue', 'skeleton_gate.choose_indices',
                              'skeleton_gate.describe/pair_features/score on supported propositional pairs',
                              'fragment_policy.select_fragments/future_savings', 'adaptive_policy.Portfolio'],
            new_adapter='Typed native lifecycle and pre-inference integration; its efficiency needs a new test',
            mathematical_contract='All admitted nodes retain native certificates and $d. Deactivation does not erase knowledge.',
            exclusions=['No external theorem targets or human labels', 'No multi-island restart or hard phase switch',
                        'No claim that old propositional model has validated ZFC transfer'])
        return result, trajectory
