"""Predict proof-support corridors before MP; retain the production prover.

The classifier learns structural patterns of first-proof steps supporting an
offline compression dictionary. It never sees a proposed MP conclusion. This
prototype does not claim to generate only skeleton vertices or to prove that
an omitted inference lacks value.
"""
from collections import Counter, defaultdict
import math
import random
import time

from certificate import encode_formula
from fragment_engine import make_fragment_generator
from fragment_policy import select_fragments
from skeleton_local import ancestors, size


def shape(t, depth=2):
    if t[0] == 'v':
        return 'v'
    if not depth:
        return t[0]
    return t[0] + '(' + ','.join(shape(x, depth-1) for x in t[1:]) + ')'


def describe(t, height):
    todo = [t]; leaves = 0; variables = set()
    while todo:
        x = todo.pop()
        if x[0] == 'v':
            leaves += 1; variables.add(x[1])
        else:
            todo.extend(x[1:])
    return dict(shape=shape(t), n=size(t), h=min(height, 8),
                repetition=min(3, int(4*(leaves-len(variables))/max(1, leaves))),
                ant=shape(t[1]) if t[0] == 'i' else '-',
                con=shape(t[2]) if t[0] == 'i' else '-')


def pair_features(minor, major):
    """Only premise syntax and already known first-certificate heights."""
    bucket = lambda n: min(9, int(math.log2(max(1, n))))
    return (f'm={minor["shape"]}', f'a={major["ant"]}', f'c={major["con"]}',
            f'ms={bucket(minor["n"])}', f'Ms={bucket(major["n"])}',
            f'mh={minor["h"]}', f'Mh={major["h"]}',
            f'mr={minor["repetition"]}', f'Mr={major["repetition"]}',
            f'cross={minor["shape"]}:{major["ant"]}',
            f'ratio={min(4, 2*minor["n"]//max(1, major["n"]))}')


def extract(records, slots, candidate_cap=64, roots_count=256):
    roots = [i for i, r in records.items() if r[3]][-roots_count:]
    chosen, report = select_fragments(records, roots, slots, candidate_cap)
    return chosen, report


def fit(corpora, slots=16, candidate_cap=64, roots_count=256):
    """Positive labels denote observed support of the training dictionary.

    Other recorded first-proof MP steps form the negative class. They remain
    valid proofs; the label means only absence from that selected support.
    Equal class priors prevent rarity alone from suppressing every candidate.
    """
    counts = [Counter(), Counter()]; totals = [0, 0]; reports = []
    for records in corpora:
        chosen, report = extract(records, slots, candidate_cap, roots_count)
        support = ancestors(records, chosen)
        heights = {}; descriptions = {}
        for i, row in records.items():
            heights[i] = 1 + max((heights[p] for p in row[3]), default=-1)
            descriptions[i] = describe(row[1], heights[i])
            if row[2] == 'MP':
                label = int(i in support)
                counts[label].update(pair_features(*(descriptions[p] for p in row[3])))
                totals[label] += 1
        reports.append(dict(anchors=chosen, support=len(support), selection=report))
    if not all(totals):
        raise ValueError('Training needs both proof-support and other MP steps')
    vocabulary = defaultdict(set)
    for f in set(counts[0]) | set(counts[1]):
        vocabulary[f.split('=', 1)[0]].add(f)
    weights = {}
    for f in set(counts[0]) | set(counts[1]):
        v = len(vocabulary[f.split('=', 1)[0]]) + 1
        weights[f] = math.log((counts[1][f]+1)/(totals[1]+v)) - math.log((counts[0][f]+1)/(totals[0]+v))
    unknown = {k: math.log((totals[0]+len(v)+1)/(totals[1]+len(v)+1)) for k, v in vocabulary.items()}
    return dict(schema='proof-corridor-naive-bayes/v1', weights=weights, unknown=unknown,
                class_counts=totals, teacher_slots=slots, candidate_cap=candidate_cap,
                roots_count=roots_count, training=reports,
                labels='Observed first-proof ancestry of an offline compression dictionary, not semantic truth or causal utility.',
                features='Premise syntax and existing first-proof heights only; no proposed conclusion, formula ID or human theorem list.')


def score(model, fs):
    return sum(model['weights'].get(f, model['unknown'].get(f.split('=', 1)[0], 0.)) for f in fs)


def choose_indices(scores, fraction, rng, exploration=.125):
    if not 0 < fraction <= 1:
        raise ValueError('Invalid retained-pair fraction')
    n = len(scores); k = min(n, max(1, math.ceil(n*fraction))) if n else 0
    if k == n:
        return list(range(n))
    explore = min(k, max(1, math.floor(k*exploration))) if k > 1 else 0
    rank = sorted(range(n), key=lambda i: (-scores[i], i))
    selected = set(rank[:k-explore])
    if explore:
        selected.update(rng.sample([i for i in range(n) if i not in selected], explore))
    return sorted(selected)  # retain the production traversal order


def make_gated_engine(archive, cfg, model):
    g = make_fragment_generator(archive, [], cfg)
    base = type(g)

    class Gated(base):
        def _add_schema(self, formula, depth, rule, premise_ids, **kwargs):
            added = super()._add_schema(formula, depth, rule, premise_ids, **kwargs)
            if added:
                e = self._id_to_entry[self._next_id-1]
                self.gate_height[e.id] = 1 + max((self.gate_height[p] for p in premise_ids), default=-1)
            return added

        def _description(self, entry):
            if entry.id not in self.gate_descriptions:
                self.gate_descriptions[entry.id] = describe(encode_formula(entry.formula), self.gate_height[entry.id])
            return self.gate_descriptions[entry.id]

        def _mp_pass(self, *, minors, depth, all_candidates, wildcard_candidates,
                     candidates_by_tag, metrics):
            if cfg['gate'] == 'all' or depth <= cfg['gate_warmup']:
                for minor in minors:
                    buckets = self._candidate_lists_for_minor(minor.formula,
                        all_candidates=all_candidates, wildcard_candidates=wildcard_candidates,
                        candidates_by_tag=candidates_by_tag)
                    n = sum(map(len, buckets))
                    self.gate_counters['compatible_seen'] += n
                    self.gate_counters['admitted'] += n
                return super()._mp_pass(minors=minors, depth=depth, all_candidates=all_candidates,
                    wildcard_candidates=wildcard_candidates, candidates_by_tag=candidates_by_tag, metrics=metrics)
            new = 0
            for minor in minors:
                if len(self._entries) >= self._schema_cap:
                    break
                candidates = [c for bucket in self._candidate_lists_for_minor(minor.formula,
                    all_candidates=all_candidates, wildcard_candidates=wildcard_candidates,
                    candidates_by_tag=candidates_by_tag) for c in bucket]
                self.gate_counters['compatible_seen'] += len(candidates)
                t = time.process_time()
                if cfg['gate'] == 'random':
                    scores = [self.gate_rng.random() for _ in candidates]
                else:
                    md = self._description(minor)
                    scores = []
                    for major in candidates:
                        fs = pair_features(md, self._description(major.entry))
                        if fs not in self.gate_score_cache:
                            self.gate_score_cache[fs] = score(model, fs)
                        scores.append(self.gate_score_cache[fs])
                indices = choose_indices(scores, cfg['gate_fraction'], self.gate_rng,
                    exploration=.125 if cfg['gate'] == 'learned' else 0.)
                self.gate_policy_cpu += time.process_time()-t
                selected = [candidates[i] for i in indices]
                if cfg.get('defer_skipped', False):
                    self.queue_skipped(minor.id, candidates, set(indices))
                self.gate_counters['admitted'] += len(selected)
                self.gate_counters['skipped_before_unification'] += len(candidates)-len(selected)
                wildcard = [c for c in selected if c.antecedent_is_wildcard]
                tagged = defaultdict(list)
                for c in selected:
                    if not c.antecedent_is_wildcard:
                        tagged[c.antecedent_tag].append(c)
                new += super()._mp_pass(minors=[minor], depth=depth, all_candidates=selected,
                    wildcard_candidates=wildcard, candidates_by_tag=tagged, metrics=metrics)
            return new

    g.__class__ = Gated
    g.gate_height = {}; g.gate_descriptions = {}; g.gate_score_cache = {}
    g.gate_rng = random.Random(cfg['seed']+99173)
    g.gate_counters = Counter(); g.gate_policy_cpu = 0.
    return g
