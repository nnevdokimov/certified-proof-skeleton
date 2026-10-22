"""Extra compression ABOVE the accumulated generator's existing dictionary.

The objective and paid definitions come from fragment_policy.costs unchanged.
Greedy additions stop when no eligible fragment has positive joint marginal
gain. No target dictionary cardinality applies unless an explicit ablation
limit is requested. This codes a forest of first proofs; it does NOT claim
physical compression of the already shared certificate DAG or saved CPU.
"""
from collections import Counter
import time

from fragment_policy import costs
from skeleton_local import ancestors


def encoding_state(records, roots, dictionary):
    roots = list(roots); selected = set(dictionary)
    support = sorted(ancestors(records, roots + sorted(selected)))
    inline = {}; bodies = {}; heights = {}
    for i in support:
        parents = records[i][3]
        if any(p >= i for p in parents):
            raise ValueError('Proof records require earlier parents')
        bodies[i] = 1 + sum(inline[p] for p in parents)
        inline[i] = 1 if i in selected else bodies[i]
        heights[i] = 1 + max((heights[p] for p in parents), default=-1)
    encoded = sum(inline[i] for i in roots) + sum(bodies[i] + 1 for i in selected)
    visits = Counter(roots)
    # Each definition expands its root instruction exactly once, even when
    # references to that definition stop expansion elsewhere in the forest.
    for i in selected:
        visits.update(records[i][3])
    for i in reversed(support):
        if i not in selected:
            for parent in records[i][3]:
                visits[parent] += visits[i]
    return dict(support=support, bodies=bodies, heights=heights,
                visits=visits, encoded_symbols=encoded)


def marginal_gain(state, node):
    # r copies of a body become r references; the new body and header cost
    # body+1. Thus gain = r*(body-1) - (body+1), including nested definitions.
    body = state['bodies'][node]
    return (state['visits'][node] - 1) * (body - 1) - 2


def compress_above(records, roots, base_dictionary=(), *, extra_limit=None):
    start = time.process_time()
    if extra_limit is not None and (type(extra_limit) is not int or extra_limit < 0):
        raise ValueError('Non-negative explicit ablation limit required')
    roots = list(roots); base = set(base_dictionary); selected = set(base)
    before = costs(records, roots, base)
    state = encoding_state(records, roots, selected)
    assert state['encoded_symbols'] == before['encoded_symbols']
    additions = []; scans = 0; stop = 'no_positive_marginal_gain'
    while True:
        eligible = [i for i in state['support'] if i not in selected
                    and state['heights'][i] >= 2 and state['visits'][i] >= 2]
        scans += len(eligible)
        scored = [(marginal_gain(state, i), i) for i in eligible]
        positive = [(gain, i) for gain, i in scored if gain > 0]
        if not positive:
            break
        if extra_limit is not None and len(additions) >= extra_limit:
            stop = 'explicit_ablation_limit_with_positive_gain_remaining'
            break
        gain, winner = max(positive, key=lambda x: (x[0], -x[1]))
        prior_cost = state['encoded_symbols']
        selected.add(winner)
        after = encoding_state(records, roots, selected)
        assert prior_cost - after['encoded_symbols'] == gain
        additions.append(dict(root=winner, marginal_saved_symbols=gain,
                              occurrences=state['visits'][winner],
                              body_symbols=state['bodies'][winner]))
        state = after
    checked = costs(records, roots, selected)
    assert checked['encoded_symbols'] == state['encoded_symbols']
    assert before['encoded_symbols'] - checked['encoded_symbols'] == sum(x['marginal_saved_symbols'] for x in additions)
    remaining = max((marginal_gain(state, i) for i in state['support']
                     if i not in selected and state['heights'][i] >= 2
                     and state['visits'][i] >= 2), default=0)
    return sorted(selected), dict(
        schema='extra-proof-compression/v1', base_dictionary=sorted(base),
        added_dictionary=[x['root'] for x in additions], dictionary=sorted(selected),
        roots=roots, root_count=len(roots), initial=before, final=checked,
        additional_saved_symbols=before['encoded_symbols'] - checked['encoded_symbols'],
        additional_fraction_of_previous_encoding=(before['encoded_symbols'] - checked['encoded_symbols']) / max(1, before['encoded_symbols']),
        trace=additions, extra_limit=extra_limit, stop=stop,
        largest_remaining_eligible_gain=remaining, candidate_scans=scans,
        cpu_seconds=time.process_time() - start,
        scope='Same first-proof forest coding model as fragment_policy; paid definitions/overlaps. Not physical shared-DAG compression, saved search time, or global optimum.')


def continuation_working_set(dictionary, base_working, roots, reuse, cap):
    """Same finite activity budget; dictionary size itself remains unconstrained.

    Prefer compressed dictionary anchors, then fill unused capacity from the
    original working set. Reuse counts precede the continuation. All excluded
    statements remain in the certified source archive, never hidden premises.
    """
    if cap < 1:
        raise ValueError('Positive activity cap required')
    roots = set(roots)
    ranked = sorted(set(dictionary) - roots, key=lambda i: (-reuse[i], i))
    ranked += sorted(set(base_working) - roots - set(ranked), key=lambda i: (-reuse[i], i))
    return sorted(roots | set(ranked[:cap]))
