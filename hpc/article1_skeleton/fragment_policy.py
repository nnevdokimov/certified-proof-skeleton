"""Exact closed proof-fragment library; no generalized inference rules.

A fragment is an existing multi-step first proof and its certified conclusion.
The cost model is the number of symbols in a forest of expanded proof terms:
one symbol per axiom, rule application, or library reference, plus one header
per library definition. This is NOT runtime, and NOT compression relative to
the already shared global proof DAG. Greedy selection has no optimality claim.
"""
from collections import Counter

from skeleton_local import ancestors


def costs(records, roots, library=()):
    """Exact acyclic grammar encoding; nested definitions charged once each."""
    roots = list(roots)
    selected = set(library)
    support = ancestors(records, roots + sorted(selected))
    expanded = {}; compact = {}; definitions = {}
    for i in sorted(support):
        parents = records[i][3]
        expanded[i] = 1 + sum(expanded[p] for p in parents)
        body = 1 + sum(compact[p] for p in parents)
        if i in selected:
            definitions[i] = body + 1  # definition header, not a free lemma
            compact[i] = 1
        else:
            compact[i] = body
    original = sum(expanded[i] for i in roots)
    encoded = sum(compact[i] for i in roots) + sum(definitions.values())
    return dict(expanded_symbols=original, encoded_symbols=encoded,
                definition_symbols=sum(definitions.values()),
                net_saved_symbols=original-encoded)


def select_fragments(records, roots, slots=8, candidate_cap=32):
    """Positive marginal forest compression, including definition overhead.

    All evidence precedes the next growth epoch. No forced quota; an empty
    library is a valid outcome. Overlapping fragments are rescored jointly.
    """
    if slots < 0 or candidate_cap < 0:
        raise ValueError('Negative library bound')
    roots = list(roots)
    support = ancestors(records, roots)
    occurrences = Counter(roots)
    heights = {}; tree_size = {}
    for i in sorted(support):
        parents = records[i][3]
        heights[i] = 1 + max((heights[p] for p in parents), default=-1)
        tree_size[i] = 1 + sum(tree_size[p] for p in parents)
    for i in sorted(support, reverse=True):
        for parent in records[i][3]:
            occurrences[parent] += occurrences[i]
    candidates = [i for i in support if heights[i] >= 2 and occurrences[i] >= 2]
    candidates.sort(key=lambda i: (-(occurrences[i]-1)*(tree_size[i]-1), i))
    candidates = candidates[:candidate_cap]
    selected = []; trace = []
    current = costs(records, roots)
    while len(selected) < slots:
        winner = None
        for i in candidates:
            if i in selected:
                continue
            proposal = costs(records, roots, selected+[i])
            gain = proposal['net_saved_symbols']-current['net_saved_symbols']
            if gain > 0 and (winner is None or (gain, -i) > (winner[0], -winner[1])):
                winner = (gain, i, proposal)
        if winner is None:
            break
        gain, i, current = winner
        selected.append(i)
        trace.append(dict(root=i, marginal_saved_symbols=gain,
                          observed_occurrences=occurrences[i],
                          expanded_proof_symbols=tree_size[i]))
    return sorted(selected), dict(training_roots=roots, candidates=candidates,
        slots=slots, candidate_cap=candidate_cap, trace=trace, **current,
        scope='Exact closed first-proof fragments; proof-tree coding cost, not runtime or DAG compression.')


def future_savings(records, roots, library):
    """Out-of-selection-epoch savings; definition cost shown separately.

    A root is a newly admitted first proof. Expand its root instruction even
    when it is itself a library member: only antecedent fragments are reused.
    Counts are exact under this proof-term model, not saved search operations.
    """
    roots = list(roots); selected = set(library)
    support = ancestors(records, roots)
    raw = {}; encoded = {}; refs = Counter()
    for i in sorted(support):
        parents = records[i][3]
        raw[i] = 1 + sum(raw[p] for p in parents)
        encoded[i] = 1 if i in selected else 1 + sum(encoded[p] for p in parents)
    visits = Counter(roots)
    # Roots are new, so cannot be in a library frozen earlier.
    if selected.intersection(roots):
        raise ValueError('Future roots must postdate the library')
    for i in sorted(support, reverse=True):
        if i in selected:
            refs[i] += visits[i]
        else:
            for p in records[i][3]: visits[p] += visits[i]
    expanded = sum(raw[i] for i in roots)
    compact = sum(encoded[i] for i in roots)
    return dict(expanded_symbols=expanded, reference_encoded_symbols=compact,
                saved_symbols=expanded-compact, references=dict(sorted(refs.items())),
                distinct_reused_fragments=sum(v > 0 for v in refs.values()),
                direct_future_uses=sum(p in selected for i in roots for p in records[i][3]))
