"""Exact closed proof-fragment library; no generalized inference rules.

A fragment is an existing multi-step first proof and its certified conclusion.
The cost model is the number of symbols in a forest of expanded proof terms:
one symbol per axiom, rule application, or library reference, plus one header
per library definition. This is NOT runtime, and NOT compression relative to
the already shared global proof DAG. Greedy selection has no optimality claim.
"""
from collections import Counter, defaultdict

from skeleton_local import ancestors, size


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


class ProofCoverage:
    """Incremental sets of distinct proofs containing each proper multi-step ancestor.

    Bit j denotes the j-th derived proof in this append-only archive. A diamond
    or a repeated parent still contributes just one bit for that proof.
    """
    def __init__(self):
        self.heights = []
        self.proofs = 0
        self.first_use = {}
        self.reach = {}
        self.statement_sizes = []
        self.consumers = {}

    def update(self, records):
        if len(records) < len(self.heights):
            raise ValueError('Coverage requires an append-only archive')
        for root in range(len(self.heights), len(records)):
            parents = records[root][3]
            formula = records[root][1]
            self.statement_sizes.append(size(formula) if formula is not None else None)
            self.heights.append(1 + max((self.heights[p] for p in parents), default=-1))
            if not parents:
                continue
            root_bit = 1 << self.proofs
            for parent in set(parents):
                if self.heights[parent] >= 2 and self.statement_sizes[parent] is not None:
                    self.consumers.setdefault(parent, []).append(self.proofs)
            pending = list(parents)
            visited = set()
            while pending:
                node = pending.pop()
                if self.heights[node] < 2 or node in visited:
                    continue
                visited.add(node)
                if node in self.reach:
                    self.reach[node] |= root_bit
                elif node in self.first_use:
                    self.reach[node] = (1 << self.first_use.pop(node)) | root_bit
                else:
                    self.first_use[node] = self.proofs
                pending.extend(records[node][3])
            self.proofs += 1

    def _compact_frontier(self, visible, instance_uses):
        """Keep all ties on the size / distinct immediate consumer Pareto frontier."""
        by_size = {}
        if instance_uses is None:
            counts = {n: len(children) if visible is None else sum(i in visible for i in children)
                      for n, children in self.consumers.items()}
        else:
            counts = instance_uses
        for node, uses in counts.items():
            if uses:
                by_size.setdefault(self.statement_sizes[node], []).append((node, uses))
        credit = {}
        best_smaller = 0
        for size in sorted(by_size):
            group = by_size[size]
            best = max(uses for _, uses in group)
            if best > best_smaller:
                credit.update((node, best) for node, uses in group if uses == best)
                best_smaller = best
        return credit

    def freeze(self, records, slots=None, *, visible_proofs=None, sort_of=None, restricted=()):
        """At export, count opportunities to replace a certified *used* premise.

        Match a whole formula in one direction with consistent, sorted bindings.
        Sources with side conditions retain only their recorded direct uses.
        A consumer using several matching premises contributes one vote.
        """
        self.update(records)
        visible = None if visible_proofs is None else set(visible_proofs)
        index_visible = None if visible is None else sum(1 << i for i in visible)
        restricted = set(restricted)
        premises = []
        uses = []
        direct = {}
        for n, consumers in self.consumers.items():
            mask = sum(1 << i for i in consumers if visible is None or i in visible)
            if mask:
                premises.append(records[n][1])
                uses.append(mask)
                direct[n] = mask.bit_count()
        index = _PremiseIndex(premises, sort_of)
        # A derived ordinal need not equal its node id (axioms can be interleaved).
        ordinal = -1
        for n in range(len(records)):
            if not records[n][3]:
                continue
            ordinal += 1
            if self.heights[n] < 2 or n in restricted or records[n][1] is None:
                continue
            if visible is not None and ordinal not in visible:
                # A certified internal lemma must occur within a visible proof.
                reach = self.reach.get(n, 0)
                if not (reach & index_visible) and self.first_use.get(n) not in visible:
                    continue
            matches = index.matches(records[n][1])
            consumers = 0
            while matches:
                low = matches & -matches
                consumers |= uses[low.bit_length() - 1]
                matches ^= low
            if consumers:
                direct[n] = consumers.bit_count()
        return self.select(slots, visible_proofs=visible, _instance_uses=direct)

    def select(self, slots=None, *, visible_proofs=None, _instance_uses=None):
        """Cover proof roots and credit direct reuse of compact, undominated claims.

        A root gets at most one coverage vote across the bank. Each frontier
        claim also gets one vote per distinct direct consumer. This preserves
        direct-use evidence when another ancestor already covers the same roots.
        Optional visibility refers to derived proof ordinals, excluding axioms.
        """
        if slots is not None and (type(slots) is not int or slots < 0):
            raise ValueError('Use a nonnegative integer or None for fragment slots')
        selected = []
        trace = []
        visible = None if visible_proofs is None else set(visible_proofs)
        if visible is None:
            remaining = (1 << self.proofs) - 1
        else:
            if any(i < 0 or i >= self.proofs for i in visible):
                raise ValueError('Unknown visible proof ordinal')
            remaining = sum(1 << i for i in visible)
        total = remaining.bit_count()
        credit = self._compact_frontier(visible, _instance_uses)
        candidates = {}
        for node, reached in self.reach.items():
            mask = reached & remaining
            if mask.bit_count() >= 2 or node in credit:
                candidates[node] = mask
        for node in credit.keys() - candidates.keys():
            mask = self.reach.get(node)
            if mask is None:
                mask = 1 << self.first_use[node] if node in self.first_use else 0
            candidates[node] = mask & remaining
        while candidates and (slots is None or len(selected) < slots):
            winner = None
            best = (0, 0, 0, 0)
            for node, reached in candidates.items():
                additional = (reached & remaining).bit_count()
                score = (additional + credit.get(node, 0), additional, -self.heights[node], -node)
                if score > best:
                    winner, best = node, score
            if winner is None:
                break
            remaining &= ~candidates.pop(winner)
            selected.append(winner)
            trace.append(dict(root=winner, additional_proofs=best[1],
                              frontier_consumers=credit.get(winner, 0), marginal_votes=best[0]))
        return sorted(selected), dict(training_proofs=total, candidates_count=len(candidates)+len(selected),
            frontier_claims=len(credit), slots=slots, trace=trace, covered_proofs=total - remaining.bit_count(),
            use_evidence='schema instances' if _instance_uses is not None else 'recorded edges',
            scope='Distinct proper proof roots and distinct consumers on the statement-size/reuse frontier.')


class _PremiseIndex:
    """One bit per premise, indexed by constructor, sort and repeated subterm."""
    def __init__(self, premises, sort_of):
        self.all = (1 << len(premises)) - 1
        self.sort_of = sort_of
        self.kinds = defaultdict(int)
        self.sorts = defaultdict(int)
        self.terms = defaultdict(dict)
        self.equalities = {}

        def add(t, path, bit):
            self.kinds[path, t[0], len(t)] |= bit
            if sort_of is not None:
                self.sorts[path, sort_of(t)] |= bit
            terms = self.terms[path]
            terms[t] = terms.get(t, 0) | bit
            if t[0] != 'v':
                for i, child in enumerate(t[1:]):
                    add(child, path + (i,), bit)

        for i, premise in enumerate(premises):
            add(premise, (), 1 << i)

    def matches(self, pattern):
        remaining = self.all
        bindings = {}
        todo = [(pattern, ())]
        while todo and remaining:
            t, path = todo.pop()
            if t[0] != 'v':
                remaining &= self.kinds.get((path, t[0], len(t)), 0)
                todo.extend((child, path + (i,)) for i, child in enumerate(t[1:]))
                continue
            if self.sort_of is not None:
                remaining &= self.sorts.get((path, self.sort_of(t)), 0)
            if t in bindings:
                pair = (bindings[t], path)
                if pair not in self.equalities:
                    left = self.terms.get(pair[0], {})
                    right = self.terms.get(path, {})
                    equal = 0
                    for subterm, mask in left.items():
                        equal |= mask & right.get(subterm, 0)
                    self.equalities[pair] = equal
                remaining &= self.equalities[pair]
            else:
                bindings[t] = path
        return remaining


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
