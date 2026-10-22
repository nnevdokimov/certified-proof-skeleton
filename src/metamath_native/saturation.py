"""Target-free given-clause search and exact finite native-operator slices.

This is a separate policy, not a change to the frozen v1 experiments.  Typed
one-way subsumption carries a substitution witness and checks mandatory $d.
It removes search redundancy, not certificates from the persistent archive.
The finite reference enumerates MGU detachment and every existing set variable
plus one fresh variable for generalisation.  It does not enumerate all possible
Metamath substitutions, nor claim completeness for ZFC theoremhood.
"""
from collections import Counter
import heapq
import time

from .terms import (canonicalise, mapped_disjoint, size, sort, variable,
                    variables)
from .growth import family
from .index import PatternIndex


def statement(proposal):
    f, dv, _, args, parents, _ = proposal
    terms = list(args.values()) + [v for _, ps in parents for v in ps.values()]
    body, mapping = canonicalise(f, terms)
    mandatory = variables(body)
    frame = tuple(sorted({tuple(sorted((mapping[x], mapping[y]))) for x, y in dv
                          if mapping[x] in mandatory and mapping[y] in mandatory}))
    return body, frame


def instance_witness(general, general_dv, specific, specific_dv):
    """Return a one-way substitution, or None; target variables stay rigid."""
    mapping = {}; pending = [(general, specific)]
    while pending:
        a, b = pending.pop()
        if sort(a) != sort(b):
            return None
        if a[0] == 'v':
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
        elif b[0] == 'v' or a[0] != b[0] or len(a) != len(b):
            return None
        else:
            pending.extend(zip(a[1:], b[1:]))
    try:
        required = mapped_disjoint(general_dv, mapping)
    except ValueError:
        return None
    return mapping if required <= set(specific_dv) else None


def generalisations(lib, parent_id):
    """All existing setvar choices and one fresh alpha-class representative."""
    node = lib.nodes[parent_id]
    choices = sorted(v for v in variables(node.formula) if v[1] == 'setvar')
    choices.append(variable('setvar', 'fresh.generalisation'))
    for x in choices:
        args = {'ph': node.formula, 'x': x}
        dv = set(node.disjoint)
        parents = [(parent_id, {v[2]: v for v in variables(node.formula)})]
        yield ('wal', x, node.formula), dv, 'ax-gen', args, parents, node.origins


def proposals_for(lib, given, previous):
    """Each ordered pair once, including self-detachment; then generalise."""
    for other in [*previous, given]:
        if lib.nodes[other].formula[0] == 'wi':
            yield 'mp', (given, other)
        if other != given and lib.nodes[given].formula[0] == 'wi':
            yield 'mp', (other, given)
    yield 'gen', (given,)


def given_clause(lib, *, seconds=60, max_nodes=20000, max_selected=1000,
                 formula_size=120, proof_height=128, subsumption=True,
                 age_every=5, permitted=None, balanced=False, contractive=False):
    """Age/weight fair scheduling; limits explicitly prevent completeness claims.

    Every selected premise meets every previously selected premise.  Initial
    schemas receive priority; every fifth subsequent selection takes the oldest
    waiting statement.  A CPU guard may interrupt a given-clause batch, which
    remains marked incomplete in the trace.  No human corpus enters this API.
    """
    if (seconds <= 0 or max_nodes < len(lib.nodes) or max_selected <= 0 or
            formula_size <= 0 or proof_height <= 0 or age_every < 1):
        raise ValueError('Invalid finite search bounds')
    start = time.process_time(); deadline = start + seconds
    allowed = set(range(len(lib.nodes))) if permitted is None else set(permitted)
    if not allowed <= set(range(len(lib.nodes))):
        raise ValueError('Invalid permitted premise')
    initial = len(lib.nodes); waiting = set(); ages = []; weights = []
    processed = []; trace = []; counters = Counter(); cover = Counter()
    family_heaps = {f: [] for f in ('propositional','predicate_logic','set_theory')}
    family_cursor = 0
    # Root/size index avoids trying structurally impossible subsumers.
    index = PatternIndex()
    def enqueue(i):
        n = lib.nodes[i]; waiting.add(i)
        heapq.heappush(ages, i)
        weight=size(n.formula) + n.height / 8
        # A priority, never a logical rejection: explicit age selections still
        # process expansions. No external formula, label, or target enters it.
        if contractive and n.parents and size(n.formula)<max(
                size(lib.nodes[p].formula) for p,_ in n.parents):
            weight-=10000
        heapq.heappush(weights, (weight, i))
        heapq.heappush(family_heaps[family(n.formula,n.origins)],
                       (weight, i))
        index.add(n.formula,i)
    for i in sorted(allowed):
        enqueue(i)
    seeds = [i for i in sorted(allowed) if not lib.nodes[i].parents]
    seed_pos = 0; stop = 'selection_budget'
    for step in range(max_selected):
        if time.process_time() >= deadline:
            stop = 'cpu_budget'; break
        if len(lib.nodes) >= max_nodes:
            stop = 'node_budget'; break
        if not waiting:
            stop = 'saturated_under_declared_bounds'; break
        while seed_pos < len(seeds) and seeds[seed_pos] not in waiting:
            seed_pos += 1
        if seed_pos < len(seeds):
            given = seeds[seed_pos]; seed_pos += 1; mode = 'seed'
        else:
            age = (step - len(seeds)) % age_every == age_every - 1
            heap = ages if age else weights
            if balanced and not age:
                names=list(family_heaps)
                for offset in range(len(names)):
                    name=names[(family_cursor+offset)%len(names)]
                    h=family_heaps[name]
                    while h and h[0][1] not in waiting:heapq.heappop(h)
                    if h:
                        heap=h;family_cursor=(family_cursor+offset+1)%len(names)
                        break
            while (heap[0] if age else heap[0][1]) not in waiting:
                heapq.heappop(heap)
            entry = heapq.heappop(heap); given = entry if age else entry[1]
            mode = 'age' if age else ('balanced_weight' if balanced else 'weight')
        waiting.remove(given); added = []; complete = True
        for rule, pair in proposals_for(lib, given, processed):
            if time.process_time() >= deadline or len(lib.nodes) >= max_nodes:
                complete = False; break
            counters[rule + '_attempts'] += 1
            try:
                candidates = ([lib.propose_mp(*pair)] if rule == 'mp'
                              else generalisations(lib, pair[0]))
                for proposal in candidates:
                    counters['valid_proposals'] += 1
                    if size(proposal[0]) > formula_size:
                        counters['size_rejected'] += 1; continue
                    height = 1 + max(lib.nodes[i].height for i, _ in proposal[4])
                    if height > proof_height:
                        counters['height_rejected'] += 1; continue
                    body, dv = statement(proposal)
                    if (body, dv) in lib.known:
                        counters['known'] += 1; continue
                    witness_id = None
                    if subsumption:
                        for i in index.candidates(body):
                            n = lib.nodes[i]
                            if instance_witness(
                                    n.formula, n.disjoint, body, dv) is not None:
                                witness_id = i; break
                    if witness_id is not None:
                        cover[witness_id] += 1; counters['subsumed'] += 1; continue
                    i, new = lib.accept(proposal)
                    if new:
                        allowed.add(i); enqueue(i); added.append(i)
                        counters['admitted'] += 1
                    if len(lib.nodes) >= max_nodes:
                        complete = False; break
            except ValueError:
                counters['invalid_or_disjoint'] += 1
        trace.append(dict(step=step, given=given, mode=mode,
                          admitted=added, batch_complete=complete))
        if complete:
            processed.append(given)
        else:
            stop = 'node_budget' if len(lib.nodes) >= max_nodes else 'cpu_budget'
            break
    return dict(stop=stop, cpu_seconds=time.process_time()-start,
                initial_nodes=initial, final_nodes=len(lib.nodes),
                selected=processed, waiting=sorted(waiting), permitted=sorted(allowed),
                counters=dict(counters), covered_instances=dict(cover), trace=trace,
                limits=dict(seconds=seconds,max_nodes=max_nodes,max_selected=max_selected,
                    formula_size=formula_size,proof_height=proof_height,
                    subsumption=subsumption,age_every=age_every,balanced=balanced,
                    contractive=contractive))


def exact_slice(lib, depth, *, seconds=120, max_nodes=100000, formula_size=None):
    """Complete levels, or an explicit incomplete result; never silently truncate.

    H[0] contains the original 29 schemas. H[d+1] applies each MGU-MP pair and
    each generalisation choice to H[d]. Size filtering, when requested, applies
    to every intermediate conclusion and forms part of the reference operator.
    No subsumption, active cap, beam, candidate cap, or admission score applies.
    """
    if depth < 0 or seconds <= 0 or max_nodes < len(lib.nodes):
        raise ValueError('Invalid finite reference bound')
    if any(n.parents for n in lib.nodes):
        raise ValueError('Exact slices must start from original schemas')
    start = time.process_time(); deadline = start + seconds
    completed = 0; previous_count = 0; levels = []; arcs = set(); hyper = set()
    counters = Counter(); stop = 'complete'
    for level in range(1, depth + 1):
        count = len(lib.nodes); complete = True; before = len(lib.nodes)
        def offer(p):
            if formula_size is not None and size(p[0]) > formula_size:
                counters['size_excluded_by_definition'] += 1; return
            i, _ = lib.accept(p)
            ps = tuple(j for j, _ in p[4])
            hyper.add((i, p[2], ps))
            arcs.update((j, i) for j in ps if j != i)
        for minor in range(count):
            for major in range(count):
                if minor < previous_count and major < previous_count:
                    continue
                if lib.nodes[major].formula[0] != 'wi':
                    continue
                if time.process_time() >= deadline or len(lib.nodes) >= max_nodes:
                    complete = False; break
                counters['mp_attempts'] += 1
                try: offer(lib.propose_mp(minor, major))
                except ValueError: counters['invalid_or_disjoint'] += 1
            if not complete: break
        if complete:
            for parent in range(previous_count, count):
                if time.process_time() >= deadline or len(lib.nodes) >= max_nodes:
                    complete = False; break
                for p in generalisations(lib, parent): offer(p)
        levels.append(dict(level=level, input_nodes=count, output_nodes=len(lib.nodes),
                           new_nodes=len(lib.nodes)-before, complete=complete))
        if not complete:
            stop = 'cpu_budget' if time.process_time() >= deadline else 'node_budget'
            break
        completed = level; previous_count = count
    return dict(status='complete' if completed == depth else 'incomplete', stop=stop,
                completed_depth=completed, requested_depth=depth, levels=levels,
                cpu_seconds=time.process_time()-start, counters=dict(counters),
                arcs=sorted(arcs), inference_hyperedges=sorted(hyper),
                operator='typed MGU-MP; Gen on every existing setvar and one fresh representative',
                formula_size=formula_size, nodes=len(lib.nodes),
                excludes='arbitrary unifier instances; arbitrary axiom instances; other Metamath rules')
