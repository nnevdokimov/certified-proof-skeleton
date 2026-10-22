#!/usr/bin/env python3
"""Read-only independent replay of a sealed structural-rule holdout.

Recompute permissions, substitution witnesses, earliest hits, common residuals,
and paired tests. Never repair a run, change a rule, or overwrite its scores.
The producer's aggregate outcome does not suffice for a confirmation claim.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'hpc/article1_skeleton')]
import metamath_native_skeleton as io
from metamath_native.proofs import Node, freeze
from metamath_native.index import PatternIndex
from metamath_native.saturation import instance_witness
from metamath_native.terms import replace, mapped_disjoint
from native_quality_evaluation import normalised_witness
from native_holdout import (METHODS, structural_profile, select_rules, merge_rules,
                           paired_inference, holm, poi_holdout)
from compression_overlay import compress_above
from native_cumulative_policy import fragment_records
from run_native_holdout import (verify_seal, check_files, complete_receipts,
                               scientific_environment, PROCESS_ENV)


def require(value, message):
    if not value:
        raise ValueError(message)


def close(x, y):
    return abs(x-y) <= 1e-9*max(1., abs(x), abs(y))


def read_node(r):
    return Node(r['id'], freeze(r['formula']), freeze(r['disjoint']), freeze(r['proof_disjoint']),
        r['assertion'], {k:freeze(v) for k,v in r['arguments'].items()},
        tuple((p, {k:freeze(v) for k,v in m.items()}) for p,m in r['parents']),
        r['height'], freeze(r['origins']))


def replay_permissions(initial, final, events, witnesses):
    require(io.encoded(final.json_nodes()[:len(initial['nodes'])]) == io.encoded(initial['nodes']), 'Initial certificate prefix changed')
    allowed = set(initial['search_ids']); times = {i:0. for i in allowed}
    next_id = len(initial['nodes']); last = 0.; seen = set()
    for event in events:
        nid = event['node']; parents = event['parents']
        require(last <= event['cpu'], 'Growth timestamps not monotone'); last = event['cpu']
        require(nid not in allowed and set(parents) <= allowed, 'Hidden search premise or duplicate event')
        n = final.nodes[nid]
        if event['promotion'] is None:
            require(nid == next_id, 'Unrecorded fresh certificate'); next_id += 1
            witness = n
        else:
            j = event['promotion']; require(j not in seen, 'Witness reused for promotion'); seen.add(j)
            witness = witnesses[j]
            require(witness.id == next_id and nid < next_id and witness.key == n.key, 'Invalid hidden-support rediscovery')
            final.check(witness)
        require(list(p for p,_ in witness.parents) == parents and witness.assertion == event['assertion'], 'Wrong event proof')
        if witness.assertion == 'ax-mp':
            actual = normalised_witness(final, final.propose_mp(*parents))
            require((actual.formula, actual.disjoint, actual.arguments, actual.parents, actual.proof_disjoint) ==
                    (witness.formula, witness.disjoint, witness.arguments, witness.parents, witness.proof_disjoint), 'Wrong MP witness')
        else:
            require(witness.assertion == 'ax-gen', 'Unknown future inference')
            final.check(witness)
        allowed.add(nid); times[nid] = event['cpu']
    require(next_id == len(final.nodes), 'Unrecorded certificate suffix')
    require(seen == set(range(len(witnesses))), 'Unclaimed promotion witnesses')
    return allowed, times


def first_hits(targets, final, times):
    """Target-oriented indexed search, independent of the streaming observer."""
    index = PatternIndex()
    for i in sorted(times):
        index.add(final.nodes[i].formula, i)
    cover = {}; exact = {}
    for j, target in enumerate(targets):
        for i in index.candidates(target['body']):
            n = final.nodes[i]
            if n.key == (target['body'], target['disjoint']):
                exact[j] = min(exact.get(j, float('inf')), times[i])
            if instance_witness(n.formula, n.disjoint, target['body'], target['disjoint']) is not None:
                cover[j] = min(cover.get(j, float('inf')), times[i])
    return cover, exact


def main(a):
    run = a.run.resolve(); protocol = verify_seal(run, a.seal, include_holdout=True)
    if (Path(__file__).resolve() != run/'code/scripts/audit_native_holdout.py'
            or any(__import__('os').environ.get(k) != v for k,v in PROCESS_ENV.items())):
        frozen = run/'code/scripts/audit_native_holdout.py'
        subprocess.run([sys.executable, str(frozen), *sys.argv[1:]], check=True,
                       env=scientific_environment()); return
    receipts = complete_receipts(run, protocol)
    opened = json.loads((run/'EVALUATION_OPENED.json').read_text())
    require(opened['blind_receipts'] == receipts and opened['seal_sha256'] == a.seal, 'Opening commitment differs')
    receipt = json.loads((run/'EVALUATION_RECEIPT.json').read_text())
    check_files(run/'evaluation/results', receipt['files'])
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    targets = [dict(t, body=freeze(t['body']), disjoint=freeze(t['disjoint']))
        for t in json.loads((run/'evaluation/TARGET_POOL.json').read_text())]
    summary = json.loads((run/'evaluation/results/SUMMARY.json').read_text())
    if protocol['phase'] != 'development_smoke':
        corpus, corpus_check = poi_holdout(run/'evaluation/poi_tptp.tgz',
            json.loads((run/'evaluation/EXPOSED_TARGETS.json').read_text()))
        require(io.encoded(corpus) == io.encoded(targets), 'Corpus/exposure filtering differs')
        require(corpus_check == protocol['corpus'], 'Corpus audit metadata differs')
    require(len(targets) == protocol['targets'] == len({(t['body'],t['disjoint']) for t in targets}), 'Duplicate targets')
    catalogue = json.loads((run/'inputs/CATALOGUE.json').read_text())
    catalogue_lib = io.restored(io.read_gzip(run/'inputs/catalogue.json.gz'))
    by_id = {r['id']:r for r in catalogue['rules']}
    scores = {}; evaluated = 0; promotions = 0; selection_hashes = set()
    complete = True; informative = True
    identities = {(r['seed'],r['method'],r['policy']) for r in summary['rows']}
    require(len(summary['rows']) == len(identities) == len(protocol['seeds'])*len(METHODS)*len(protocol['policies']), 'Missing/duplicate evaluation rows')
    for seed in protocol['seeds']:
        block = run/'construction'/str(seed); source = io.restored(io.read_gzip(block/'source.json.gz'))
        source_row = json.loads((block/'SOURCE_RESULT.json').read_text())
        source_state = io.read_gzip(block/'source.json.gz')
        initial_checkpoint = io.read_gzip(run/'inputs/initial.json.gz')
        require(source_state['nodes'][:len(initial_checkpoint['nodes'])] == initial_checkpoint['nodes'], 'Inherited source prefix changed')
        accumulated = source_row['result']['accumulated']
        source_codec = json.loads((block/'SOURCE_COMPRESSION.json').read_text())
        expected_source = set(accumulated['active_final'])|set(accumulated['fragment_library'])|set(source_codec['dictionary'])
        require(expected_source == set(source_state['search_ids']), 'Unexpected source search interface')
        initial_union = set(); hits = {}
        for method in METHODS:
            directory = block/method
            profile = structural_profile(source, start=protocol['source_initial_nodes']) if method == 'adaptive' else {}
            selected, selection = select_rules(catalogue, profile, protocol['fixed_ids'], method,
                protocol['quota'], seed+50_000_000)
            stored = json.loads((directory/'SELECTION.json').read_text())
            require(selected == stored['selected_ids'] and selection['extra_ids'] == stored['extra_ids'], 'Selection not target-blind frozen rule')
            require(profile == stored['source_profile'], 'Source profile changed')
            if method == 'adaptive': selection_hashes.add(selection['selection_sha256'])
            initial = io.read_gzip(directory/'initial.json.gz')
            # Reconstruct the projected interface from explicit source/rule
            # roots, not the producer's declared search rights alone.
            merged = io.restored(io.read_gzip(block/'source.json.gz'))
            rule_roots = merge_rules(merged, catalogue_lib, [by_id[i]['node'] for i in selected])
            codec = json.loads((directory/'COMPRESSION.json').read_text())
            _, recoded = compress_above(fragment_records(merged),
                [n.id for n in merged.nodes if n.parents], set(source_state['search_ids'])|set(rule_roots))
            require({k:v for k,v in recoded.items() if k != 'cpu_seconds'} ==
                    {k:v for k,v in codec.items() if k != 'cpu_seconds'}, 'Unpaid or target-dependent upper compression')
            source_state = io.read_gzip(block/'source.json.gz')
            selected_roots = set(source_state['search_ids'])|set(rule_roots)|set(codec['dictionary'])
            projected, permitted, _ = merged.project(sorted(selected_roots))
            require(io.encoded(projected.json_nodes()) == io.encoded(initial['nodes']), 'Unexpected interface/support certificate')
            require(set(permitted)|{n.id for n in projected.nodes if not n.parents} == set(initial['search_ids']), 'Hidden support received initial search rights')
            build = json.loads((directory/'BUILD.json').read_text())
            source_row = json.loads((block/'SOURCE_RESULT.json').read_text())
            require(close(build['total_before_continuation_cpu'], build['method_cold_cpu']+
                source_row['source_common_cpu']+protocol['catalogue_cold_cpu']), 'Incomplete cold-cost arithmetic')
            for policy in protocol['policies']:
                case = directory/policy; final_state = io.read_gzip(case/'final.json.gz'); final = io.restored(final_state)
                events = io.read_gzip(case/'events.json.gz')
                witnesses = [read_node(r) for r in io.read_gzip(case/'witnesses.json.gz')]
                allowed, times = replay_permissions(initial, final, events, witnesses)
                require(allowed == set(final_state['search_ids']), 'Final search rights differ')
                # Reconstruct the actual novelty/subsumption admission condition.
                idx = PatternIndex()
                for i in initial['search_ids']: idx.add(final.nodes[i].formula, i)
                for e in events:
                    n = final.nodes[e['node']]
                    require(all(instance_witness(final.nodes[i].formula, final.nodes[i].disjoint,
                        n.formula, n.disjoint) is None for i in idx.candidates(n.formula)), 'Admission used an already available subsumed statement')
                    idx.add(n.formula, n.id)
                cover, exact = first_hits(targets, final, times)
                initial_union.update(j for j,t in cover.items() if t == 0.)
                dest = run/'evaluation/results'/(str(seed)+'_'+method+'_'+policy)
                stored_hits = json.loads((dest/'HITS.json').read_text())
                require({int(j):t for j,t in stored_hits['first_cover'].items()} == cover, 'First-cover observer error')
                require({int(j):t for j,t in stored_hits['first_exact'].items()} == exact, 'Exact-hit observer error')
                for j, mapping in stored_hits['mappings'].items():
                    j = int(j); mapping = {freeze(k):freeze(v) for k,v in mapping}
                    nid = stored_hits['covering'][str(j)]; n = final.nodes[nid]
                    require(close(times[nid], cover[j]), 'Wrong first-cover witness time')
                    require(replace(n.formula, mapping) == targets[j]['body'], 'Wrong target instance')
                    require(mapped_disjoint(n.disjoint, mapping) <= set(targets[j]['disjoint']), 'Lost target $d')
                proof = dest/'targets.mm'
                claimed = {int(x) for x in re.findall(r'^quality\.target\.(\d+) \$p ',proof.read_text(),re.M)}
                require(claimed == set(cover), 'Missing explicit target certificate')
                io.independent_verify(run/'inputs/metamath', proof,
                    out/(str(seed)+'_'+method+'_'+policy+'.verify.log'))
                blind = json.loads((case/'RESULT.json').read_text())
                require(blind['policy'] == policy and blind['seed'] == seed+60_000_000+
                    protocol['policies'].index(policy)*2_000_000, 'Wrong continuation seed or policy')
                require(close(blind['total_before_growth_cpu'], build['total_before_continuation_cpu']+blind['setup_cpu']), 'Missing continuation setup cost')
                require(all(0 <= e['cpu'] <= blind['growth_cpu'] for e in events), 'Out-of-run event timestamp')
                expected_complete = blind['growth_cpu'] >= max(protocol['future_cpu'],protocol['total_cpu'])
                require(blind['completed_horizon'] == expected_complete, 'False completed-horizon flag')
                complete &= expected_complete
                hits[method,policy] = cover, exact, blind
                evaluated += len(events); promotions += len(witnesses)
        residual = set(range(len(targets)))-initial_union
        informative &= bool(residual)
        stored_residual = json.loads((run/'evaluation/results'/(str(seed)+'.RESIDUAL.json')).read_text())
        require(residual == set(stored_residual['residual']), 'Common residual differs')
        for method in METHODS:
            for policy in protocol['policies']:
                cover, exact, blind = hits[method,policy]
                row = next(r for r in summary['rows'] if r['seed'] == seed and r['method'] == method and r['policy'] == policy)
                require(row['residual_size'] == len(residual), 'Residual denominator differs')
                for scope, horizon, offset in (('future',protocol['future_cpu'],0.),
                    ('total',protocol['total_cpu'],blind['total_before_growth_cpu'])):
                    ids = {j for j in residual if cover.get(j,float('inf'))+offset <= horizon}
                    exact_ids = {j for j in residual if exact.get(j,float('inf'))+offset <= horizon}
                    auc = sum(max(0.,horizon-cover.get(j,float('inf'))-offset) for j in residual)/max(1.,horizon*len(residual))
                    require(ids == set(row[scope]['residual_ids']) and len(ids) == row[scope]['new_residual'], 'Primary endpoint differs')
                    require(len(exact_ids) == row[scope]['exact_new_residual'] and close(auc,row[scope]['normalised_auc']), 'Secondary endpoint differs')
                    scores[seed,method,policy,scope] = len(ids)
    pvalues = {}
    for control in protocol['comparisons']:
        tests = []
        for scope in ('future','total'):
            differences = [sum(scores[s,'adaptive',p,scope]-scores[s,control,p,scope]
                for p in protocol['policies'])/len(protocol['policies']) for s in protocol['seeds']]
            result = paired_inference(differences)
            require(result == summary['comparisons'][control][scope], 'Paired inference differs')
            tests.append(result['p_one_sided'])
        pvalues[control] = max(tests)
    require(holm(pvalues) == summary['holm_adjusted_p'], 'Multiplicity adjustment differs')
    require(len(selection_hashes) == summary['adaptive_distinct_selections'], 'False adaptive diversity claim')
    eligible = complete and informative and len(selection_hashes) >= 2 and protocol['phase'] != 'development_smoke'
    statistical = all(holm(pvalues)[c] < .05 and all(summary['comparisons'][c][s]['mean'] >= 1.
        for s in ('future','total')) for c in protocol['comparisons'])
    conclusion = ('candidate_superiority_pending_independent_audit' if eligible and statistical
                  else 'no_confirmed_adaptive_superiority' if eligible else 'inconclusive_or_development_only')
    require(eligible == summary['eligible'] and complete == summary['all_horizons_complete'] and
        informative == summary['informative_residual'] and conclusion == summary['status'], 'Unsupported confirmation claim')
    final = dict(status='passed', producer_conclusion=summary['status'],
        confirmation_supported=(summary['status'] == 'candidate_superiority_pending_independent_audit'),
        audited_source_blocks=len(protocol['seeds']), audited_cells=len(summary['rows']),
        independently_replayed_admission_events=evaluated, promotions=promotions,
        audited_targets=len(targets), seal_sha256=a.seal,
        residuals_hits_certificates_permissions_and_statistics_recomputed=True,
        limitation='Same declared finite corpus and sign-exchangeability assumptions; not universal optimality')
    io.write_json(out/'INDEPENDENT_REVIEW.json', final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--seal',required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
