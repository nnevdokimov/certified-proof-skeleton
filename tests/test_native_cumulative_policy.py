"""Adapters reuse the accumulated policies without weakening typed proofs."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'hpc/article1_skeleton')]
from metamath_native.basis import load_basis
from metamath_native.proofs import Library
from metamath_native.terms import variable, syntax, sort
from metamath_native.online_skeleton import OnlineConfig
from native_cumulative_policy import CumulativeGrowth, propositional_term, fragment_records
from fragment_policy import costs, ProofCoverage

SOURCE = ROOT / 'output/article1_hpc_runs/skeleton-utility-final-20261006-v1/bundle/inputs/set.mm'
MODEL = ROOT / 'output/article1_skeleton_gate_pilot_20261016_v1/training/prop_200.model.json'


def make(attempts=200, **options):
    lib = Library(load_basis(SOURCE)[0]); lib.initialise()
    bounds = dict(active_cap=16, queue_cap=64, rot_patience=2)
    bounds.update(options)
    engine = CumulativeGrowth(lib, OnlineConfig(attempts=attempts, pair_pool=8), 940001,
                              model=json.loads(MODEL.read_text()), **bounds)
    return lib, engine


def test_propositional_adapter_never_erases_quantifiers_or_types():
    p = variable('wff', 'p'); x = variable('setvar', 'x')
    assert propositional_term(syntax('wi', p, syntax('wn', p))) == ('i', ('v', 'p'), ('n', ('v', 'p')))
    assert propositional_term(syntax('wal', x, p)) is None
    assert propositional_term(x) is None


def test_shared_mechanisms_are_executed_and_certificates_stay_closed():
    lib, engine = make(400)
    result, _ = engine.run(policy='uniform')
    c = result['accumulated']
    assert c['lifecycle']['frozen_prop_model_scores'] > 0
    assert c['lifecycle']['native_fallback_scores'] > 0
    assert c['lifecycle']['replay_attempts'] > 0
    assert c['lifecycle']['rot_retired'] + c['lifecycle']['active_cap_retired'] > 0
    assert c['fragment_reports']
    assert result['posthoc_core_extraction'] is False
    assert all(len(set(s['live']) - engine.roots) <= 16 for s in c['snapshots'])
    assert all(s['pending'] <= 64 for s in c['snapshots'])
    for n in lib.nodes:
        lib.check(n)
    for s in c['snapshots']:
        assert all(i < s['nodes'] for i in s['live'])


def test_inactive_certificates_cannot_silently_supply_subsumption():
    lib, engine = make()
    derived, _ = lib.accept(lib.propose_mp(0, 0)); engine._register(lib.nodes[derived])
    assert derived not in engine.live
    assert derived not in engine.index.candidates(lib.nodes[derived].formula)


def test_native_fragment_adapter_preserves_roles_and_definition_cost():
    lib, engine = make()
    engine.run(policy='uniform')
    records = fragment_records(lib)
    root = len(lib.nodes) - 1
    for n in lib.nodes:
        assert records[n.id][3] == tuple(p for p, _ in n.parents)
    assert costs(records, [root])['net_saved_symbols'] == 0


def test_transfer_retains_early_evidence_after_many_unrelated_proofs():
    # The two old uses of node 2 are followed by a thousand unrelated proofs.
    parents = [(), (0, 0), (1, 0), (2, 2), (2, 0)] + [(0, 0)] * 1000
    records = {i: (i, None, None, p) for i, p in enumerate(parents)}
    coverage = ProofCoverage()
    coverage.update(dict(list(records.items())[:4]))
    assert coverage.select(4)[0] == []
    coverage.update(records)
    coverage.update(records)
    chosen, report = coverage.select(4)
    assert chosen == [2]
    assert report['training_proofs'] == len(parents) - 1
    assert report['covered_proofs'] == 2


def test_transfer_counts_distinct_proofs_and_excludes_own_root():
    # Node 2 is used twice in one proof; node 4 occurs in two distinct proofs.
    parents = [(), (0, 0), (1, 0), (2, 2), (0, 1), (4, 0), (0, 4)]
    records = {i: (i, None, None, p) for i, p in enumerate(parents)}
    coverage = ProofCoverage()
    coverage.update(records)
    chosen, report = coverage.select(4)
    assert chosen == [4] and report['covered_proofs'] == 2
    assert coverage.select(0)[0] == []


def test_bank_stops_at_coverage_saturation_and_respects_an_explicit_cap():
    parents = [()]
    anchors = []
    for _ in range(7):
        start = len(parents)
        parents.extend([(0, 0), (start, 0), (start + 1, 0), (0, start + 1)])
        anchors.append(start + 1)
    cov = ProofCoverage()
    cov.update({i: (i, None, None, p) for i, p in enumerate(parents)})
    assert cov.select()[0] == anchors
    assert cov.select(4)[0] == anchors[:4]
    assert cov.select(100)[0] == anchors


def test_transfer_bank_reaches_the_live_engine_with_checked_proofs():
    lib, engine = make(400)
    result, _ = engine.run(policy='uniform')
    state = result['accumulated']
    assert state['fragment_selection'] == 'transfer'
    assert state['fragment_bank_limit'] is None
    assert sorted(r['root'] for r in state['transfer_report']['trace']) == state['transfer_bank']
    assert state['transfer_report']['training_proofs'] == sum(bool(n.parents) for n in lib.nodes)
    assert any(r['chosen'] for r in state['fragment_reports'])
    for i in state['transfer_bank']:
        lib.check(lib.nodes[i])
    for decision in state['fragment_reports']:
        assert sum(r['additional_proofs'] for r in decision['report']['trace']) == decision['report']['covered_proofs']
        for i in decision['chosen']:
            lib.check(lib.nodes[i])


def test_compact_frontier_preserves_direct_evidence_after_root_coverage():
    v = ('v', 'wff', 'p')
    short = syntax('wn', v)
    long = syntax('wi', short, short)
    parents = [(), (0, 0), (1, 0), (2, 0), (2, 3), (2, 0), (2, 0)]
    records = {i: (i, short if i == 3 else long, None, p) for i, p in enumerate(parents)}
    cov = ProofCoverage()
    cov.update(records)
    chosen, report = cov.select()
    assert chosen == [2, 3]
    assert report['trace'][1] == dict(root=3, additional_proofs=0, frontier_consumers=1, marginal_votes=1)
    assert cov.select(1)[0] == [2]
    # Hide the only direct consumer of the compact claim. Proof ordinals omit the axiom.
    assert cov.select(visible_proofs={1, 2, 4, 5})[0] == [2]
    assert cov.select(visible_proofs=set())[0] == []


def test_compact_frontier_keeps_ties_and_deduplicates_repeated_parent():
    v = ('v', 'wff', 'p')
    parents = [(), (0, 0), (1, 0), (1, 0), (2, 2), (3, 3)]
    records = {i: (i, v, None, p) for i, p in enumerate(parents)}
    policy = ProofCoverage()
    policy.update(records)
    chosen, report = policy.select()
    assert chosen == [2, 3]
    assert [r['frontier_consumers'] for r in report['trace']] == [1, 1]


def test_frozen_bank_matches_used_instances_and_respects_source_side_conditions():
    x = variable('wff', 'x')
    pattern = syntax('wi', x, syntax('wn', x))
    t = syntax('wn', x)
    used = syntax('wi', t, syntax('wn', t))
    other = syntax('wi', t, syntax('wn', x))
    # The first schema has no edge at all; both premises of node 5 are node 4.
    parents = [(), (0, 0), (1, 0), (1, 0), (1, 0), (4, 4), (3, 0)]
    formulas = [x, t, pattern, other, used, t, t]
    records = {i: (i, formulas[i], None, p) for i, p in enumerate(parents)}
    cov = ProofCoverage()
    chosen, report = cov.freeze(records, sort_of=sort)
    assert 2 in chosen
    row = next(r for r in report['trace'] if r['root'] == 2)
    assert row['frontier_consumers'] == 1 and row['additional_proofs'] == 0
    assert 2 not in cov.freeze(records, sort_of=sort, restricted={2})[0]
    assert 2 not in cov.freeze(records, sort_of=sort, visible_proofs={1, 2, 3, 5})[0]


def test_frozen_claim_cannot_match_another_variable_sort():
    p = variable('wff', 'p')
    x = variable('setvar', 'x')
    parents = [(), (0, 0), (1, 0), (1, 0), (3, 3)]
    records = {i: (i, term, None, ps) for i, (term, ps) in
               enumerate(zip([p, p, x, syntax('wn', p), p], parents))}
    assert 2 not in ProofCoverage().freeze(records, sort_of=sort)[0]


def test_bank_limit_is_independent_of_working_limits():
    _, limited = make(active_cap=2, fragment_slots=7)
    assert limited.bank_limit == 7
    _, unlimited = make(active_cap=None, queue_cap=None)
    result, _ = unlimited.run(policy='uniform')
    state = result['accumulated']
    assert state['fragment_bank_limit'] is None and state['active_cap'] is None
    assert state['deferred']['cap'] is None
    assert not state['lifecycle'].get('active_cap_retired')
