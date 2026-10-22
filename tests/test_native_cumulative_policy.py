"""Adapters reuse the accumulated policies without weakening typed proofs."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'hpc/article1_skeleton')]
from metamath_native.basis import load_basis
from metamath_native.proofs import Library
from metamath_native.terms import variable, syntax
from metamath_native.online_skeleton import OnlineConfig
from native_cumulative_policy import CumulativeGrowth, propositional_term, fragment_records
from fragment_policy import costs

SOURCE = ROOT / 'output/article1_hpc_runs/skeleton-utility-final-20261006-v1/bundle/inputs/set.mm'
MODEL = ROOT / 'output/article1_skeleton_gate_pilot_20261016_v1/training/prop_200.model.json'


def make(attempts=200):
    lib = Library(load_basis(SOURCE)[0]); lib.initialise()
    engine = CumulativeGrowth(lib, OnlineConfig(attempts=attempts, pair_pool=8), 940001,
                              model=json.loads(MODEL.read_text()), active_cap=16,
                              queue_cap=64, rot_patience=2)
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
