"""Frozen-skeleton quality requires strict search/verification separation."""
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'hpc/article1_skeleton')]
import metamath_native_skeleton as io
from metamath_native.basis import load_basis
from metamath_native.proofs import Library
from metamath_native.terms import variables, variable, syntax, canonicalise
from native_quality_evaluation import (FrozenContinuation, normalised_witness,
    CoverageMonitor, target_proof_text, continuation_quality)

SOURCE = io.DEFAULT_SOURCE


@pytest.fixture(scope='module')
def basis():
    return load_basis(SOURCE)[0]


def initial(basis):
    lib = Library(basis); lib.initialise(); return lib


def test_new_witness_matches_native_certificate_exactly(basis):
    lib = initial(basis)
    q = lib.propose_mp(0, 0)
    witness = normalised_witness(lib, q)
    i, fresh = lib.accept(q)
    assert fresh and witness == lib.nodes[i]


def test_hidden_support_is_neither_premise_nor_duplicate_oracle(basis):
    lib = initial(basis)
    hidden, _ = lib.accept(lib.propose_mp(0, 0))
    e = FrozenContinuation(lib, range(29), 62)
    assert hidden not in e.available_ids()
    assert hidden not in e.index.candidates(lib.nodes[hidden].formula)
    assert lib.nodes[hidden].key not in e.search_known
    event = e.offer(lib.propose_mp(0, 0))
    assert event['node'] == hidden and event['promotion'] == 0
    assert e.counts['support_rediscovered'] == 1
    assert hidden in e.available_ids() and len(e.witnesses) == 1
    for n in e.export_library().nodes:
        e.export_library().check(n)


def test_inaccessible_stored_parent_cannot_enter_a_proof(basis):
    lib = initial(basis)
    hidden, _ = lib.accept(lib.propose_mp(0, 0))
    e = FrozenContinuation(lib, range(29), 62)
    with pytest.raises(AssertionError, match='Hidden certificate'):
        e.offer(lib.propose_gen(hidden))


def test_target_observer_never_changes_rng_search_or_certificates(basis):
    a, b = initial(basis), initial(basis)
    x = FrozenContinuation(a, range(29), 94)
    y = FrozenContinuation(b, range(29), 94)
    targets = [dict(body=a.nodes[0].formula, disjoint=a.nodes[0].disjoint)]
    monitor = CoverageMonitor(targets)
    for _ in range(300):
        left, right = x.step('mixed'), y.step('mixed')
        if right:
            monitor.observe(b, right['node'], 1.)
        assert left == right
    assert a.json_nodes() == b.json_nodes() and x.counts == y.counts


def test_continuation_depth_and_anchor_credit_do_not_use_hidden_ancestry(basis):
    lib = initial(basis)
    i, _ = lib.accept(lib.propose_mp(0, 0))
    e = FrozenContinuation(lib, [*range(29), i], 92)
    ev = e.offer(lib.propose_gen(i)); ev['cpu'] = .5
    assert ev['depth'] == 1 and ev['anchor_mask'].bit_count() == 1
    assert e.depths[i] == 0  # A frozen anchor starts a new proof horizon.
    result = continuation_quality(e, .25, set())
    assert result['available_admissions'] == 0


def test_censored_coverage_and_auc_are_not_success_imputation(basis):
    lib = initial(basis)
    t = [dict(body=lib.nodes[0].formula, disjoint=lib.nodes[0].disjoint),
         dict(body=syntax('wn', variable('wff', 'x')), disjoint=())]
    m = CoverageMonitor(t); m.observe(lib, 0, 0.)
    r = m.readout(2., [])
    assert r['covered'] == 1 and r['restricted_mean_first_cover_cpu'] == 1.
    assert r['normalised_coverage_auc'] == .5


def test_official_verifier_checks_promotions_and_target_instances(basis, tmp_path):
    lib = initial(basis)
    hidden, _ = lib.accept(lib.propose_mp(0, 0))
    e = FrozenContinuation(lib, range(29), 99)
    e.offer(lib.propose_mp(0, 0))
    monitor = CoverageMonitor([dict(body=lib.nodes[0].formula, disjoint=lib.nodes[0].disjoint),
        dict(body=lib.nodes[hidden].formula, disjoint=lib.nodes[hidden].disjoint)])
    for i in e.available_ids():
        monitor.observe(lib, i, 0.)
    path = tmp_path/'quality.mm'
    path.write_text(target_proof_text(e.export_library(), monitor))
    check = io.independent_verify(io.DEFAULT_VERIFIER, path, tmp_path/'verify.log')
    assert check['status'] == 'verified'
    assert 'quality.target.' in path.read_text()


def test_one_way_instance_need_not_match_exact_statement(basis, tmp_path):
    lib = initial(basis)
    p, q = variable('wff','p'), variable('wff','q')
    body = canonicalise(syntax('wi',p,syntax('wi',syntax('wn',q),p)))[0]
    m = CoverageMonitor([dict(body=body, disjoint=())])
    ax1 = next(n for n in lib.nodes if n.assertion == 'ax-1')
    m.observe(lib,ax1.id,0.)
    assert not m.first_exact and m.first_cover == {0:0.}
    path = tmp_path/'instance.mm'
    path.write_text(target_proof_text(lib,m))
    assert io.independent_verify(io.DEFAULT_VERIFIER,path,tmp_path/'verify.log')['status']=='verified'


def test_observer_cannot_discard_disjoint_restrictions(basis, tmp_path):
    lib = initial(basis)
    ax5 = next(n for n in lib.nodes if n.assertion == 'ax-5')
    assert ax5.disjoint
    m = CoverageMonitor([dict(body=ax5.formula,disjoint=()),
                         dict(body=ax5.formula,disjoint=ax5.disjoint)])
    m.observe(lib,ax5.id,0.)
    assert m.first_cover == {1:0.} and m.first_exact == {1:0.}
    path = tmp_path/'disjoint.mm'
    path.write_text(target_proof_text(lib,m))
    assert io.independent_verify(io.DEFAULT_VERIFIER,path,tmp_path/'verify.log')['status']=='verified'
