from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'hpc/article1_skeleton')]
import metamath_native_skeleton as io
from metamath_native.basis import load_basis
from metamath_native.proofs import Library,proof_text
from metamath_native.saturation import instance_witness
from metamath_native.terms import variable,syntax,canonicalise
from native_rule_synthesis import (ProofFactory,abstract_programme,boundary_partitions,
    observed_programmes,synthesise_rules,ClosedCompiler)


@pytest.fixture(scope='module')
def basis():return load_basis(io.DEFAULT_SOURCE)[0]


def initial(basis):
    lib=Library(basis);lib.initialise();return lib


def test_partitions_do_not_duplicate_boundary_equalities():
    assert list(boundary_partitions(1))==[(0,)]
    assert len(set(boundary_partitions(3)))==5
    assert len(set(boundary_partitions(4)))==15


def test_joint_unification_detects_occurs_cycle():
    with pytest.raises(ValueError):abstract_programme(('H','H'),(0,0),ProofFactory())


def test_unrelated_closed_axioms_cannot_count_as_open_hypotheses(basis):
    lib=initial(basis);f=ProofFactory();p=variable('wff','p')
    with pytest.raises(ValueError,match='undischarged'):
        ClosedCompiler(lib).compile(f.hypothesis(0,p))


def test_deduction_synthesises_composition_without_a_target_search(basis,tmp_path):
    f=ProofFactory();proof,holes=abstract_programme((('H','H'),'H'),(0,1,2),f)
    # Discharge the carried input first, leaving two implications as the rule
    # interface. The expected expression checks the implementation, not search.
    for j in (0,2,1):proof=f.discharge(proof,j,holes[j])
    p,q,r=(variable('wff',s) for s in ('p','q','r'))
    expected=syntax('wi',syntax('wi',p,q),syntax('wi',syntax('wi',q,r),syntax('wi',p,r)))
    assert canonicalise(proof.formula)[0]==canonicalise(expected)[0]
    lib=initial(basis);nid=ClosedCompiler(lib).compile(proof)
    assert instance_witness(lib.nodes[nid].formula,lib.nodes[nid].disjoint,expected,()) is not None
    path=tmp_path/'composition.mm';path.write_text(proof_text(lib)[0])
    assert io.independent_verify(io.DEFAULT_VERIFIER,path,tmp_path/'verify.log')['status']=='verified'


def test_shared_boundary_synthesises_repeated_use_not_independent_copies(basis):
    f=ProofFactory();proof,holes=abstract_programme(('H',('H','H')),(0,0,1),f)
    for j in (0,1):proof=f.discharge(proof,j,holes[j])
    p,q=variable('wff','p'),variable('wff','q')
    expected=syntax('wi',syntax('wi',p,syntax('wi',p,q)),syntax('wi',p,q))
    assert canonicalise(proof.formula)[0]==canonicalise(expected)[0]
    lib=initial(basis);nid=ClosedCompiler(lib).compile(proof)
    lib.check(lib.nodes[nid])


def test_generalisation_cuts_the_open_fragment(basis):
    lib=initial(basis);i,_=lib.accept(lib.propose_gen(0))
    motifs=observed_programmes(lib,2)
    assert motifs=={'H':len(lib.nodes)}
    assert lib.nodes[i].assertion=='ax-gen'


def test_all_observed_programmes_compile_and_pass_official_verifier(basis,tmp_path):
    lib=initial(basis)
    # Obtain observed MP motifs from the same native rule, not human lemmas.
    a,_=lib.accept(lib.propose_mp(0,0))
    lib.accept(lib.propose_mp(a,0))
    outputs,report=synthesise_rules(lib,maximum_mp=2)
    assert outputs and report['new_rule_schema_count']>0
    assert all(not lib.nodes[i].disjoint for i in outputs)
    path=tmp_path/'synthesised.mm';path.write_text(proof_text(lib)[0])
    assert io.independent_verify(io.DEFAULT_VERIFIER,path,tmp_path/'verify.log')['status']=='verified'


def test_final_schemas_do_not_depend_on_human_reference_or_goal_labels(basis):
    lib=initial(basis)
    lib.accept(lib.propose_mp(0,0))
    outputs,report=synthesise_rules(lib,maximum_mp=1)
    assert not any('label' in r or 'target_label' in r for r in report['records'])
    assert all(r['status'] in ('verified_native','bounded_out') for r in report['records'])


def test_exhausted_short_programme_class_does_not_spawn_endless_padding(basis):
    lib=initial(basis)
    a,_=lib.accept(lib.propose_mp(0,0))
    lib.accept(lib.propose_mp(a,0))
    # The other binary tree shape occurs in a proof whose major has a parent.
    for i in range(len(lib.nodes)):
        try:
            lib.accept(lib.propose_mp(0,i))
        except ValueError:
            pass
    # A first synthesis can itself expose a previously unobserved tree shape.
    # Once the finite motif class stops changing, it must not create padding.
    stable=False
    for _ in range(4):
        motifs=set(observed_programmes(lib,2))
        synthesise_rules(lib,maximum_mp=2)
        if set(observed_programmes(lib,2))==motifs:
            before=len(lib.nodes)
            synthesise_rules(lib,maximum_mp=2)
            assert len(lib.nodes)==before
            stable=True;break
    assert stable
