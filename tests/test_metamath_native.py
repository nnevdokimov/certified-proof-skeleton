"""Check types, native frames, $d, recovery, and independent Metamath export."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from metamath_native.basis import load_basis,basis_text,LABELS
from metamath_native.terms import (parse,render,variable,syntax,unify,mapped_disjoint,variables,validate)
from metamath_native.proofs import Library,proof_text

SOURCE=ROOT/'output/article1_hpc_runs/skeleton-utility-final-20261006-v1/bundle/inputs/set.mm'
VERIFIER=ROOT/'output/metamath_native_skeleton_20261020_v1/metamath'


@pytest.fixture(scope='module')
def basis():return load_basis(SOURCE)[0]


@pytest.fixture
def library(basis):
    lib=Library(basis);lib.initialise();return lib


def test_exact_explicit_basis_and_no_human_theorems(basis):
    assert {a.label for a in basis}==LABELS
    assert len(basis)==43 and sum(a.kind=='logical' for a in basis)==31
    text,_=basis_text(basis)
    assert '$p' not in text.replace('no imported $p statements.','')
    assert 'ax-groth' not in text and 'ax-inf2' not in text and 'ax-ac2' not in text


def test_round_trip_all_logical_axioms(basis):
    for a in basis:
        if a.kind=='logical':assert render(parse(a.expression[1:],a.types))==a.expression[1:]


def test_type_mismatch_fails():
    with pytest.raises(ValueError):unify(variable('setvar','x'),variable('wff','ph'))


def test_setvar_does_not_unify_with_class_coercion():
    x=variable('setvar','x')
    with pytest.raises(ValueError):unify(x,syntax('cv',x))


def test_occurs_check():
    p=variable('wff','p')
    with pytest.raises(ValueError):unify(p,syntax('wn',p))


def test_metavariable_disjointness_rejects_capture():
    x=variable('setvar','x');ph=variable('wff','ph')
    body=syntax('wceq',syntax('cv',x),syntax('cv',x))
    with pytest.raises(ValueError):mapped_disjoint([(x,ph)],{x:x,ph:body})


def test_disjointness_tracks_variables_inside_quantifiers():
    x=variable('setvar','x');y=variable('setvar','y');p=variable('wff','p')
    mapped=mapped_disjoint([(x,p)],{x:x,p:syntax('wal',y,p)})
    assert tuple(sorted((x,y))) in mapped and tuple(sorted((x,p))) in mapped


def test_seed_count_and_native_dv(library):
    assert len(library.nodes)==29
    sep=next(n for n in library.nodes if n.assertion=='ax-sep')
    assert len(sep.disjoint)==5


def test_mp_and_generalisation(library):
    i,added=library.accept(library.propose_mp(0,0));assert added
    j,added=library.accept(library.propose_gen(i));assert added
    assert library.nodes[j].height==2 and len(library.nodes[j].parents)==1


def test_identical_proofs_do_not_duplicate_nodes(library):
    p=library.propose_mp(0,0)
    assert library.accept(p)[1]
    assert not library.accept(p)[1]


def test_restore_and_continue(library,basis):
    library.accept(library.propose_mp(0,0))
    restored=Library.restore(basis,json.loads(json.dumps(library.json_nodes())))
    assert restored.json_nodes()==library.json_nodes()
    assert restored.accept(restored.propose_gen(29))==library.accept(library.propose_gen(29))


def test_tampered_conclusion_fails(library,basis):
    rows=json.loads(json.dumps(library.json_nodes()))
    rows[-1]['formula']=['v','wff','np0']
    with pytest.raises(ValueError):Library.restore(basis,rows)


def test_tampered_disjoint_frame_fails(library,basis):
    rows=json.loads(json.dumps(library.json_nodes()))
    row=next(r for r in rows if r['assertion']=='ax-sep');row['disjoint']=[]
    with pytest.raises(ValueError):Library.restore(basis,rows)


def test_non_earlier_parent_fails(library,basis):
    library.accept(library.propose_mp(0,0));rows=json.loads(json.dumps(library.json_nodes()))
    rows[-1]['parents'][0][0]=len(rows)-1
    with pytest.raises(ValueError):Library.restore(basis,rows)


def test_generated_native_file_passes_independent_verifier(library,tmp_path):
    if not VERIFIER.exists():pytest.skip('Official Metamath verifier not built')
    library.accept(library.propose_mp(0,0));library.accept(library.propose_gen(0))
    library.accept(library.propose_gen(next(n.id for n in library.nodes if n.assertion=='ax-sep')))
    text,ids=proof_text(library);path=tmp_path/'library.mm';path.write_text(text)
    result=subprocess.run([str(VERIFIER),'read library.mm','verify proof *','exit'],cwd=tmp_path,
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert '?Error' not in result.stdout,result.stdout
    assert 'verified' in result.stdout.lower(),result.stdout


def test_selected_export_keeps_transitive_support(library):
    i,_=library.accept(library.propose_mp(0,0));j,_=library.accept(library.propose_gen(i))
    text,ids=proof_text(library,[j])
    assert ids==[0,i,j]
    assert library.nodes[j].label+' $p' in text


def test_recursive_types_do_not_trust_constructor_tag():
    with pytest.raises(ValueError):validate(('wi',variable('setvar','x'),variable('wff','p')))


def test_wrong_arity_fails():
    with pytest.raises(ValueError):validate(('wn',))


def test_variable_cannot_inject_metamath_tokens():
    with pytest.raises(ValueError):validate(('v','wff','p $. bad'))


def test_balanced_family_selection_and_restart(library,basis):
    import time
    from metamath_native.growth import config,epoch,select_core
    cfg=config(max_nodes=256,core_slots=64)
    epoch(library,cfg,1,time.process_time()+30)
    restored=Library.restore(basis,json.loads(json.dumps(library.json_nodes())))
    first=epoch(library,cfg,2,time.process_time()+30)
    second=epoch(restored,cfg,2,time.process_time()+30)
    assert first==second and library.json_nodes()==restored.json_nodes()
    core=select_core(library,16)
    assert len(core)==len(set(core))==16
    assert all(library.nodes[i].parents for i in core)


def test_recover_only_receipted_checkpoint(library,tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_cli',ROOT/'scripts/metamath_native_skeleton.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    (tmp_path/'states').mkdir();state=dict(epoch=8,nodes=library.json_nodes())
    path=tmp_path/'states/000008.json.gz';cli.write_gzip(path,state)
    cli.write_json(tmp_path/'states/000008.json.receipt.json',dict(epoch=8,nodes=len(library.nodes),sha256=cli.sha(path)))
    # A later partial file has no receipt and cannot displace the committed state.
    (tmp_path/'states/000009.json.gz').write_bytes(b'incomplete')
    recovered,chosen=cli.load_committed(tmp_path)
    assert recovered==json.loads(json.dumps(state)) and chosen==path


def test_full_setmm_export_verifies_without_human_lemma_dependencies(library,tmp_path):
    if not VERIFIER.exists():pytest.skip('Official verifier not built')
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_cli_full',ROOT/'scripts/metamath_native_skeleton.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    library.accept(library.propose_mp(0,0))
    text,_=proof_text(library,original_include='set.mm')
    # Link to the local pinned source, never a remote or mutable download.
    (tmp_path/'set.mm').symlink_to(SOURCE)
    path=tmp_path/'new.mm';path.write_text(text)
    result=cli.independent_verify(VERIFIER,path,tmp_path/'verify.log','native.*')
    assert result['status']=='verified'


def test_independent_verifier_rejects_false_claim_even_when_read_succeeds(library,tmp_path):
    if not VERIFIER.exists():pytest.skip('Official verifier not built')
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_cli_negative',ROOT/'scripts/metamath_native_skeleton.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    text,_=proof_text(library)
    text=text.replace('native.0 $p |- ( np0 -> ( np1 -> np0 ) )','native.0 $p |- np0')
    path=tmp_path/'bad.mm';path.write_text(text)
    with pytest.raises(ValueError):cli.independent_verify(VERIFIER,path,tmp_path/'bad.log')


def test_projection_retains_exact_proofs_not_unrelated_archive(library):
    import time
    from metamath_native.growth import config,epoch
    epoch(library,config(max_nodes=256,core_slots=32),1,time.process_time()+30)
    root=len(library.nodes)-1
    projected,roots,mapping=library.project([root])
    assert len(projected.nodes)<len(library.nodes)
    assert projected.nodes[roots[0]].formula==library.nodes[root].formula
    assert projected.nodes[roots[0]].disjoint==library.nodes[root].disjoint
    assert set(mapping)<=set(range(len(library.nodes)))


def test_core_only_search_never_activates_passive_proof_support(library):
    import time
    from metamath_native.growth import config,epoch,select_core
    cfg=config(max_nodes=256,core_slots=16)
    for i in range(1,4):epoch(library,cfg,i,time.process_time()+30)
    projected,core,mapping=library.project(select_core(library,16))
    allowed=set(core)|{n.id for n in projected.nodes if not n.parents}
    passive=set(range(len(projected.nodes)))-allowed
    row=epoch(projected,cfg,4,time.process_time()+30,allowed)
    active=set(row['active_minors'])|set(row['active_majors'])
    assert not active&passive
    assert all({p for p,_ in projected.nodes[i].parents}<=allowed for i in row['admitted_ids'])


def test_corrupted_receipt_is_not_a_checkpoint(library,tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_cli_corrupt',ROOT/'scripts/metamath_native_skeleton.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    (tmp_path/'states').mkdir();path=tmp_path/'states/000008.json.gz'
    cli.write_gzip(path,dict(epoch=8,nodes=library.json_nodes()))
    cli.write_json(tmp_path/'states/000008.json.receipt.json',dict(epoch=8,nodes=len(library.nodes),sha256='0'*64))
    with pytest.raises(ValueError):cli.load_committed(tmp_path)


@pytest.mark.parametrize('updates',[{'max_nodes':10,'core_slots':4},{'cpu_seconds':float('nan')},
    {'minors':1.5},{'targets':['x = x']},{'unrecognised':1}])
def test_resource_and_target_contract(updates):
    from metamath_native.growth import config
    with pytest.raises(ValueError):config(**updates)
