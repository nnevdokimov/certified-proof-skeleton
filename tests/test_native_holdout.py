import json
from pathlib import Path
import sys
import tarfile
import io as bytes_io
from dataclasses import replace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'hpc/article1_skeleton')]
import metamath_native_skeleton as io
from metamath_native.basis import load_basis
from metamath_native.proofs import Library, proof_text
from metamath_native.terms import variable, syntax, canonicalise
from native_holdout import (grammar, profile_key, structural_profile, make_catalogue,
    select_rules, merge_rules, parse_poi, unfold_prop, poi_holdout, paired_inference, holm)
from run_native_holdout import verify_seal, complete_receipts, endpoint


def conjecture(term='i(A,A)', names='A'):
    return f'fof(f5,conjecture,( ! [{names}] : is_theorem({term}) )).'


def test_binary_grammar_complete_without_duplicates():
    assert [len(grammar(n)) for n in range(4)] == [1,1,2,5]
    assert len(set(grammar(3))) == 5


def test_poi_parser_alpha_canonicalises_and_keeps_negation():
    p = variable('wff','p')
    assert parse_poi(conjecture('i(A,n(n(A)))')) == canonicalise(syntax('wi',p,syntax('wn',syntax('wn',p))))[0]
    assert parse_poi(conjecture('i(B,B)','B')) == parse_poi(conjecture())


@pytest.mark.parametrize('text',[
    conjecture('evil(A)'), conjecture('i(A,X)'), conjecture('i(A,A);halt'),
    conjecture()+conjecture(), "include('private.p').\n"+conjecture(),
    conjecture('i(A,A)','A,A'), conjecture('i(A,A)','')])
def test_poi_parser_fails_closed(text):
    with pytest.raises(ValueError): parse_poi(text)


def test_definition_expansion_prevents_false_holdout():
    p,q = variable('wff','p'), variable('wff','q')
    assert unfold_prop(syntax('wa',p,q)) == syntax('wn',syntax('wi',p,syntax('wn',q)))
    assert unfold_prop(syntax('wo',p,q)) == syntax('wi',syntax('wn',p),q)
    assert unfold_prop(syntax('wb',p,q)) == syntax('wn',syntax('wi',syntax('wi',p,q),syntax('wn',syntax('wi',q,p))))


def test_holdout_removes_exposed_and_deduplicates_without_size_tuning(tmp_path):
    archive=tmp_path/'poi.tgz'
    with tarfile.open(archive,'w:gz') as t:
        for name,text in [('a',conjecture()),('b',conjecture('i(A,i(B,A))','A,B')),
                          ('c',conjecture('i(X,i(Y,X))','X,Y'))]:
            data=text.encode(); member=tarfile.TarInfo('poi_tptp/poi_000001_'+name+'.p')
            member.size=len(data);t.addfile(member,bytes_io.BytesIO(data))
    rows,report=poi_holdout(archive,[dict(body=parse_poi(conjecture()))])
    assert len(rows)==1 and len(rows[0]['labels'])==2
    assert report['excluded']['previous_exact_after_definitions']==1


def test_unsafe_archive_member_never_extracts(tmp_path):
    archive=tmp_path/'evil.tgz'
    with tarfile.open(archive,'w:gz') as t:
        data=conjecture().encode();m=tarfile.TarInfo('../escape.p');m.size=len(data)
        t.addfile(m,bytes_io.BytesIO(data))
    with pytest.raises(ValueError,match='unsafe'):poi_holdout(archive,[])
    assert not (tmp_path.parent/'escape.p').exists()


def test_poi_named_constants_are_excluded_not_reinterpreted(tmp_path):
    archive=tmp_path/'poi.tgz'
    with tarfile.open(archive,'w:gz') as t:
        for name,text in [('a',conjecture()),('b',conjecture('i(truth,A)'))]:
            data=text.encode();m=tarfile.TarInfo('poi_tptp/poi_000001_'+name+'.p');m.size=len(data)
            t.addfile(m,bytes_io.BytesIO(data))
    rows,report=poi_holdout(archive,[])
    assert len(rows)==1 and report['excluded']['named_constants_outside_frozen_language']==1


def catalogue_fixture():
    return dict(short_ids=['short'], rules=[dict(id='short',minimum_mp=1,formula_size=5,features=[]),
        dict(id='left',minimum_mp=3,formula_size=10,features=[profile_key((('H','H'),'H'),(0,1,2))]),
        dict(id='right',minimum_mp=3,formula_size=10,features=[profile_key(('H',('H','H')),(0,1,2))])])


def test_real_adaptation_changes_with_source_not_human_goal():
    c=catalogue_fixture();left={c['rules'][1]['features'][0]:20};right={c['rules'][2]['features'][0]:20}
    l,_=select_rules(c,left,['left'],'adaptive',1,123)
    r,_=select_rules(c,right,['left'],'adaptive',1,123)
    assert l==['left','short'] and r==['right','short']
    assert select_rules(c,left,['left'],'fixed',1,123)[0]==select_rules(c,right,['left'],'fixed',1,123)[0]


def test_quota_controls_count_distinct_delivered_rules_and_keep_short_set():
    c=catalogue_fixture()
    for method in ('adaptive','fixed','random'):
        ids,row=select_rules(c,{},['right'],method,1,123)
        assert len(ids)==2 and 'short' in ids and len(row['extra_ids'])==1
    assert len(select_rules(c,{},['right'],'fixed_all',1,123)[0])==3
    with pytest.raises(ValueError):select_rules(c,{},['right'],'adaptive',3,123)


def test_source_profile_stops_at_gen_and_respects_new_prefix():
    lib=Library(load_basis(io.DEFAULT_SOURCE)[0]);lib.initialise();initial=len(lib.nodes)
    mp,_=lib.accept(lib.propose_mp(0,0));lib.accept(lib.propose_gen(mp))
    p=structural_profile(lib,start=initial)
    assert p and structural_profile(lib,start=len(lib.nodes))=={}
    assert all(json.loads(key)[0] != 'ax-gen' for key in p)


def test_seal_rejects_any_frozen_change(tmp_path):
    p=tmp_path/'PROTOCOL.json';io.write_json(p,{'phase':'smoke'})
    io.write_json(tmp_path/'SEAL.json',dict(protocol_sha256=io.sha(p),files=[dict(path='PROTOCOL.json',sha256=io.sha(p))]))
    seal=io.sha(tmp_path/'SEAL.json');assert verify_seal(tmp_path,seal)['phase']=='smoke'
    p.write_text('{}')
    with pytest.raises(ValueError,match='changed'):verify_seal(tmp_path,seal)
    with pytest.raises(ValueError,match='commitment'):verify_seal(tmp_path,'wrong')


def test_no_holdout_open_before_all_planned_blocks(tmp_path):
    with pytest.raises(ValueError,match='all frozen blocks'):complete_receipts(tmp_path,{'seeds':[1,2]})
    assert not (tmp_path/'EVALUATION_OPENED.json').exists()


def test_blind_seal_check_does_not_read_targets_but_opening_checks_them(tmp_path):
    protocol=tmp_path/'PROTOCOL.json';io.write_json(protocol,{'phase':'smoke'})
    directory=tmp_path/'evaluation';directory.mkdir()
    target=directory/'TARGET_POOL.json';io.write_json(target,[{'private':True}])
    io.write_json(tmp_path/'SEAL.json',dict(protocol_sha256=io.sha(protocol),files=[
        dict(path='PROTOCOL.json',sha256=io.sha(protocol)),
        dict(path='evaluation/TARGET_POOL.json',sha256=io.sha(target))]))
    seal=io.sha(tmp_path/'SEAL.json');target.write_text('changed')
    assert verify_seal(tmp_path,seal)['phase']=='smoke'
    with pytest.raises(ValueError,match='changed'):verify_seal(tmp_path,seal,include_holdout=True)


def test_late_results_do_not_earn_cpu_credit():
    class M: first_cover={0:0.,1:2.,2:4.};first_exact={1:2.}
    assert endpoint(M(),{1,2},3.)['new_residual']==1
    assert endpoint(M(),{1,2},3.,offset=2.)['new_residual']==0
    assert endpoint(M(),set(),3.)['normalised_auc']==0


def test_paired_sign_flip_and_multiplicity_not_pseudoreplication():
    assert paired_inference([1]*4)['p_one_sided']==1/16
    assert paired_inference([0]*4)['p_one_sided']==1.
    assert paired_inference([-1]*4)['p_one_sided']==1.
    assert holm({'a':.01,'b':.03,'c':.20})=={'a':.03,'b':.06,'c':.20}
    with pytest.raises(ValueError):paired_inference([])


def test_common_short_catalogue_and_selected_certificates_stay_checked():
    basis=load_basis(io.DEFAULT_SOURCE)[0];rules,catalogue=make_catalogue(basis,maximum_mp=2)
    assert catalogue['short_ids'] and len(set(catalogue['short_ids']))==len(catalogue['rules'])
    lib=Library(basis);lib.initialise()
    allowed=merge_rules(lib,rules,[catalogue['rules'][0]['node']])
    assert allowed
    for n in lib.nodes:lib.check(n)
    assert len(allowed)<len(lib.nodes)  # Support does not inherit search rights.


def test_import_recomputes_origins_after_a_duplicate_keeps_another_proof(tmp_path):
    basis=load_basis(io.DEFAULT_SOURCE)[0]
    original=next(a for a in basis if a.label=='ax-1')
    alias=replace(original,label='ax-1-copy')
    # Both libraries admit the same two axiom labels. Their canonical duplicate
    # chooses a different first certificate solely through initial order.
    rule_lib=Library([*basis,alias]);rule_lib.initialise()
    k=next(n.id for n in rule_lib.nodes if n.assertion=='ax-1')
    output,new=rule_lib.accept(rule_lib.propose_mp(k,k));assert new
    lib=Library([alias,*basis]);lib.initialise()
    existing=lib.nodes[lib.known[rule_lib.nodes[k].key]]
    assert existing.origins==('ax-1-copy',)
    allowed=merge_rules(lib,rule_lib,[output])
    assert len(allowed)==1 and lib.nodes[allowed[0]].origins==('ax-1-copy',)
    for n in lib.nodes:lib.check(n)
    path=tmp_path/'merged.mm';path.write_text(proof_text(lib)[0])
    assert io.independent_verify(io.DEFAULT_VERIFIER,path,tmp_path/'verify.log')['status']=='verified'
