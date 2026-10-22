#!/usr/bin/env python3
"""Build, continue, inspect, expand, and audit a typed Metamath-native library."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import re
import resource
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from metamath_native.basis import Assertion,load_basis,basis_text,LOGIC,ZFC,DEFINITIONS,SYNTAX
from metamath_native.proofs import Library,freeze,proof_text
from metamath_native.terms import size,render
from metamath_native.growth import config,epoch,select_core,family

DEFAULT_SOURCE=ROOT/'output/article1_hpc_runs/skeleton-utility-final-20261006-v1/bundle/inputs/set.mm'
DEFAULT_VERIFIER=ROOT/'output/metamath_native_skeleton_20261020_v1/metamath'


def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value):
    with Path(path).open('xb') as f:f.write(encoded(value))
def write_gzip(path,value):
    with Path(path).open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,compresslevel=1) as z:z.write(encoded(value))
def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def load_committed(directory):
    """Only a hashed receipt, not an unreceipted partial file, commits a state."""
    directory=Path(directory)
    if (directory/'library.json.gz').exists() and (directory/'RESULT.json').exists():
        result=json.loads((directory/'RESULT.json').read_text());path=directory/'library.json.gz'
        if sha(path)!=result['checkpoint_sha256']:raise ValueError('Published checkpoint hash mismatch')
        return read_gzip(path),path
    receipts=sorted((directory/'states').glob('*.receipt.json'))
    if not receipts:raise ValueError('No committed state; do not resume a partial file')
    receipt=json.loads(receipts[-1].read_text());path=directory/'states'/f"{receipt['epoch']:06d}.json.gz"
    if sha(path)!=receipt['sha256']:raise ValueError('Checkpoint receipt mismatch')
    state=read_gzip(path)
    if state['epoch']!=receipt['epoch'] or len(state['nodes'])!=receipt['nodes']:raise ValueError('Checkpoint metadata mismatch')
    return state,path
def restored(state):
    basis=[Assertion(x['label'],x['kind'],freeze(x['expression']),freeze(x['floats']),
        freeze(x['essentials']),freeze(x['disjoint'])) for x in state['basis']]
    return Library.restore(basis,state['nodes'])


def independent_verify(executable,path,out,label='*'):
    wall=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=subprocess.run([str(Path(executable).resolve()),'set scroll continuous',
        'read '+path.name,'verify proof '+label,'exit'],cwd=path.parent,
        text=True,capture_output=True,timeout=300)
    log=result.stdout+'\n'+result.stderr
    with out.open('x') as f:f.write(log)
    # Metamath can exit0 after a command error; inspect its actual report too.
    if (result.returncode!=0 or '?Error' in log or '?Sorry' in log or
            'No errors were found' not in log):
        raise ValueError('Independent Metamath verification did not confirm success: '+str(out))
    if label=='*':
        if 'All proofs in the database were verified' not in log:
            raise ValueError('Parser success is not proof verification: '+str(out))
    else:
        expected=set(re.findall(r'^(native\.[0-9]+) \$p ',path.read_text(),re.M))
        verified=set(re.findall(r'\bnative\.[0-9]+\b',log.split('MM> verify proof '+label,1)[-1]))
        if not expected or verified!=expected:raise ValueError('Not every requested native proof was checked')
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(status='verified',verifier_sha256=sha(executable),log_sha256=sha(out),
        wall_seconds=time.perf_counter()-wall,child_cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
        proof_file_sha256=sha(path),command='VERIFY PROOF '+label)


def summary(lib,core):
    support=set(core);pending=list(core)
    while pending:
        for i,_ in lib.nodes[pending.pop()].parents:
            if i not in support:support.add(i);pending.append(i)
    return dict(archive_nodes=len(lib.nodes),derived_nodes=sum(bool(n.parents) for n in lib.nodes),
        core_anchors=len(core),core_proof_support=len(support),
        reused_core_anchors=sum(lib.reuse[i]>0 for i in core),
        core_families=dict(Counter(family(lib.nodes[i].formula,lib.nodes[i].origins) for i in core)),
        archive_families=dict(Counter(family(n.formula,n.origins) for n in lib.nodes)),
        maximum_proof_height=max(n.height for n in lib.nodes),
        maximum_formula_size=max(size(n.formula) for n in lib.nodes),
        set_theory_axioms_used=sorted(set(ZFC)&{x for n in lib.nodes if n.parents for x in n.origins}),
        disjoint_theorems=sum(bool(n.disjoint) for n in lib.nodes),
        human_theorems_imported=0)


def build(args):
    start=time.process_time();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'states').mkdir();(out/'inputs').mkdir()
    source=args.source.resolve()
    if args.command=='continue':
        state,input_state_path=load_committed(args.input);lib=restored(state)
        source=args.input/'inputs/set.mm'
        if sha(source)!=state['source_sha256']:raise ValueError('Original set.mm changed')
        original,_=load_basis(source)
        if [asdict(a) for a in original]!=[asdict(a) for a in lib.basis]:raise ValueError('Checkpoint changed the original basis')
        cfg=config(**dict(state['config'],max_nodes=args.nodes,core_slots=args.core,cpu_seconds=args.cpu))
        source_keys=set(lib.known)
        current=state['epoch'];prior=dict(path=str(args.input),checkpoint_sha256=sha(input_state_path),
            archive_nodes=len(lib.nodes),epoch=current)
        allowed=set(state['search_ids']) if state.get('search_ids') is not None else None
        if args.from_core:
            lib,core,projection=lib.project(state['core'])
            allowed=set(core)|{n.id for n in lib.nodes if not n.parents}
            prior.update(core_only=True,projected_archive_nodes=len(lib.nodes),initial_core_ids=core,
                projection={str(i):j for i,j in projection.items()})
        if cfg['max_nodes']<=len(lib.nodes):raise ValueError('Continuation must increase node budget')
    else:
        basis,source_hash=load_basis(source);lib=Library(basis);lib.initialise()
        cfg=config(max_nodes=args.nodes,core_slots=args.core,cpu_seconds=args.cpu,seed=args.seed,
            max_formula_size=args.formula_size,max_proof_height=args.proof_height)
        current=0;prior=None;allowed=None;source_keys=set()
    initial_nodes=len(lib.nodes)
    shutil.copyfile(source,out/'inputs/set.mm')
    source_hash=sha(out/'inputs/set.mm');counters=Counter();trace=[];stop='max_epochs';written_epoch=None
    deadline=start+cfg['cpu_seconds']
    def checkpoint():
        nonlocal written_epoch
        state=dict(schema='metamath-native-library/v1',source_sha256=source_hash,
            basis=[asdict(a) for a in lib.basis],nodes=lib.json_nodes(),config=cfg,
            epoch=current,core=select_core(lib,cfg['core_slots'],allowed),search_ids=sorted(allowed) if allowed is not None else None,
            counters=dict(counters),prior=prior)
        path=out/'states'/f'{current:06d}.json.gz';write_gzip(path,state)
        write_json(path.with_suffix('.receipt.json'),dict(epoch=current,sha256=sha(path),nodes=len(lib.nodes)))
        written_epoch=current
        return state,path
    checkpoint()
    for current in range(current+1,current+1+cfg['max_epochs']):
        if time.process_time()>=deadline:stop='cpu_budget';current-=1;break
        if len(lib.nodes)>=cfg['max_nodes']:stop='node_budget';current-=1;break
        row=epoch(lib,cfg,current,deadline,allowed);trace.append(row);counters.update(row['counters'])
        if allowed is not None:allowed.update(row['admitted_ids'])
        if current%cfg['checkpoint_every']==0:
            checkpoint();print(json.dumps(dict(epoch=current,nodes=len(lib.nodes),cpu_sec=time.process_time()-start)),flush=True)
    if written_epoch!=current:state,path=checkpoint()
    else:
        path=out/'states'/f'{current:06d}.json.gz';state=read_gzip(path)
    shutil.copyfile(path,out/'library.json.gz')
    for name,selected,include in [('library.mm',None,None),('core.mm',state['core'],None),
                                  ('against_setmm.mm',None,'inputs/set.mm')]:
        text,ids=proof_text(lib,selected,include)
        with (out/name).open('x') as f:f.write(text)
    checks={}
    if args.verifier:
        for name,label in [('library.mm','*'),('core.mm','*'),('against_setmm.mm','native.*')]:
            checks[name]=independent_verify(args.verifier,out/name,out/(name+'.verify.log'),label)
    data=summary(lib,state['core'])
    result=dict(status='verified' if len(checks)==3 else 'not_independently_verified',summary=data,
        stop_reason=stop,growth_cpu_seconds=time.process_time()-start,config=cfg,verification=checks,
        continuation=dict(initial_archive_nodes=initial_nodes,new_to_this_library=len(lib.nodes)-initial_nodes,
            new_to_original_archive=sum(n.key not in source_keys for n in lib.nodes[initial_nodes:]) if prior else None,
            prior=prior,initial_core_used_by_new_proofs=len(set(prior.get('initial_core_ids',[]))&
                {i for n in lib.nodes[initial_nodes:] for i,_ in n.parents}) if prior else None),
        costs=dict(parent_process_cpu_seconds=time.process_time()-start,
            independent_verifier_cpu_seconds=sum(c['child_cpu_seconds'] for c in checks.values()),
            independent_verifier_wall_seconds=sum(c['wall_seconds'] for c in checks.values())),
        source_sha256=source_hash,source_axiom_labels=dict(logic=LOGIC,ZFC=ZFC,definitions=DEFINITIONS,syntax=SYNTAX),
        checkpoint_sha256=sha(out/'library.json.gz'),trace=trace,
        limits=['Finite experimental library, not complete ZFC or a claim of practical mathematical usefulness.',
            'Only primitive syntax in the explicit basis; no class abstraction or higher-level defined vocabulary.',
            'A verified proof can carry mandatory disjoint-variable restrictions; they must accompany reuse.',
            'The full input set.mm is included for independent verification only; no human $p enters search seeds.',
            'Core selection uses past reuse and typed syntactic families; no optimality or held-out superiority claim.',
            'A CPU-limited candidate window can change the trajectory; checkpoints preserve actual accepted proofs, not unexecuted work.',
            'Archive nodes, selected anchors, proof support, and active search premises remain distinct.'])
    write_json(out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','summary','stop_reason','growth_cpu_seconds')},ensure_ascii=False,indent=2))


def inspect(args):
    state,_=load_committed(args.input);lib=restored(state)
    if args.command=='audit':
        if sha(args.input/'inputs/set.mm')!=state['source_sha256']:raise ValueError('Source hash mismatch')
        original,_=load_basis(args.input/'inputs/set.mm')
        if [asdict(a) for a in original]!=[asdict(a) for a in lib.basis]:raise ValueError('Extracted basis drift')
        for name,selected,include in [('library.mm',None,None),('core.mm',state['core'],None),('against_setmm.mm',None,'inputs/set.mm')]:
            text,ids=proof_text(lib,selected,include)
            if (args.input/name).read_text()!=text:raise ValueError('Proof export mismatch')
        print(json.dumps(dict(status='certificate_and_export_replayed',**summary(lib,state['core'])),indent=2))
    elif args.command=='inspect':
        print(json.dumps(summary(lib,state['core']),indent=2))
        found=0
        for n in lib.nodes:
            expression=' '.join(render(n.formula))
            if args.contains and args.contains not in expression:continue
            if args.core_only and n.id not in state['core']:continue
            print(json.dumps(dict(id=n.id,label=n.label,formula=expression,disjoint=n.disjoint,
                reuse=lib.reuse[n.id],height=n.height,origins=n.origins)))
            found+=1
            if found>=args.limit:break
    elif args.command=='expand':
        if not 0<=args.id<len(lib.nodes):raise ValueError('Unknown node')
        text,ids=proof_text(lib,[args.id])
        with args.out.open('x') as f:f.write(text)
        print(json.dumps(dict(formula=' '.join(render(lib.nodes[args.id].formula)),proof_nodes=len(ids),
            expanded_shared_DAG=str(args.out),mandatory_disjoint=lib.nodes[args.id].disjoint),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('build','continue'):
        p=sub.add_parser(command);p.add_argument('--out',type=Path,required=True)
        p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
        p.add_argument('--verifier',type=Path,default=DEFAULT_VERIFIER)
        p.add_argument('--nodes',type=int,default=4096);p.add_argument('--core',type=int,default=1024)
        p.add_argument('--cpu',type=float,default=180.);p.add_argument('--seed',type=int,default=20261020)
        p.add_argument('--formula-size',type=int,default=160);p.add_argument('--proof-height',type=int,default=64)
        if command=='continue':
            p.add_argument('--input',type=Path,required=True)
            p.add_argument('--from-core',action='store_true',help='Discard non-core archive outside proof support; only anchors, original axioms, and fresh conclusions serve as premises')
    for command in ('inspect','audit','expand'):
        p=sub.add_parser(command);p.add_argument('--input',type=Path,required=True)
        if command=='inspect':
            p.add_argument('--contains',default='');p.add_argument('--limit',type=int,default=10);p.add_argument('--core-only',action='store_true')
        if command=='expand':p.add_argument('--id',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command in ('build','continue'):build(args)
    else:inspect(args)


if __name__=='__main__':main()
