"""Finite target-free epochs over a native ZFC basis, with explicit resource guards.

Candidates remain proposals until admission. The persistent archive keeps every
admitted certificate; core membership and active search premises remain distinct.
This engineering policy has no optimality, completeness, or human-value claim.
"""
from collections import Counter
import hashlib
import math
import time

from .basis import ZFC
from .terms import size,variables,canonicalise,render


def family(formula,origins):
    if set(origins)&set(ZFC):return 'set_theory'
    pending=[formula]
    while pending:
        t=pending.pop()
        if t[0] in ('wal','wex','wceq','wcel'):return 'predicate_logic'
        if t[0]!='v':pending.extend(t[1:])
    return 'propositional'


def stable(value):return hashlib.sha256(repr(value).encode()).hexdigest()


def quality(library,n):
    return (1+math.log1p(library.reuse[n.id]))*(1+math.log1p(n.height))/math.sqrt(size(n.formula))


def interleave(groups,limit):
    result=[];offset=0
    while len(result)<limit and any(offset<len(g) for g in groups):
        for g in groups:
            if offset<len(g) and len(result)<limit:result.append(g[offset])
        offset+=1
    return result


def select_core(library,slots,allowed=None):
    groups=[]
    for f in ('propositional','predicate_logic','set_theory'):
        group=[n for n in library.nodes if n.parents and (allowed is None or n.id in allowed) and family(n.formula,n.origins)==f]
        group.sort(key=lambda n:(-library.reuse[n.id],-quality(library,n),size(n.formula),n.id))
        groups.append([n.id for n in group])
    return interleave(groups,slots)


def active_nodes(library,limit,seed,epoch,major=False,allowed=None):
    groups=[]
    for f in ('propositional','predicate_logic','set_theory'):
        group=[n for n in library.nodes if (allowed is None or n.id in allowed) and (not major or n.formula[0]=='wi') and family(n.formula,n.origins)==f]
        # Half established/reused fragments, half a changing exploratory frontier.
        established=sorted(group,key=lambda n:(-quality(library,n),n.id))
        changing=sorted(group,key=lambda n:stable((seed,epoch,n.id)))
        order=[];seen=set()
        for n in interleave([established,changing],len(group)*2):
            if n.id not in seen:seen.add(n.id);order.append(n.id)
        groups.append(order)
    return interleave(groups,limit)


def epoch(library,cfg,index,deadline,allowed=None):
    minors=active_nodes(library,cfg['minors'],cfg['seed'],index,allowed=allowed)
    majors=active_nodes(library,cfg['majors'],cfg['seed']+1,index,True,allowed)
    groups={k:[] for k in ('propositional','predicate_logic','set_theory')}
    seen=set();stats=Counter();checks=0
    def offer(proposal):
        f,dv,assertion,args,parents,origins=proposal
        stats['valid_proposals']+=1
        if size(f)>cfg['max_formula_size']:
            stats['formula_size_rejected']+=1;return
        if 1+max((library.nodes[i].height for i,_ in parents),default=-1)>cfg['max_proof_height']:
            stats['proof_height_rejected']+=1;return
        proof_terms=list(args.values())+[v for _,ps in parents for v in ps.values()]
        canonical,mapping=canonicalise(f,proof_terms)
        mandatory=variables(canonical)
        frame=tuple(sorted({tuple(sorted((mapping[x],mapping[y]))) for x,y in dv
            if mapping[x] in mandatory and mapping[y] in mandatory}))
        key=canonical,frame
        if key in library.known or key in seen:
            stats['duplicate_proposals']+=1;return
        seen.add(key);group=family(f,origins)
        parent_uses=sum(library.reuse[i] for i,_ in parents)
        # No future consumer or human target enters this deliberately simple score.
        score=(1+math.log1p(parent_uses))*(1+math.log1p(len(variables(f))))/math.sqrt(size(f))
        groups[group].append((score,stable(key),proposal))
    for minor in minors:
        for major in majors:
            if checks>=cfg['pairs_per_epoch'] or time.process_time()>=deadline:break
            checks+=1;stats['mp_attempts']+=1
            try:offer(library.propose_mp(minor,major))
            except ValueError as exc:
                stats['invalid_mp_or_dv']+=1
        if checks>=cfg['pairs_per_epoch'] or time.process_time()>=deadline:break
    # Generalisation uses only theorem premises; no open assumptions enter.
    for parent in minors[:cfg['gen_per_epoch']]:
        if time.process_time()>=deadline:break
        for existing in (False,True):
            stats['gen_attempts']+=1
            try:offer(library.propose_gen(parent,existing))
            except ValueError:stats['invalid_gen_or_dv']+=1
    for group in groups.values():group.sort(key=lambda x:(-x[0],x[1]))
    ordered=interleave(list(groups.values()),cfg['admit_per_epoch'])
    admitted=[]
    for _,_,proposal in ordered:
        if len(library.nodes)>=cfg['max_nodes']:break
        node_id,new=library.accept(proposal)
        if new:admitted.append(node_id)
    stats['admitted']=len(admitted)
    return dict(epoch=index,active_minors=minors,active_majors=majors,
        admitted_ids=admitted,candidate_family_counts={k:len(v) for k,v in groups.items()},counters=dict(stats))


def config(**updates):
    c=dict(seed=20261020,max_nodes=4096,core_slots=1024,max_formula_size=160,
        max_proof_height=64,minors=32,majors=96,pairs_per_epoch=3072,
        gen_per_epoch=8,admit_per_epoch=32,max_epochs=1024,cpu_seconds=180.,checkpoint_every=8)
    unknown=set(updates)-set(c)
    if unknown:raise ValueError('Unknown or externally directed configuration fields: '+str(sorted(unknown)))
    c.update(updates)
    if set(c)&{'target','targets','goal','goals','watchlist'}:raise ValueError('No external theorem targets')
    if any(not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<=0 for k in c):
        raise ValueError('All resource bounds must be finite and positive')
    if any(type(value) is not int for key,value in c.items() if key!='cpu_seconds'):
        raise ValueError('Integer resource bounds required')
    if c['max_nodes']<29:raise ValueError('The original basis already contains29 initial statements')
    if c['core_slots']>c['max_nodes']:raise ValueError('Core cannot exceed total node limit')
    return c
