"""Engineering prototype: proof-backed anchors + bounded local completion.

The target-free selector is a heuristic, NOT a globally optimal method. It never
receives query targets. Supporting proof nodes are not exposed as search facts.
Local completion is explicitly target-conditioned (inverse cognition), not
evidence of intrinsic/emergent topology. Only prop and modal K are supported.
"""
from collections import Counter
import contextlib
import gc
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import signal
import sys
import time

from certificate import Checker, InvalidProof, axioms, canonical, encode_formula, term

_BUNDLE = Path(__file__).resolve().parents[2]
ROOT = _BUNDLE


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def size(t):
    return 1 if t[0]=='v' else 1+sum(size(x) for x in t[1:])


def features(t):
    """Shallow syntactic coverage, not a claim of semantic equivalence."""
    def shape(x,depth):
        if x[0]=='v':return 'v'
        if depth==0:return x[0]
        return x[0]+'('+','.join(shape(y,depth-1) for y in x[1:])+')'
    result={'root:'+shape(t,2)}
    def visit(x):
        if x[0]!='v':
            result.add('local:'+shape(x,1))
            for child in x[1:]:visit(child)
    visit(t)
    if t[0]=='i':
        result.add('antecedent:'+shape(t[1],2))
        result.add('consequent:'+shape(t[2],2))
    return frozenset(result)


def ancestors(records, roots):
    found=set();pending=list(roots)
    while pending:
        n=pending.pop()
        if n in found:continue
        if n not in records:raise InvalidProof(f'Missing proof dependency {n}')
        found.add(n);pending.extend(records[n][3])
    return found


def load_source(cell):
    start=time.process_time()
    receipt=json.loads((cell/'result.json').read_text())
    file=cell/'certificate.jsonl.gz'
    if digest(file)!=receipt['verification']['certificate_sha256']:
        raise InvalidProof('Source certificate hash mismatch')
    if receipt['verification']['status']!='verified':
        raise InvalidProof('Source was not verified')
    records={};reuse=Counter();work={}
    with gzip.open(file,'rt') as f:
        for line in f:
            nid,formula,rule,parents=json.loads(line)
            if nid in records or any(p not in records or p>=nid for p in parents):
                raise InvalidProof('Source order, cycle, duplicate or missing parent')
            formula=canonical(term(formula))
            records[nid]=(nid,formula,rule,tuple(parents))
            reuse.update(set(parents))
            work[nid]=min(10**9,1+sum(work[p] for p in parents))
    if len(records)!=receipt['verification']['nodes']:
        raise InvalidProof('Source count mismatch')
    return dict(records=records,reuse=reuse,work=work,receipt=receipt,
        source_certificate_sha256=digest(file),load_cpu_sec=time.process_time()-start)


def build(source, policy='coverage', anchors=32, candidate_cap=2048, seed=101, eligible_ids=None):
    """All methods use the same target-free candidate pool and anchor count.

Coverage objective: sum_f max_{v in S, f in features(v)} q(v), with
q=(1+log(1+direct_uses))*(1+log(first_proof_tree_work)). Greedy marginal
gain is divided by newly required proof-record bytes. This cost depends on S,
so no cardinality/submodular approximation guarantee is asserted.
"""
    if policy not in {'axioms','smallest','reuse','coverage','random','pool'}:
        raise ValueError('Unknown policy')
    if anchors<0 or candidate_cap<1:raise ValueError('Invalid skeleton budget')
    if policy=='axioms':return axiom_package(source['receipt']['config']['substrate'])
    start=time.process_time();records=source['records']
    roots={i for i,r in records.items() if not r[3]}
    eligible=[i for i,r in records.items() if r[3]]
    if eligible_ids is not None:
        allowed=set(eligible_ids)
        if not allowed<=set(records):raise ValueError('Unknown eligible candidate')
        eligible=[i for i in eligible if i in allowed]
    sizes={i:size(records[i][1]) for i in eligible}
    order_key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode()).digest()
    # Shared pool: short formulas, productive formulas, then seed-stable sample.
    third=max(1,candidate_cap//3)
    pool=set(sorted(eligible,key=lambda i:(sizes[i],i))[:third])
    pool.update(sorted(eligible,key=lambda i:(-source['reuse'][i],sizes[i],i))[:third])
    pool.update(sorted((i for i in eligible if i not in pool),key=order_key)[:max(0,candidate_cap-len(pool))])
    pool=sorted(pool);selected=[];support=set(roots)
    record_bytes={}
    def bytes_for(i):
        if i not in record_bytes:
            record_bytes[i]=len(json.dumps(records[i],separators=(',',':')).encode())+1
        return record_bytes[i]
    count=len(pool) if policy=='pool' else min(anchors,len(pool))
    if policy!='coverage':
        if policy=='smallest':order=sorted(pool,key=lambda i:(sizes[i],i))
        elif policy=='reuse':order=sorted(pool,key=lambda i:(-source['reuse'][i],sizes[i],i))
        elif policy=='pool':order=pool
        else:order=sorted(pool,key=order_key)
        selected=order[:count];support.update(ancestors(records,selected))
    else:
        fsets={i:features(records[i][1]) for i in pool}
        q={i:(1+math.log1p(source['reuse'][i]))*(1+math.log(source['work'][i])) for i in pool}
        closures={i:ancestors(records,[i]) for i in pool}
        covered={};remaining=set(pool)
        for _ in range(count):
            best=None;best_key=None
            for i in sorted(remaining):
                gain=sum(max(0,q[i]-covered.get(f,0)) for f in fsets[i])
                added=sum(bytes_for(n) for n in closures[i]-support)
                key=(gain/max(1,added),gain,-sizes[i],-i)
                if best_key is None or key>best_key:best,best_key=i,key
            selected.append(best);remaining.remove(best);support.update(closures[best])
            for f in fsets[best]:covered[f]=max(covered.get(f,0),q[best])
    selection_cpu=time.process_time()-start
    check_start=time.process_time();checker=Checker(source['receipt']['config']['substrate'])
    proof=[records[i] for i in sorted(support)]
    for row in proof:checker.accept(*row)
    package=dict(schema='proof-backed-skeleton/v1',policy=policy,
        substrate=source['receipt']['config']['substrate'],anchors=selected,axioms=sorted(roots),
        proof_records=proof,source_certificate_sha256=source['source_certificate_sha256'],
        selection=dict(requested_anchors=anchors,candidate_cap=candidate_cap,candidates=len(pool),seed=seed,
            objective='Syntactic feature coverage weighted by direct use and first-proof work, per marginal proof bytes; heuristic only'),
        costs=dict(source_load_cpu_sec=source['load_cpu_sec'],selection_cpu_sec=selection_cpu,
            skeleton_check_cpu_sec=time.process_time()-check_start,
            source_precomputation_cpu_sec=source['receipt']['whole_worker_cpu_sec'],
            source_archive_bytes=source['receipt']['archive_bytes'],
            source_certificate_bytes=source['receipt']['verification']['certificate_bytes'],
            support_record_bytes=sum(bytes_for(i) for i in support)),
        verification=checker.summary(),
        limits=['No targets used in selection; no held-out evaluation or global optimality claim.',
                'The source archive is a supplied precomputation; its time and bytes must not be treated as zero.',
                'Supporting proof nodes are not available to retrieval or generation.',
                'Only one first proof per anchor; no completeness or equivalence claim.'])
    package['package_sha256']=package_hash(package)
    return package


def package_hash(package):
    return hashlib.sha256(json.dumps({k:v for k,v in package.items() if k!='package_sha256'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def axiom_package(substrate):
    """A genuine from-scratch control: no source-archive cost or dependency."""
    start=time.process_time()
    proof=[(i,f,name,()) for i,(name,f) in enumerate(axioms(substrate).items())]
    checker=Checker(substrate)
    for row in proof:checker.accept(*row)
    p=dict(schema='proof-backed-skeleton/v1',policy='axioms',substrate=substrate,
        anchors=[],axioms=[r[0] for r in proof],proof_records=proof,
        source_certificate_sha256=None,selection=dict(requested_anchors=0,candidates=0),
        costs=dict(source_load_cpu_sec=0.,selection_cpu_sec=0.,
            skeleton_check_cpu_sec=time.process_time()-start,source_precomputation_cpu_sec=0.,
            source_archive_bytes=0,source_certificate_bytes=0,
            support_record_bytes=sum(len(json.dumps(r).encode())+1 for r in proof)),
        verification=checker.summary(),limits=['Original axiom schemas only; no precomputed source graph.'])
    p['package_sha256']=package_hash(p)
    return p


def validate(package):
    if package_hash(package)!=package['package_sha256']:raise InvalidProof('Skeleton hash mismatch')
    checker=Checker(package['substrate'])
    records={}
    for raw in package['proof_records']:
        nid,formula,rule,parents=raw
        row=(nid,canonical(term(formula)),rule,tuple(parents))
        checker.accept(*row);records[nid]=row
    if any(i not in records for i in package['anchors']+package['axioms']):
        raise InvalidProof('Anchor has no certificate')
    if any(records[i][3] for i in package['axioms']):raise InvalidProof('Derived fact advertised as axiom')
    if {records[i][2] for i in package['axioms']}!=set(checker.axioms):
        raise InvalidProof('Incomplete original axiom base')
    return records,checker


def decode(t):
    from core.formula import PropVar,Implies,Not,Box
    if t[0]=='v':return PropVar('s'+str(t[1]))
    if t[0]=='i':return Implies(decode(t[1]),decode(t[2]))
    if t[0]=='n':return Not(decode(t[1]))
    if t[0]=='b':return Box(decode(t[1]))
    raise ValueError('Unsupported term')


def retrieve(package,records,goal,limit):
    """Goal-conditioned neighborhood. Only anchors, never supporting nodes."""
    from core.schema import match
    gfeatures=features(goal);gf=decode(goal)
    def key(nid):
        f=records[nid][1]
        # Concrete implication bridges outrank shared shallow syntax.
        bridge=f[0]=='i' and f[2][0]!='v' and match(decode(f[2]),gf) is not None
        overlap=len(features(f)&gfeatures)/max(1,len(features(f)|gfeatures))
        return (-int(f==goal),-int(bridge),-overlap,size(f),nid)
    return sorted(package['anchors'],key=key)[:limit]


class Found(Exception):pass
class Budget(Exception):pass
class BackwardLimit(Exception):pass


def backward_local(records,checker,seed_ids,goal,max_depth,max_nodes,max_visits=1000):
    """Bounded sufficient backward rules; every accepted step is replayed forward.

Only exact seeds and locally proved formulas close leaves. Supporting archive
nodes cannot close them. Matching a consequent is a proposal, never a proof.
"""
    from core.schema import match,apply_subst,schema_mp
    known={records[i][1]:i for i in seed_ids}
    majors=[i for i in seed_ids if records[i][1][0]=='i' and records[i][1][2][0]!='v']
    next_id=max(records)+1
    stats=dict(goals_visited=0,major_match_attempts=0,forward_replay_attempts=0,new_nodes=0)
    failed=set()
    def append(f,rule,parents):
        nonlocal next_id
        if f in known:return known[f]
        if stats['new_nodes']>=max_nodes:raise BackwardLimit('node_limit')
        checker.accept(next_id,f,rule,parents)
        records[next_id]=(next_id,f,rule,tuple(parents))
        known[f]=next_id;next_id+=1;stats['new_nodes']+=1
        return next_id-1
    def visit(f,depth,path):
        if f in known:return known[f]
        if depth==0 or f in path or (f,depth) in failed:return None
        stats['goals_visited']+=1
        if stats['goals_visited']>max_visits:raise BackwardLimit('goal_limit')
        path=path|{f}
        if f[0]=='b' and checker.substrate=='modal':
            minor=visit(f[1],depth-1,path)
            if minor is not None:return append(f,'Nec',[minor])
        candidates=[];target=decode(f)
        for major in majors:
            stats['major_match_attempts']+=1
            mf=decode(records[major][1]);sigma=match(mf.consequent,target)
            if sigma is not None:
                sub=canonical(encode_formula(apply_subst(sigma,mf.antecedent)))
                if sub!=f:candidates.append((size(sub),major,sub))
        for _,major,sub in sorted(candidates)[:8]:
            minor=visit(sub,depth-1,path)
            if minor is None:continue
            stats['forward_replay_attempts']+=1
            result=schema_mp(decode(records[minor][1]),decode(records[major][1]))
            if result is not None and canonical(encode_formula(result))==f:
                return append(f,'MP',[minor,major])
        failed.add((f,depth))
        return None
    try:
        result=visit(goal,max_depth,set())
        stats['stop_reason']='proved' if result is not None else 'exhausted'
    except BackwardLimit as exc:
        # A phase-local search bound is not the CPU deadline for the full query.
        # Return normally so the forward generator can use the remaining budget.
        result=None;stats['stop_reason']=str(exc)
    return result,stats


def complete(package,goal,*,cpu_budget=2.,radius=3,local_width=16,retrieve_count=16,max_nodes=10000,
             prune_mode='none',prune_patience=2):
    """Use the existing generator, but expand seed proofs back to original axioms.

Budget covers validation, retrieval, generation and final replay. A proof whose
finalization exceeds the CPU deadline is reported but NOT credited as in-budget.
Exact canonical goal matching only; no conjecture or subsumption is called proof.
"""
    if cpu_budget<=0 or radius<1 or local_width<1 or retrieve_count<0 or max_nodes<4:
        raise ValueError('Invalid completion budget')
    if prune_mode not in ('none','pool','major','both') or prune_patience<1:
        raise ValueError('Invalid local pruning configuration')
    start=time.process_time();wall=time.perf_counter()
    if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
    from core.schema import canonicalize_with_hash
    from hilbert.schema_generator import SchemaGenerator
    from hilbert.utility import UtilityRanker,UtilityWeights
    if signal.getitimer(signal.ITIMER_PROF)[0] or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('Refusing to replace an active timer')
    goal=canonical(term(goal));records,checker=validate(package)
    original_ids=set(records)
    selected=retrieve(package,records,goal,retrieve_count)
    seed_ids=package['axioms']+selected
    by_formula={records[i][1]:i for i in seed_ids}
    target=decode(goal);target_hash=canonicalize_with_hash(target)[1]
    target_id=by_formula.get(goal);g=None;stop='initial';backward_stats={}
    validation_retrieval_cpu=time.process_time()-start
    if target_id is None and validation_retrieval_cpu<cpu_budget:
        base=SchemaGenerator;extra={}
        if prune_mode!='none':
            from mechanisms import pruning_class
            base=pruning_class(SchemaGenerator)
            extra=dict(prune_mode=prune_mode,prune_patience=prune_patience)
        class Seeded(base):
            def _init_axioms(self):
                super()._init_axioms()
                for i in selected:self._add_schema(decode(records[i][1]),0,'seed',[])
            def _add_schema(self,formula,depth,rule,premise_ids,**kwargs):
                added=super()._add_schema(formula,depth,rule,premise_ids,**kwargs)
                if depth>0 and target_hash in self._entries:raise Found()
                return added
        old_prof=signal.getsignal(signal.SIGPROF);old_alarm=signal.getsignal(signal.SIGALRM)
        def expire(signum,frame):raise Budget()
        signal.signal(signal.SIGPROF,expire);signal.signal(signal.SIGALRM,expire)
        signal.setitimer(signal.ITIMER_PROF,max(.001,cpu_budget-(time.process_time()-start)))
        signal.setitimer(signal.ITIMER_REAL,max(5,cpu_budget*3))
        try:
            target_id,backward_stats=backward_local(records,checker,seed_ids,goal,radius,max_nodes)
            if target_id is not None:stop='backward_verified'
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    g=Seeded(propositional_base='metamath_prop' if package['substrate']=='prop' else 'modal_k',
                        theory=None,max_depth=radius,max_schemas=max_nodes,max_formula_size=200,
                        n_workers=1,beam_width=local_width,
                        utility_ranker=UtilityRanker([('query',target)],UtilityWeights()),
                        provenance_mode='full',subsumption_mode='off',_use_env_overrides=False,**extra)
                    g.generate()
                stop=g.stats.get('stopped_reason','radius_or_node_limit')
        except Found:stop='found'
        except Budget:stop='budget'
        finally:
            signal.setitimer(signal.ITIMER_PROF,0);signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGPROF,old_prof);signal.signal(signal.SIGALRM,old_alarm)
        if g is not None and target_hash in g._entries:
            local={e.id:e for e in g._entries.values()};end=g._entries[target_hash].id
            needed=set();pending=[end]
            while pending:
                i=pending.pop()
                if i in needed:continue
                needed.add(i);pending.extend(local[i].premise_ids)
            mapping={};next_id=max(records)+1
            for i in sorted(needed):
                e=local[i];f=canonical(encode_formula(e.formula))
                if not e.premise_ids:
                    if f not in by_formula:raise InvalidProof('Uncertified local seed')
                    mapping[i]=by_formula[f]
                else:
                    row=(next_id,f,e.rule,tuple(mapping[p] for p in e.premise_ids))
                    checker.accept(*row);records[next_id]=row
                    mapping[i]=next_id;next_id+=1
            target_id=mapping[end]
    if target_id is not None:
        proof=[records[i] for i in sorted(ancestors(records,[target_id]))]
        final=Checker(package['substrate'])
        for row in proof:final.accept(*row)
        if final.nodes[target_id]!=goal:raise InvalidProof('Wrong final goal')
    else:proof=[]
    expanded_proof_height=checker.heights[target_id] if target_id is not None else None
    generated=len(g._entries) if g is not None else 0
    forward_stats={k:g.stats.get(k,0) for k in
        ('mp_candidate_pairs_raw','mp_candidate_pairs_filtered','mp_successes',
         'mp_filtered_size','mp_filtered_dup')} if g is not None else {}
    pruning=dict(mode=prune_mode,patience=prune_patience,
        pruned_pool=getattr(g,'pruned_pool',0),pruned_major=getattr(g,'pruned_major',0))
    del g;gc.collect()
    elapsed=time.process_time()-start
    solved=target_id is not None
    return dict(schema='skeleton-local-completion/v1',goal=goal,verified=solved,
        credited=solved and elapsed<=cpu_budget,
        status=('verified_in_budget' if elapsed<=cpu_budget else 'verified_over_budget') if solved else 'not_proved',
        stop_reason=stop,query_cpu_sec=elapsed,query_wall_sec=time.perf_counter()-wall,
        validation_retrieval_cpu_sec=validation_retrieval_cpu,cpu_budget_sec=cpu_budget,
        package_sha256=package['package_sha256'],selected_anchors=selected,
        local_generated_nodes=generated,backward=backward_stats,forward=forward_stats,pruning=pruning,
        size_cap_bound=bool(forward_stats.get('mp_filtered_size',0)),max_formula_size=200,
        new_proof_nodes=sum(row[0] not in original_ids for row in proof),
        expanded_proof_height=expanded_proof_height,
        proof_records=proof,target_id=target_id,radius=radius,local_width=local_width,
        scope='Goal-conditioned local completion, with seeds expanded to original axioms; not a scale-free emergence experiment.')
