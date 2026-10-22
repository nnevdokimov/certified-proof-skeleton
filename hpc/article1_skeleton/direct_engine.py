"""Target-free, certificate-preserving forward epochs. No theorem query API.

Only the production SchemaGenerator generates experimental trajectories. The
separate direct_oracle enumerator is a read-only finite reference. The persistent
archive stores ALL checked first proofs; it is not hidden behind the small active
skeleton. Selection and export costs count toward the cumulative budget.
"""
from collections import Counter
import contextlib
import gc
import gzip
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import time

from certificate import Checker,InvalidProof,axioms,canonical,encode_formula,mp
from skeleton_local import ROOT,ancestors,build,decode,digest,package_hash
from mechanisms import pruning_class
from worker import rss,write_json


class StopWindow(Exception):pass


class Archive:
    def __init__(self,substrate):
        self.substrate=substrate;self.records={};self.formulas={}
        self.checker=Checker(substrate);self.reuse=Counter();self.work={};self.axiom_ids=[]
        for name,f in axioms(substrate).items():
            nid=self.add(f,name,());self.axiom_ids.append(nid)

    def add(self,f,rule,parents):
        f=canonical(f);parents=tuple(parents)
        if rule=='MP':
            if len(parents)!=2 or mp(*(self.records[p][1] for p in parents))!=f:
                raise InvalidProof('Invalid mapped MP')
        elif rule=='Nec':
            if self.substrate!='modal' or len(parents)!=1 or canonical(('b',self.records[parents[0]][1]))!=f:
                raise InvalidProof('Invalid mapped Nec')
        elif rule not in self.checker.axioms or parents or f!=self.checker.axioms[rule]:
            raise InvalidProof('Not an original axiom or checked rule')
        if f in self.formulas:return self.formulas[f]
        nid=len(self.records);self.checker.accept(nid,f,rule,parents)
        self.records[nid]=(nid,f,rule,parents);self.formulas[f]=nid
        self.reuse.update(set(parents))
        self.work[nid]=min(10**9,1+sum(self.work[p] for p in parents))
        return nid

    def merge(self,entries,seeds,mapping=None):
        """Replay local first proofs; depth-zero seeds map to old certificates."""
        if mapping is None:mapping={}
        permitted=set(self.axiom_ids)|set(seeds);before=len(self.records);duplicates=0
        for e in sorted(entries,key=lambda x:x.id):
            if e.id in mapping:continue
            f=canonical(encode_formula(e.formula))
            if e.rule=='epoch_seed':
                nid=self.formulas.get(f)
                if e.premise_ids or nid not in permitted:raise InvalidProof('Uncertified epoch seed')
            else:
                if any(p not in mapping for p in e.premise_ids):raise InvalidProof('Missing local parent')
                known=f in self.formulas
                nid=self.add(f,e.rule,[mapping[p] for p in e.premise_ids])
                if known and e.premise_ids:duplicates+=1
            mapping[e.id]=nid
        return mapping,dict(new_archive_nodes=len(self.records)-before,rediscovered_archive_formulas=duplicates)

    def export(self,path):
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=1) as f:
            for row in self.records.values():f.write(json.dumps(row,separators=(',',':'))+'\n')
        return dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size,nodes=len(self.records))

    def package(self,anchors,policy):
        support=ancestors(self.records,list(anchors)+self.axiom_ids)
        proof=[self.records[i] for i in sorted(support)]
        checker=Checker(self.substrate)
        for row in proof:checker.accept(*row)
        package=dict(schema='proof-backed-skeleton/v1',substrate=self.substrate,policy=policy,
            anchors=sorted(anchors),axioms=list(self.axiom_ids),proof_records=proof,
            verification=checker.summary(),selection=dict(target_free=True),
            limits=['Active skeleton plus its proof support; the full discovery archive is accounted separately.'])
        package['package_sha256']=package_hash(package)
        return package


def make_generator(archive,seeds,cfg,epoch):
    sys.path.insert(0,str(ROOT/'src')) if str(ROOT/'src') not in sys.path else None
    from hilbert.schema_generator import SchemaGenerator
    from hilbert.utility import UtilityRanker,UtilityWeights
    base=pruning_class(SchemaGenerator) if cfg['prune_mode']!='none' else SchemaGenerator
    class Forward(base):
        def _init_axioms(self):
            super()._init_axioms()
            for nid in seeds:self._add_schema(decode(archive.records[nid][1]),0,'epoch_seed',[])

        def _beam_tiebreak_key(self,entry):
            round_id=self.current_depth if cfg['policy']=='continuous' else (self.epoch-1)*cfg['epoch_steps']+self.current_depth
            return hashlib.sha256(f'{cfg["seed"]}:{round_id}:{entry.schema_hash}'.encode()).digest()

        def _generate_depth(self,depth):
            if rss()>cfg['soft_memory_bytes']:raise StopWindow('soft_memory')
            self.current_depth=depth
            result=super()._generate_depth(depth)
            self.pending_metrics=dict(result[1],depth=depth)
            if self.beam_width is None:self.capture()
            return result

        def _activate_from_pool(self,depth):
            result=super()._activate_from_pool(depth)
            self.capture()
            return result

        def capture(self):
            # The generator's node cap may cut a layer short. Retain the last
            # whole layer, never advertise the capped prefix as a full layer.
            if len(self._entries)>=cfg['max_nodes']:return
            metrics=self.window_metrics+[self.pending_metrics]
            # Assign a whole, self-contained checkpoint only after generation
            # and activation have finished; interruption retains the old tuple.
            snapshot=(tuple(self._entries.values()),metrics,
                dict(depth=self.current_depth,active_ids=sorted(self._activated_ids),
                    passive=len(self._passive_pool),pruned_pool=getattr(self,'pruned_pool',0),
                    pruned_major=getattr(self,'pruned_major',0),
                    size_exclusions=self.stats.get('mp_filtered_size',0)))
            self.snapshot=snapshot;self.window_metrics=metrics

    zero=dict(resonance_exact=0.,resonance_shape=0.,subsume=0.,goal_step=0.,relevance=0.,importance=0.)
    weights=UtilityWeights(**zero,rediscovery=float(cfg['intrinsic_utility']),size=1.)
    extra=dict(prune_mode=cfg['prune_mode'],prune_patience=2) if cfg['prune_mode']!='none' else {}
    g=Forward(propositional_base='metamath_prop' if archive.substrate=='prop' else 'modal_k',
        theory=None,max_depth=cfg['epoch_steps'],max_schemas=cfg['max_nodes'],
        max_formula_size=cfg['max_formula_size'],n_workers=1,beam_width=cfg['beam'],
        utility_ranker=UtilityRanker([],weights) if cfg['beam'] is not None else None,
        provenance_mode='full',subsumption_mode='off',_use_env_overrides=False,**extra)
    g.snapshot=None;g.window_metrics=[];g.pending_metrics={};g.current_depth=0;g.epoch=epoch
    return g


def forward_window(g,allowance):
    if signal.getitimer(signal.ITIMER_PROF)[0] or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('Refusing to overwrite a running timer')
    oldp=signal.getsignal(signal.SIGPROF);olda=signal.getsignal(signal.SIGALRM)
    running=False
    def stop(signum,frame):
        nonlocal running
        # A one-shot window can raise at most once. In particular a late
        # SIGPROF/SIGALRM must not escape through timer teardown.
        if running:
            running=False
            raise StopWindow('cpu_frontier' if signum==signal.SIGPROF else 'wall_frontier')
    reason='unknown';start=time.process_time()
    try:
        signal.signal(signal.SIGPROF,stop);signal.signal(signal.SIGALRM,stop)
        try:
            running=True
            signal.setitimer(signal.ITIMER_PROF,max(.001,allowance))
            signal.setitimer(signal.ITIMER_REAL,max(10.,allowance*4))
            with contextlib.redirect_stdout(io.StringIO()):g.generate()
            reason=g.stats.get('stopped_reason','finished')
        finally:
            # The outer except also covers delivery just before this assignment.
            running=False
    except StopWindow as exc:reason=str(exc)
    finally:
        # Discard pending signals owned by these timers before restoring the
        # caller's handlers (which may be SIG_DFL). Existing timers are refused
        # above; this function is only used by isolated main-thread workers.
        signal.signal(signal.SIGPROF,signal.SIG_IGN);signal.signal(signal.SIGALRM,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_PROF,0);signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGPROF,oldp);signal.signal(signal.SIGALRM,olda)
    return reason,time.process_time()-start


def run_epochs(cfg,out):
    """Cumulative budget includes generation, checks, selection and checkpoint export."""
    if any(k in cfg for k in ('targets','goal','watchlist')):raise ValueError('No goals in direct experiment')
    if cfg['policy'] not in ('continuous','reset','all','smallest','random','reuse','coverage'):
        raise ValueError('Unknown direct policy')
    if (not cfg['cpu_budgets'] or min(cfg['cpu_budgets'])<=0 or
        cfg['cpu_budgets']!=sorted(set(cfg['cpu_budgets'])) or
        not 0<cfg['generation_fraction']<1 or cfg['epoch_steps']<1 or
        cfg['max_epochs']<1 or cfg['anchors']>cfg['candidate_cap']):
        raise ValueError('Invalid direct resource bounds')
    out.mkdir(parents=True,exist_ok=False)
    start=time.process_time();wall=time.perf_counter();archive=Archive(cfg['substrate'])
    rows=[];g=None;seeds=[];mapping={};initial_seeds=[];stop='max_epochs';discarded_cpu=0.
    for epoch in range(1,cfg['max_epochs']+1):
        remaining=max(cfg['cpu_budgets'])-(time.process_time()-start)
        if remaining<cfg.get('min_remaining_sec',.05):stop='total_cpu';break
        if rss()>cfg['soft_memory_bytes']:stop='soft_memory';break
        if len(seeds)>cfg['max_seed_nodes']:stop='all_lemma_seed_guard';break
        constructing=time.process_time()
        if cfg['policy']!='continuous' or g is None:
            g=make_generator(archive,seeds,cfg,epoch);mapping={};initial_seeds=list(seeds)
        else:
            g.max_depth+=cfg['epoch_steps'];g.epoch=epoch
        g.snapshot=None;g.window_metrics=[]
        construction_cpu=time.process_time()-constructing
        remaining=max(cfg['cpu_budgets'])-(time.process_time()-start)
        if remaining<=0:stop='construction_budget';break
        # Reserve part of the remaining budget for proof checks and persistence.
        # This is not a hard total-time guarantee: over-budget commits get no credit.
        reason,generation_cpu=forward_window(g,remaining*cfg['generation_fraction'])
        if g.snapshot is None:
            discarded_cpu+=generation_cpu;stop=reason;break
        entries,metrics,snap=g.snapshot
        merge_start=time.process_time();mapping,merged=archive.merge(entries,initial_seeds,mapping)
        merge_cpu=time.process_time()-merge_start
        folder=out/f'epoch_{epoch:03d}';folder.mkdir()
        export_start=time.process_time();cert=archive.export(folder/'certificate.jsonl.gz')
        archive_export_cpu=time.process_time()-export_start
        selection_start=time.process_time()
        if cfg['policy']=='continuous':
            active={mapping[i] for i in snap['active_ids']}-set(archive.axiom_ids)
            package=archive.package(active,'continuous');seeds=[]
        elif cfg['policy']=='all':
            seeds=[i for i in archive.records if i not in archive.axiom_ids]
            package=archive.package(seeds,'all')
        elif cfg['policy']=='reset':
            seeds=[];package=archive.package([],'reset')
        else:
            # This source is the current live, freshly persisted archive, not a
            # free precomputation. Worker timings below are authoritative and
            # never sum this diagnostic source cost repeatedly across epochs.
            source=dict(records=archive.records,reuse=archive.reuse,work=archive.work,load_cpu_sec=0.,
                source_certificate_sha256=cert['sha256'],receipt=dict(config=dict(substrate=cfg['substrate']),
                    whole_worker_cpu_sec=time.process_time()-start,archive_bytes=cert['bytes'],
                    verification=dict(certificate_bytes=cert['bytes'])))
            package=build(source,cfg['policy'],cfg['anchors'],cfg['candidate_cap'],cfg['seed']+epoch)
            seeds=list(package['anchors'])
        selection_cpu=time.process_time()-selection_start
        export_start=time.process_time()
        with gzip.open(folder/'skeleton.json.gz','wt',encoding='utf-8',compresslevel=1) as f:
            json.dump(package,f,separators=(',',':'))
        skeleton_bytes=(folder/'skeleton.json.gz').stat().st_size
        skeleton_export_cpu=time.process_time()-export_start
        n=len(entries);majors=sum(type(e.formula).__name__=='Implies' for e in entries)
        cleanup_start=time.process_time()
        if cfg['policy']!='continuous':g=None
        del entries;gc.collect()
        row=dict(epoch=epoch,local_generation=snap['depth'],archive_nodes=len(archive.records),
            expanded_max_proof_height=max(archive.checker.heights.values()),active_anchors=len(package['anchors']),
            active_proof_support_nodes=len(package['proof_records']),passive_nodes_before_reset=snap['passive'],
            local_materialized_nodes=n,local_all_pairs_upper=n*majors,
            **merged,committed_generation_metrics=metrics,
            construction_cpu_sec=construction_cpu,generation_cpu_sec=generation_cpu,
            certificate_merge_cpu_sec=merge_cpu,selection_and_support_check_cpu_sec=selection_cpu,
            archive_export_cpu_sec=archive_export_cpu,skeleton_export_cpu_sec=skeleton_export_cpu,
            cleanup_cpu_sec=time.process_time()-cleanup_start,stop_reason=reason,
            pruned_pool=snap['pruned_pool'],pruned_major=snap['pruned_major'],
            size_exclusions=snap['size_exclusions'],peak_rss_bytes=rss(),
            complete_local_layers=True,
            archive_certificate=cert,skeleton_file_sha256=digest(folder/'skeleton.json.gz'),
            skeleton_bytes=skeleton_bytes,package_sha256=package['package_sha256'],
            archive_verification=archive.checker.summary())
        # Serialization of the substantive checkpoint is charged before its
        # timestamp. A final tiny timestamp/progress write is reported separately.
        write_json(folder/'checkpoint.json',row)
        row['commit_cpu_sec']=time.process_time()-start
        row['commit_wall_sec']=time.perf_counter()-wall
        row['within_max_budget']=row['commit_cpu_sec']<=max(cfg['cpu_budgets'])
        write_json(folder/'checkpoint.json',row);rows.append(row)
        write_json(out/'progress.json',dict(config=cfg,rows=rows))
        print(json.dumps(dict(epoch=epoch,formulas=row['archive_nodes'],anchors=row['active_anchors'],
            cpu=round(row['commit_cpu_sec'],3),credited=row['within_max_budget'])),flush=True)
        if not row['within_max_budget']:stop='finalization_over_budget';break
        if reason not in ('max_depth',):stop=reason;break
        if len(archive.records)>=cfg['max_archive_nodes']:stop='archive_node_guard';break
    del g;gc.collect()
    readouts=[dict(cpu_budget_sec=b,checkpoint=next((r for r in reversed(rows) if r['commit_cpu_sec']<=b),None))
        for b in cfg['cpu_budgets']]
    return dict(schema='direct-epochs-cell/v1',config=cfg,status='completed',stop_reason=stop,
        rows=rows,budget_readouts=readouts,whole_algorithm_cpu_sec=time.process_time()-start,
        whole_algorithm_wall_sec=time.perf_counter()-wall,discarded_window_generation_cpu_sec=discarded_cpu,
        peak_rss_bytes=rss(),target_free=True,
        limits=['No target formulas or backwards search; original axioms and intrinsic scores only.',
            'Archive includes all admitted first proofs and is accounted separately from active proof support.',
            'Only complete local generations are admitted; interrupted frontiers are discarded and their time is charged.',
            'Budget readouts exclude commits finalized after the budget; soft guards are not hard memory guarantees.',
            'Local generation restarts at epoch boundaries; expanded certificate heights never reset.',
            'No claim that a finite biased trajectory covers the entire proof space.'])
