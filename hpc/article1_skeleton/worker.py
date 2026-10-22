"""Target-free generation pilot. Not a certified cross-epoch experiment.

CPU-budget readouts use only completed generation checkpoints. Work after a
checkpoint is charged but is never credited as an extra result. Archive and
verification costs are reported separately and included in whole-worker costs.
"""
import argparse
import contextlib
import gc
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    temp.replace(path)

def verify_bundle(root):
    files=json.loads((root/'source_manifest.json').read_text())
    for name,digest in files.items():
        if sha(root/name)!=digest: raise ValueError(f'Source hash mismatch: {name}')
    return sha(root/'source_manifest.json')

def tie_key(seed, depth, formula_hash):
    # A shared counter-based perturbation across arms; no call-order RNG drift.
    return hashlib.sha256(f'{seed}:{depth}:{formula_hash}'.encode()).digest()

def budget_rows(rows,budgets):
    return [dict(cpu_budget_sec=b, checkpoint=next((r for r in reversed(rows) if r['cpu_checkpoint_sec']<=b),None)) for b in budgets]

def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024

class BudgetStop(Exception): pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-modal',action='store_true')
    a=p.parse_args()
    total_cpu=time.process_time(); total_wall=time.perf_counter()
    fingerprint=verify_bundle(ROOT)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    cfg=dict(protocol['smoke_modal'] if a.smoke_modal else protocol['smoke'] if a.smoke else protocol['tasks'][a.index])
    tag='smoke_modal' if a.smoke_modal else 'smoke' if a.smoke else f"cell_{a.index:03d}"
    out=ROOT/'results'/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise RuntimeError(f'Result already exists: {out}')
    sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('pilot_prune',ROOT/'experiments/wild_growth/prune.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    from hilbert.schema_generator import SchemaGenerator
    from hilbert.utility import UtilityRanker,UtilityWeights
    from hilbert.watchlists import get_watchlist
    from core.schema import canonicalize_with_hash
    checkpoint=((),[]); current_depth=0; depth_metrics={}; selection_sec=0.; pruning_sec=0.
    base_cls=mod.PruningGenerator if cfg['prune'] else SchemaGenerator
    extra={}
    if 'prune_mode' in cfg:
        from mechanisms import pruning_class
        base_cls=pruning_class(SchemaGenerator)
        extra=dict(prune_mode=cfg['prune_mode'],prune_patience=cfg['prune_patience'])

    class Instrumented(base_cls):
        def _generate_depth(self,depth):
            nonlocal current_depth,depth_metrics
            if checkpoint[1] and time.process_time()-start_cpu>=max(cfg['cpu_budgets']): raise BudgetStop('cpu_budget')
            if rss()>cfg['soft_memory_bytes']: raise BudgetStop('soft_memory')
            current_depth=depth
            count,depth_metrics=super()._generate_depth(depth)
            return count,depth_metrics
        def _beam_tiebreak_key(self,entry):
            return tie_key(cfg['seed'],current_depth,entry.schema_hash)
        def _prune_rot(self,depth):
            nonlocal pruning_sec
            t=time.process_time()
            super()._prune_rot(depth)
            pruning_sec+=time.process_time()-t
        def _activate_from_pool(self,depth):
            nonlocal checkpoint,selection_sec
            rows=checkpoint[1]
            t=time.process_time(); n=super()._activate_from_pool(depth)
            selection_sec+=time.process_time()-t
            entries=tuple(self._entries.values())
            ids={e.id for e in entries}
            if any(pid not in ids for e in entries for pid in e.premise_ids):
                raise RuntimeError('Missing parent in first-derivation archive')
            hashes=sorted(e.schema_hash for e in entries)
            # Post-hoc readout only; targets never supplied to the generator.
            found=set(hashes)
            row=dict(depth=depth,formulas=len(entries),active=len(self._activated_ids),
                passive=len(self._passive_pool),posthoc_hits=[k for k,h in targets if h in found],
                formula_set_sha256=hashlib.sha256('\n'.join(hashes).encode()).hexdigest(),
                pruned_pool=getattr(self,'pruned_pool',0),pruned_major=getattr(self,'pruned_major',0),
                selection_including_pruning_cpu_sec=selection_sec,pruning_cpu_sec=pruning_sec,
                generation_metrics=dict(depth_metrics),peak_rss_bytes=rss())
            if 'prune_mode' in cfg:
                row.update(prune_diagnostics=dict(self.prune_diagnostics),
                    indexed_majors=len(self._implication_candidate_by_id),
                    cumulative_generation_counters={k:v for k,v in self.stats.items()
                        if k.startswith(('mp_', 'bulk_', 'nec_')) and isinstance(v,(int,float))})
            # Commit in memory and atomically on disk. The checkpoint records
            # only whole generations, never a state interrupted by the timer.
            row['cpu_checkpoint_sec']=time.process_time()-start_cpu
            row['wall_checkpoint_sec']=time.perf_counter()-start_wall
            write_json(out/'progress.json',dict(config=cfg,rows=rows+[row]))
            row['cpu_checkpoint_sec']=time.process_time()-start_cpu
            row['wall_checkpoint_sec']=time.perf_counter()-start_wall
            checkpoint=(entries,rows+[row])
            return n

    base,wlname=mod.SUBSTRATES[cfg['substrate']]
    target_formulas=get_watchlist(wlname)
    if cfg.get('extended_readout'):
        from mechanisms import diagnostic_targets
        target_formulas=diagnostic_targets(cfg['substrate'],target_formulas,get_watchlist)
    targets=[(k,canonicalize_with_hash(f)[1]) for k,f in target_formulas]
    zero=dict(resonance_exact=0.,resonance_shape=0.,subsume=0.,goal_step=0.,relevance=0.,importance=0.)
    weights=UtilityWeights(**zero,rediscovery=cfg.get('rediscovery_weight',float(cfg['utility'])),
        size=cfg.get('size_weight',float(cfg['utility'])))
    start_cpu=time.process_time();start_wall=time.perf_counter()
    def stop(signum,frame): raise BudgetStop('cpu_timer' if signum==signal.SIGPROF else 'wall_timer')
    signal.signal(signal.SIGPROF,stop);signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_PROF,max(cfg['cpu_budgets']))
    signal.setitimer(signal.ITIMER_REAL,cfg['wall_limit_sec'])
    reason='unknown';g=None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            g=Instrumented(propositional_base=base,theory=None,max_depth=cfg['max_depth'],
                max_schemas=cfg['max_schemas'],max_formula_size=200,n_workers=1,beam_width=cfg['beam'],
                utility_ranker=UtilityRanker([],weights),provenance_mode='full',
                subsumption_mode='off',_use_env_overrides=False,**extra)
            g.generate()
        reason=g.stats.get('stopped_reason','finished')
    except BudgetStop as exc:
        reason=str(exc)
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);signal.setitimer(signal.ITIMER_REAL,0)
    generation_cpu=time.process_time()-start_cpu
    committed,rows=checkpoint
    # First derivations only; not all alternative hyperedges, not an independently
    # checked logical certificate. Self-produced pickle is bound to source hashes.
    archive_start=time.process_time()
    archive=out/'first_derivations.pkl.gz'
    with gzip.open(archive,'wb',compresslevel=1) as f:
        pickle.dump(dict(config=cfg,source_manifest_sha256=fingerprint,
            entries=[(e.id,e.formula,e.depth,e.rule,tuple(e.premise_ids)) for e in committed]),f,protocol=4)
    archive_cpu=time.process_time()-archive_start
    verification=None
    if cfg.get('independent_check'):
        from certificate import Checker,encode_formula
        # The checker stores only formulas/first proofs, never the full generator graph.
        cleanup_start=time.process_time()
        del g
        gc.collect()
        cleanup_cpu=time.process_time()-cleanup_start
        check_start=time.process_time()
        checker=Checker(cfg['substrate'])
        cert=out/'certificate.jsonl.gz'
        with gzip.open(cert,'wt',encoding='utf-8',compresslevel=1) as f:
            for e in committed:
                formula=encode_formula(e.formula)
                checker.accept(e.id,formula,e.rule,list(e.premise_ids))
                f.write(json.dumps([e.id,formula,e.rule,list(e.premise_ids)],separators=(',',':'))+'\n')
        verification=dict(checker.summary(),check_and_export_cpu_sec=time.process_time()-check_start,
            generator_cleanup_cpu_sec=cleanup_cpu,certificate_sha256=sha(cert),
            certificate_bytes=cert.stat().st_size,
            checker_sha256=sha(Path(__file__).with_name('certificate.py')))
        write_json(out/'verification.json',verification)
    archive_digest=sha(archive)
    package_versions={name:importlib.metadata.version(name) for name in ('numpy','networkx')}
    report=dict(schema='skeleton-pilot-cell/v1',config=cfg,status='completed',stop_reason=reason,
        checkpoints=rows,budget_readouts=budget_rows(rows,cfg['cpu_budgets']),
        generation_cpu_sec=generation_cpu,archive_cpu_sec=archive_cpu,
        verification=verification,target_readout_labels=[k for k,h in targets],
        whole_worker_cpu_sec=time.process_time()-total_cpu,whole_worker_wall_sec=time.perf_counter()-total_wall,
        peak_rss_bytes=rss(),archive_bytes=archive.stat().st_size,archive_sha256=archive_digest,
        source_manifest_sha256=fingerprint,python=sys.version,host=os.uname().nodename,
        package_versions=package_versions,python_executable=sys.executable,
        slurm_account=os.environ.get('SLURM_JOB_ACCOUNT'),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        limits=protocol['limits'])
    if not rows: raise RuntimeError('No whole generation completed; calibration needed')
    write_json(out/'result.json',report)
    print(json.dumps(dict(cell=tag,stop_reason=reason,completed_depth=rows[-1]['depth'],formulas=rows[-1]['formulas'])))

if __name__=='__main__': main()
