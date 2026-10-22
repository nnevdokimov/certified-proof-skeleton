"""Bounded FIFO of unevaluated MP opportunities, with explicit premise slots.

Only active certified premises may participate in replay. Retired premises can
return through the existing bounded activation mechanism, consuming ordinary
beam slots. Queue overflow remains explicit; no completeness claim follows.
"""
from collections import Counter, deque
import hashlib
from itertools import islice
import time

from skeleton_gate import make_gated_engine


class OpportunityQueue:
    def __init__(self, cap):
        if cap<1:raise ValueError('Positive queue cap required')
        self.cap=cap;self.pairs=deque();self.known=set();self.counters=Counter()
        self.done=deque();self.done_set=set();self.history_cap=cap*4

    def add(self, pair):
        pair=tuple(pair)
        if pair in self.done_set:
            self.counters['replayed_offers_ignored']+=1;return
        if pair in self.known:
            self.counters['duplicate_offers']+=1;return
        if len(self.pairs)>=self.cap:
            self.counters['overflow_rejected']+=1;return
        self.pairs.append(pair);self.known.add(pair);self.counters['enqueued']+=1

    def pop_ready(self, active, quota, scan):
        ready=[]
        for _ in range(min(scan,len(self.pairs))):
            if len(ready)>=quota:break
            pair=self.pairs.popleft()
            if all(i in active for i in pair):
                self.known.remove(pair);ready.append(pair)
                self.done.append(pair);self.done_set.add(pair)
                if len(self.done)>self.history_cap:self.done_set.remove(self.done.popleft())
            else:
                self.pairs.append(pair);self.counters['blocked_rotations']+=1
        self.counters['dequeued_for_replay']+=len(ready)
        return ready

    def reservations(self, active, retired, depth, cooldown, slots):
        chosen=[]
        for pair in islice(self.pairs,256):
            missing=[i for i in pair if i not in active and i not in chosen]
            if not missing:continue
            if any(depth-retired.get(i,-10**9)<cooldown for i in missing):continue
            additions=sorted(set(pair)-set(chosen))
            if len(chosen)+len(additions)>slots:continue
            chosen.extend(additions)
            if len(chosen)==slots:break
        return chosen

    def summary(self,with_hash=True):
        h=hashlib.sha256()
        if with_hash:
            for a,b in self.pairs:h.update(f'{a}:{b};'.encode())
        return dict(pending=len(self.pairs),cap=self.cap,counters=dict(self.counters),
                    history=len(self.done),history_cap=self.history_cap,
                    queue_sha256=h.hexdigest() if with_hash else None,first_pairs=list(islice(self.pairs,8)))


def make_deferred_engine(archive,cfg,model):
    g=make_gated_engine(archive,cfg,model)
    if not cfg.get('defer_skipped',False):return g
    base=type(g)

    class Deferred(base):
        def queue_skipped(self,minor,candidates,selected):
            for j,candidate in enumerate(candidates):
                if j not in selected:self.opportunities.add((minor,candidate.entry.id))

        def _generate_depth(self,depth):
            new,metrics=super()._generate_depth(depth)
            import hilbert.schema_generator as sg
            begin=time.process_time();replayed=0;fresh=0
            ready=self.opportunities.pop_ready(self.live,cfg['replay_quota'],cfg['replay_scan'])
            for minor,major in ready:
                if len(self._entries)>=self._schema_cap:
                    # The node cap terminates this trajectory; no resume or
                    # completeness claim follows from the pending queue.
                    self.replay_counters['node_cap_unexecuted']+=1;continue
                assert minor in self.live and major in self.live
                result=sg.schema_mp(self._id_to_entry[minor].formula,self._id_to_entry[major].formula,
                    inputs_canonical=True,canonicalize_output=False)
                replayed+=1
                if result is not None:
                    self.stats['mp_successes']+=1
                    if self._add_schema(result,depth,'MP',[minor,major]):fresh+=1
            self.replay_counters['attempts']+=replayed;self.replay_counters['new_formulae']+=fresh
            self.replay_cpu_sec+=time.process_time()-begin
            metrics.update(replay_attempts=replayed,replay_new_formulae=fresh,
                           replay_pending=len(self.opportunities.pairs))
            self.pending_metrics=dict(metrics,depth=depth)
            return new+fresh,metrics

        def _activate_from_pool(self,depth):
            reserved=self.opportunities.reservations(self.live,self.retired_at,depth,
                cfg['reactivation_cooldown'],cfg['replay_parent_slots'])
            self.library=set(reserved)
            self.library_until={i:depth+2 for i in reserved}
            self.replay_counters['parent_reservations']+=len(reserved)
            result=super()._activate_from_pool(depth)
            if self.snapshot is not None:
                self.snapshot['deferred']=self.opportunities.summary(False)
                self.snapshot['replay_counters']=dict(self.replay_counters)
                self.snapshot['replay_parent_reservations']=reserved
            return result

    g.__class__=Deferred
    g.opportunities=OpportunityQueue(cfg['deferred_cap'])
    g.replay_counters=Counter();g.replay_cpu_sec=0.
    return g
