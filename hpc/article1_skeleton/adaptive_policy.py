"""Frozen candidate portfolio with immediate or delayed observational credit.

Delayed first-proof reuse is not a causal effect. The separate lemma-availability
interventions test that distinction. No future label enters an earlier choice.
"""
from collections import Counter
import math
import random


ACTIONS=(('all',1.,0),('learned',.5,0),('learned',.5,128),('learned',.25,128))
POLICIES=('all','half','half_replay','quarter_replay','random_switch','immediate','delayed')
FIXED={'all':0,'half':1,'half_replay':2,'quarter_replay':3}


class Portfolio:
    def __init__(self,policy,seed,block=8,delay=32,alpha=.35,epsilon=.125):
        if policy not in POLICIES or block<1 or delay<1:raise ValueError('Invalid portfolio')
        self.policy=policy;self.rng=random.Random(seed+712971);self.block=block;self.delay=delay
        self.alpha=alpha;self.epsilon=epsilon;self.visits=[0]*4;self.updates=[0]*4
        self.values=[0.]*4;self.observations=[];self.feedback=[];self.current=None
        self.last_action=0;self.decisions=[]

    def before(self,epoch,work):
        for obs in self.observations:
            if obs['closed'] or obs['end'] is None:continue
            ready=obs['end']+(self.delay if self.policy=='delayed' else 0)
            if epoch<=ready:continue
            gain=(.2*obs['new']+obs['reuse']) if self.policy=='delayed' else obs['new']
            reward=gain/(gain+max(1,obs['work'])/1000)
            a=obs['action'];self.values[a]=.75*self.values[a]+.25*reward
            self.updates[a]+=1;obs['closed']=True
            self.feedback.append(dict(window=obs['id'],closed_epoch=epoch,action=a,reward=reward,
                new=obs['new'],future_reuse_credit=obs['reuse'],work=obs['work']))
        if epoch<=2:return 0
        if self.current is None or epoch-self.current['start']>=self.block:
            if self.policy in FIXED:a=FIXED[self.policy]
            elif self.policy=='random_switch':a=self.rng.randrange(4)
            elif 0 in self.visits:a=self.visits.index(0)
            elif self.rng.random()<self.epsilon:a=self.rng.randrange(4)
            else:
                t=1+sum(self.visits)
                a=max(range(4),key=lambda j:(self.values[j]+self.alpha*math.sqrt(math.log(t)/(1+self.visits[j])),-j))
            obs=dict(id=len(self.observations),action=a,start=epoch,end=None,initial_work=work,
                     work=0,new=0,reuse=0.,closed=False)
            self.observations.append(obs);self.current=obs;self.visits[a]+=1;self.last_action=a
            self.decisions.append(dict(epoch=epoch,action=a,visits=list(self.visits),
                completed_updates=list(self.updates),values=list(self.values)))
        return self.last_action

    def after(self,epoch,work):
        if self.current is not None and epoch-self.current['start']+1>=self.block:
            self.current['end']=epoch
            self.current['work']=max(0,work-self.current['initial_work'])

    def created(self):
        if self.current is None:return None
        self.current['new']+=1
        return self.current['id']

    def reuse(self,owner,created_epoch,epoch,proof_work):
        if owner is None or epoch<=created_epoch or epoch-created_epoch>self.delay:return
        obs=self.observations[owner]
        if not obs['closed']:obs['reuse']+=min(8.,math.log2(1+proof_work))

    def summary(self):
        return dict(policy=self.policy,action=self.last_action,visits=list(self.visits),
            updates=list(self.updates),values=list(self.values),decisions=len(self.decisions),
            pending=sum(not x['closed'] for x in self.observations),feedback=self.feedback[-4:])

    def export(self):
        return dict(summary=self.summary(),decisions=self.decisions,observations=self.observations,
                    feedback=self.feedback,credit='Observed later use of a closed first proof; not a causal label.')


def make_adaptive_engine(archive,cfg,model,meter):
    from deferred_gate import OpportunityQueue
    from skeleton_gate import make_gated_engine
    runtime=dict(cfg,defer_skipped=True)
    g=make_gated_engine(archive,runtime,model);base=type(g)

    class Adaptive(base):
        def _add_schema(self,formula,depth,rule,premise_ids,**kwargs):
            added=super()._add_schema(formula,depth,rule,premise_ids,**kwargs)
            if added:
                e=self._id_to_entry[self._next_id-1]
                self.proof_work[e.id]=min(10**6,1+sum(self.proof_work.get(p,1) for p in premise_ids))
                if premise_ids:
                    for p in set(premise_ids):
                        if p in self.ownership:
                            owner,when=self.ownership[p]
                            self.portfolio.reuse(owner,when,depth,self.proof_work[p])
                    self.ownership[e.id]=(self.portfolio.created(),depth)
            return added

        def queue_skipped(self,minor,candidates,selected):
            for j,c in enumerate(candidates):
                if j not in selected:self.opportunities.add((minor,c.entry.id))

        def _generate_depth(self,depth):
            a=self.portfolio.before(depth,meter.counts['unify_visits'])
            gate,fraction,quota=ACTIONS[a]
            runtime.update(gate=gate,gate_fraction=fraction,defer_skipped=quota>0)
            self.quota=quota;self.selected_action=a
            new,metrics=super()._generate_depth(depth)
            import hilbert.schema_generator as sg
            fresh=attempts=0
            for minor,major in self.opportunities.pop_ready(self.live,quota,cfg['replay_scan']):
                if len(self._entries)>=self._schema_cap:break
                result=sg.schema_mp(self._id_to_entry[minor].formula,self._id_to_entry[major].formula,
                    inputs_canonical=True,canonicalize_output=False)
                attempts+=1
                if result is not None:
                    self.stats['mp_successes']+=1
                    if self._add_schema(result,depth,'MP',[minor,major]):fresh+=1
            self.replay_counters['attempts']+=attempts;self.replay_counters['new_formulae']+=fresh
            metrics.update(replay_attempts=attempts,replay_new_formulae=fresh,action=a)
            self.pending_metrics=dict(metrics,depth=depth)
            return new+fresh,metrics

        def _activate_from_pool(self,depth):
            reserved=self.opportunities.reservations(self.live,self.retired_at,depth,
                cfg['reactivation_cooldown'],cfg['replay_parent_slots']) if self.quota else []
            self.library=set(reserved);self.library_until={i:depth+2 for i in reserved}
            self.replay_counters['parent_reservations']+=len(reserved)
            result=super()._activate_from_pool(depth)
            self.portfolio.after(depth,meter.counts['unify_visits'])
            if self.snapshot is not None:
                self.snapshot.update(deferred=self.opportunities.summary(False),
                    replay_counters=dict(self.replay_counters),replay_parent_reservations=reserved)
            return result

    g.__class__=Adaptive
    g.portfolio=Portfolio(cfg['adaptive_policy'],cfg['seed'],cfg['decision_block'],cfg['credit_delay'])
    g.opportunities=OpportunityQueue(cfg['deferred_cap']);g.replay_counters=Counter()
    g.proof_work={};g.ownership={};g.quota=0;g.selected_action=0
    return g
