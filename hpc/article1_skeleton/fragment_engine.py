"""Versioned persistent forward engine; frozen historical runners unchanged.

First-proof provenance is explicit in all new arms. A finite active search
skeleton is distinct from the unbounded-by-theory, resource-capped proof archive.
Retiring a search premise never erases its certificate. Reactivation does not
change a proof's derivation depth or relabel a lemma as an original axiom.
"""
from collections import Counter
import hashlib
import sys

from certificate import encode_formula, canonical
from direct_engine import StopWindow
from mechanisms import pruning_class
from skeleton_local import ROOT
from worker import rss


def make_fragment_generator(archive, seeds, cfg):
    sys.path.insert(0, str(ROOT/'src')) if str(ROOT/'src') not in sys.path else None
    from hilbert.schema_generator import SchemaGenerator
    from hilbert.utility import UtilityRanker, UtilityWeights
    legacy = cfg['arm'] == 'legacy_delta'
    base = pruning_class(SchemaGenerator) if legacy else SchemaGenerator

    class Engine(base):
        def _init_axioms(self):
            super()._init_axioms()
            for i in seeds:
                from skeleton_local import decode
                self._add_schema(decode(archive.records[i][1]), 0, 'epoch_seed', [])
            self.original_axioms = {e.id for e in self.entry_log
                                    if e.rule != 'epoch_seed' and not e.premise_ids}
            self.live = {e.id for bucket in self._by_depth.values() for e in bucket}
            for i in self.live:
                self.activated_at[i] = 0

        def _add_schema(self, formula, depth, rule, premise_ids, **kwargs):
            added = super()._add_schema(formula, depth, rule, premise_ids, **kwargs)
            if added:
                e = self._id_to_entry[self._next_id-1]
                self.entry_log.append(e)
                if e.id in self._passive_pool: self.queued_at[e.id] = self.current_depth
                for p in set(premise_ids):
                    self.successful_uses[p] += 1
                    self.last_use[p] = self.current_depth
            return added

        def _on_rediscovery_event(self, existing, premise_ids):
            super()._on_rediscovery_event(existing, premise_ids)
            if (not legacy and existing.id not in self.live and
                    existing.id not in self._passive_pool and
                    self.current_depth-self.retired_at.get(existing.id, -10**9) >= cfg['reactivation_cooldown'] and
                    self.requeued_this_layer < cfg['reactivation_slots']):
                self._passive_pool[existing.id] = existing
                self.queued_at[existing.id] = self.current_depth
                self.requeued_this_layer += 1
                self.lifecycle['rediscovery_requeues'] += 1
                self.events.append(dict(kind='rediscovery_requeue', id=existing.id,
                                        round=self.current_depth))

        def _beam_tiebreak_key(self, entry):
            return hashlib.sha256(f'{cfg["seed"]}:{self.current_depth}:{entry.schema_hash}'.encode()).digest()

        def _generate_depth(self, depth):
            if rss() > cfg['soft_memory_bytes']: raise StopWindow('soft_memory')
            self.current_depth = depth; self.requeued_this_layer = 0
            result = super()._generate_depth(depth)
            self.pending_metrics = dict(result[1], depth=depth)
            return result

        def _drop_live(self, ids, reason):
            ids = set(ids)-self.original_axioms
            if not ids: return
            self.live.difference_update(ids); self._activated_ids.difference_update(ids)
            for depth in self._by_depth:
                self._by_depth[depth] = [e for e in self._by_depth[depth] if e.id not in ids]
            self._all_implication_candidates = [c for c in self._all_implication_candidates if c.entry.id not in ids]
            self._wildcard_implication_candidates = [c for c in self._wildcard_implication_candidates if c.entry.id not in ids]
            for tag in self._implication_candidates_by_tag:
                self._implication_candidates_by_tag[tag] = [c for c in self._implication_candidates_by_tag[tag] if c.entry.id not in ids]
            for i in sorted(ids):
                self._implication_candidate_by_id.pop(i, None)
                self.retired_at[i] = self.current_depth
                self.events.append(dict(kind=reason, id=i, round=self.current_depth))
            self.lifecycle[reason] += len(ids)

        def _activate_from_pool(self, depth):
            if legacy:
                n = super()._activate_from_pool(depth)
                self.live = {e.id for bucket in self._by_depth.values() for e in bucket}
            else:
                # The lease is based on search activation or genuine new uses,
                # not on certificate depth (imported depth-zero lemmas expire).
                expired = [i for i in self.live-self.original_axioms
                    if depth-max(self.activated_at.get(i, 0), self.last_use.get(i, 0)) >= cfg['active_lease']
                    and depth >= self.library_until.get(i, 0)]
                self._drop_live(expired, 'active_lease_expired')
                stale = [i for i in self._passive_pool
                         if depth-self.queued_at.get(i, depth) >= cfg['passive_lease']]
                for i in stale:
                    self._passive_pool.pop(i); self.retired_at[i] = depth
                    self.events.append(dict(kind='passive_lease_expired', id=i, round=depth))
                self.lifecycle['passive_lease_expired'] += len(stale)
                # Proof fragments occupy ordinary beam slots; they never add
                # an unreported auxiliary search basis.
                forced = []
                for i in sorted(self.library):
                    if (i not in self.live and depth < self.library_until.get(i, 0) and
                            depth-self.retired_at.get(i, -10**9) >= cfg['reactivation_cooldown']):
                        forced.append(i)
                forced = forced[:min(cfg['fragment_slots'], self.beam_width)]
                before = set(self.live)
                for i in forced:
                    e = self._id_to_entry[i]
                    self._passive_pool.pop(i, None)
                    self._activated_ids.add(i); self._by_depth.setdefault(depth, []).append(e)
                    self._register_entry(e)
                    self.lifecycle['fragment_reactivations'] += 1
                    self.events.append(dict(kind='fragment_reactivation', id=i, round=depth))
                width = self.beam_width
                try:
                    self.beam_width = width-len(forced)
                    n = super()._activate_from_pool(depth) if self.beam_width else 0
                finally:
                    self.beam_width = width
                new = {e.id for e in self._by_depth.get(depth, [])}
                self.live |= new
                for i in new-before:
                    self.activated_at[i] = depth
                    if i in self.retired_at: self.lifecycle['actual_reactivations'] += 1
                excess = len(self.live-self.original_axioms)-cfg['active_cap']
                if excess > 0:
                    # Keep the fresh frontier first, then a leased library and
                    # recently useful premises. Deterministic bound, no erasure.
                    victims = sorted(self.live-self.original_axioms,
                        key=lambda i: (i in new, depth < self.library_until.get(i, 0),
                                       self.last_use.get(i, -1), self.successful_uses[i], i))[:excess]
                    self._drop_live(victims, 'active_cap_retired')
                excess = len(self._passive_pool)-cfg['passive_cap']
                if excess > 0:
                    victims = sorted(self._passive_pool,
                        key=lambda i: (self._rediscovery_counts.get(i, 0), -self._id_to_entry[i].formula.size(), i))[:excess]
                    for i in victims:
                        self._passive_pool.pop(i); self.retired_at[i] = depth
                        self.events.append(dict(kind='passive_cap_retired', id=i, round=depth))
                    self.lifecycle['passive_cap_retired'] += len(victims)
                n += len(forced)
            # Empty buckets preserve the epoch clock after retirement.
            self._by_depth.setdefault(depth, [])
            if len(self._entries) < cfg['max_nodes']:
                self.snapshot = dict(end=len(self.entry_log), depth=depth,
                    active_ids=sorted(self.live), passive_ids=sorted(self._passive_pool),
                    indexed_major_ids=sorted(self._implication_candidate_by_id),
                    metrics=dict(self.pending_metrics), lifecycle=dict(self.lifecycle),
                    events=list(self.events),
                    size_exclusions=self.stats.get('mp_filtered_size', 0))
                self.events = []
            return n

        def set_library(self, ids):
            old = self.library
            for i in ids:
                if i not in old or self.last_use.get(i, -1) > self.library_selected_at.get(i, -1):
                    self.library_until[i] = self.current_depth+cfg['library_lease']
                self.library_selected_at[i] = self.current_depth
            self.library = set(ids)
            for i in old-set(ids): self.library_until.pop(i, None)

    weights = UtilityWeights(resonance_exact=0., resonance_shape=0., subsume=0.,
        goal_step=0., relevance=0., importance=0., rediscovery=1., size=1.)
    extra = dict(prune_mode='both', prune_patience=2) if legacy else {}
    g = Engine(propositional_base='metamath_prop' if archive.substrate == 'prop' else 'modal_k',
        theory=None, max_depth=1, max_schemas=cfg['max_nodes'],
        max_formula_size=cfg['max_formula_size'], n_workers=1, beam_width=cfg['beam'],
        utility_ranker=UtilityRanker([], weights), provenance_mode='first',
        subsumption_mode='off', _use_env_overrides=False, **extra)
    g.entry_log = []; g.current_depth = 0; g.snapshot = None; g.pending_metrics = {}
    g.live = set(); g.original_axioms = set(); g.activated_at = {}; g.queued_at = {}
    g.retired_at = {}; g.last_use = {}; g.successful_uses = Counter()
    g.lifecycle = Counter(); g.events = []; g.requeued_this_layer = 0
    g.library = set(); g.library_until = {}; g.library_selected_at = {}
    return g
