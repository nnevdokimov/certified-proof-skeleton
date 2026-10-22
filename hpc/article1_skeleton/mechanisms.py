"""Frozen follow-up mechanism panel, not the final held-out Metamath experiment."""
import random


ARMS = (
    ('neutral_none', False, False, 'none', 2),
    ('size_none', False, True, 'none', 2),
    ('rediscovery_none', True, False, 'none', 2),
    ('utility_none', True, True, 'none', 2),
    ('neutral_both2', False, False, 'both', 2),
    ('utility_pool2', True, True, 'pool', 2),
    ('utility_major2', True, True, 'major', 2),
    ('utility_both2', True, True, 'both', 2),
    ('utility_both4', True, True, 'both', 4),
)


def protocol(run_id):
    common = dict(beam=64, max_depth=128, max_schemas=500000,
                  soft_memory_bytes=4*1024**3, independent_check=True,
                  extended_readout=True)
    tasks = []
    for panel, seeds, arms, budgets in (
        ('mechanisms', range(101, 111), ARMS, [10, 30, 120]),
        ('scale', range(201, 206), tuple(a for a in ARMS if a[0] in
            ('neutral_none', 'utility_none', 'neutral_both2', 'utility_both2')), [30, 120, 600]),
    ):
        for substrate in ('prop', 'modal'):
            for seed in seeds:
                for name, rediscovery, size, mode, patience in arms:
                    tasks.append(dict(common, panel=panel, arm=name, substrate=substrate,
                        seed=seed, utility=bool(rediscovery and size), prune=mode != 'none',
                        rediscovery_weight=float(rediscovery), size_weight=float(size),
                        prune_mode=mode, prune_patience=patience,
                        cpu_budgets=budgets, wall_limit_sec=max(budgets)*2+120))
    random.Random(20261004).shuffle(tasks)
    smoke = dict(tasks[0], substrate='prop', panel='smoke', arm='utility_both2',
                 utility=True, prune=True, rediscovery_weight=1., size_weight=1.,
                 prune_mode='both', prune_patience=2, cpu_budgets=[3], max_depth=4,
                 max_schemas=10000, wall_limit_sec=90)
    return dict(schema='article1-mechanisms/v1', suite='mechanisms', run_id=run_id,
        tasks=tasks, smoke=smoke, smoke_modal=dict(smoke, substrate='modal'),
        shuffle_seed=20261004, policy='75% score selection / 25% age; target-free',
        primary_readout='Per substrate/panel: independently checked diagnostic-target hits at final construction budget. Pair by seed; formulas and generation depth are secondary. Compare utility_both2 against size_none and utility_none in mechanism panel; no theorem-level independence.',
        timing='Generation budget includes candidate work, selection, pruning, diagnostics and checkpoints. Export, independent replay and startup are separate and included in whole-worker CPU. These are not total-budget solve curves.',
        analysis='Report every cell including failed/capped/timeout. Separate substrates and panels. Paired seed-level differences and intervals; no pooled vertex-level tests. Adjust exploratory multiple comparisons; no parameter choice from scale results claimed as held-out.',
        readout='Historical 14/4 lists plus deep_prop and nested modal necessity diagnostics. Public, not independent held-out Metamath targets. No target is supplied to the ranker.',
        limits=['This closes a mechanism follow-up, NOT the complete article experiment plan.',
                'No certified inter-epoch transfer, hidden Metamath benchmark, or AI calls.',
                'Replay checks first derivations only, not full alternative provenance or completeness.',
                'Age floor does not guarantee completeness under finite budgets and pruning.',
                'Soft RSS guard is checked at generation boundaries; not a Slurm memory request or hard cap.',
                'No automatic claim of speedup, scale-free structure, or absence of target leakage.',
                'Expected generation allocation: 12.67 CPU hours; finalization and checks add time.'])


def pruning_class(base):
    class SelectivePruning(base):
        def __init__(self, *, prune_mode, prune_patience, **kw):
            self.prune_mode = prune_mode
            self.prune_patience = prune_patience
            self._activation_round = {}
            self._pruned = set()
            self.pruned_pool = self.pruned_major = 0
            self.prune_diagnostics = {}
            super().__init__(**kw)

        def _activate_from_pool(self, depth):
            self._prune_rot(depth)
            before = set(self._activated_ids)
            n = super()._activate_from_pool(depth)
            for node_id in self._activated_ids - before:
                self._activation_round[node_id] = depth
            return n

        def _prune_rot(self, depth):
            reuse = {}
            for entry in self._entries.values():
                for p in entry.premise_ids:
                    reuse[p] = reuse.get(p, 0) + 1
            rc = self._rediscovery_counts
            diag = dict(indexed=0, nonaxiom=0, old_enough=0, zero_reuse=0, eligible=0)
            dead = []
            for nid, cand in self._implication_candidate_by_id.items():
                diag['indexed'] += 1
                if cand.entry.depth == 0: continue
                diag['nonaxiom'] += 1
                birth = self._activation_round.get(nid)
                if birth is None or depth-birth < self.prune_patience: continue
                diag['old_enough'] += 1
                if reuse.get(nid, 0): continue
                diag['zero_reuse'] += 1
                if rc.get(nid, 0): continue
                diag['eligible'] += 1
                dead.append(nid)
            self.prune_diagnostics = diag
            if dead and self.prune_mode in ('major', 'both'):
                ds = set(dead)
                self._all_implication_candidates = [c for c in self._all_implication_candidates if c.entry.id not in ds]
                self._wildcard_implication_candidates = [c for c in self._wildcard_implication_candidates if c.entry.id not in ds]
                for tag, values in list(self._implication_candidates_by_tag.items()):
                    self._implication_candidates_by_tag[tag] = [c for c in values if c.entry.id not in ds]
                for nid in dead: self._implication_candidate_by_id.pop(nid)
                self._pruned.update(ds)
                self.pruned_major += len(dead)
            if self.prune_mode in ('pool', 'both'):
                rot = [nid for nid, e in self._passive_pool.items()
                       if depth-e.depth >= self.prune_patience and not rc.get(nid, 0)]
                for nid in rot: self._passive_pool.pop(nid)
                self._pruned.update(rot)
                self.pruned_pool += len(rot)
    return SelectivePruning


def diagnostic_targets(substrate, historical, watchlists):
    targets = list(historical)
    if substrate == 'prop':
        return targets + watchlists('deep_prop')
    from core.formula import Box, Implies, PropVar
    p, q = PropVar('p'), PropVar('q')
    for label, formula in [('id', Implies(p, p)), ('ax1', Implies(p, Implies(q, p)))]:
        for depth in range(1, 9):
            formula = Box(formula)
            if (label, depth) in {('id', 1), ('id', 2), ('ax1', 1)}: continue
            targets.append((f'box{depth}-{label}', formula))
    return targets
