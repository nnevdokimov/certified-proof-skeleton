"""Direct, proof-closed growth with pre-inference control and episodic RL.

Every admitted statement stays in the output. There is no larger theorem
archive followed by core extraction. Candidate *parent pairs* form a bounded
cheap pool; only the selected pair reaches the typed prover. Rejected inferred
conclusions still cost work and enter counters. This is not an oracle that
generates only useful lemmas, nor an asymptotic scale-free guarantee.

The policy chooses a sampling mechanism, not a theorem target. Its utility
reward measures observational reuse and fixed-policy continuation, not a
counterfactual saving in proof search. A separate optional topology reward
explicitly steers a finite degree profile; that track cannot establish
intrinsic/emergent topology.
"""
from collections import Counter
from dataclasses import dataclass, asdict
import math
import time

import numpy as np
from scipy.special import zeta

from .growth import family
from .index import PatternIndex
from .proofs import Library
from .saturation import instance_witness, statement
from .terms import size, variables, variable

MODES = ('uniform', 'reuse', 'bridge', 'contract', 'frontier')
FEATURES = 8


@dataclass(frozen=True)
class OnlineConfig:
    attempts: int = 2000
    max_nodes: int = 2000
    formula_size: int = 160
    proof_height: int = 64
    pair_pool: int = 12
    block: int = 16
    gen_probability: float = .08
    subsumption: bool = True
    target_alpha: float = 2.5
    target_xmin: int = 4
    topology_weight: float = 0.

    def validate(self):
        for name in ('attempts', 'max_nodes', 'formula_size', 'proof_height',
                     'pair_pool', 'block', 'target_xmin'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Positive integer required: ' + name)
        if (not 0 <= self.gen_probability <= 1 or self.target_alpha <= 1
                or self.topology_weight < 0 or not all(math.isfinite(x) for x in
                (self.gen_probability, self.target_alpha, self.topology_weight))):
            raise ValueError('Invalid policy bounds')


def profile_loss(degrees, *, alpha=2.5, xmin=4):
    """Fixed-grid conditional survival discrepancy; NOT a power-law GOF test.

    A minimum-tail penalty prevents a single hub from satisfying the objective.
    Neither an estimated exponent nor a bootstrap p enters training.
    """
    d = np.asarray(degrees, dtype=int)
    tail = d[d >= xmin]
    grid = xmin * np.array([1, 2, 4, 8, 16])
    desired = zeta(alpha, grid) / zeta(alpha, xmin)
    observed = np.array([np.mean(tail >= k) if len(tail) else 0. for k in grid])
    shortage = max(0., 1 - len(tail) / 30.)
    monopoly = max(0., (float(d.max()) / max(1, int(d.sum())) if len(d) else 0.) - .30)
    return float(np.mean((observed - desired) ** 2) + shortage ** 2 + monopoly ** 2)


class ActorCritic:
    """Small softmax actor with a linear Monte-Carlo value baseline.

    One episodic policy-gradient update uses the frozen rollout probabilities.
    History summaries are features, not a claim of a Markov sufficient statistic.
    No theorem-proving or regret guarantee follows from this learner.
    """
    def __init__(self, actor=None, critic=None, learning_rate=.03):
        self.actor = np.zeros((len(MODES), FEATURES)) if actor is None else np.array(actor, dtype=float)
        self.critic = np.zeros(FEATURES) if critic is None else np.array(critic, dtype=float)
        if (self.actor.shape != (len(MODES), FEATURES) or self.critic.shape != (FEATURES,)
                or not np.isfinite(self.actor).all() or not np.isfinite(self.critic).all()
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('Invalid policy parameters')
        self.learning_rate = learning_rate
        self.updates = 0

    def probabilities(self, x):
        logits = self.actor @ x
        p = np.exp(logits - np.max(logits))
        return p / p.sum()

    def choose(self, x, rng):
        p = self.probabilities(x)
        return int(rng.choice(len(MODES), p=p)), p

    def learn(self, trajectory, terminal_reward, discount=.98):
        if not trajectory:
            return
        if not math.isfinite(terminal_reward):
            raise ValueError('Non-finite terminal feedback')
        running = terminal_reward
        returns = []
        for row in reversed(trajectory):
            running = row['reward'] + discount * running
            returns.append(running)
        returns.reverse()
        actor_gradient = np.zeros_like(self.actor)
        critic_gradient = np.zeros_like(self.critic)
        for row, target in zip(trajectory, returns):
            x, p, a = np.array(row['features']), np.array(row['probabilities']), row['action']
            advantage = float(np.clip(target - self.critic @ x, -5., 5.))
            score = -p.copy(); score[a] += 1.
            entropy = -float(np.dot(p, np.log(np.maximum(p, 1e-15))))
            entropy_gradient = -p * (np.log(np.maximum(p, 1e-15)) + entropy)
            actor_gradient += np.outer(advantage * score + .01 * entropy_gradient, x)
            critic_gradient += advantage * x
        self.actor += self.learning_rate * actor_gradient / len(trajectory)
        self.critic += .1 * critic_gradient / len(trajectory)
        self.updates += 1

    def serialise(self):
        return dict(schema='native-online-actor-critic/v1', modes=list(MODES),
                    features=FEATURES, actor=self.actor.tolist(), critic=self.critic.tolist(),
                    learning_rate=self.learning_rate, updates=self.updates)


class OnlineGrowth:
    def __init__(self, lib, config, seed):
        config.validate()
        if config.max_nodes < len(lib.nodes):
            raise ValueError('Node budget smaller than starting library')
        self.lib, self.config = lib, config
        self.rng = np.random.default_rng(seed)
        self.initial_nodes = len(lib.nodes)
        self.counts = Counter()
        self.seen_pairs = set()
        self.bodies = {n.formula for n in lib.nodes}
        self.sizes = []
        self.families = []
        self.major = []
        self.tree_steps = []
        self.index = PatternIndex()
        for n in lib.nodes:
            self._register(n)

    def _register(self, n):
        self.sizes.append(size(n.formula))
        self.families.append(family(n.formula, n.origins))
        if n.formula[0] == 'wi':
            self.major.append(n.id)
        self.tree_steps.append(min(10 ** 12, 1 + sum(self.tree_steps[p] for p, _ in n.parents)))
        self.index.add(n.formula, n.id)

    def features(self):
        n = len(self.lib.nodes)
        d = [self.lib.reuse[i] for i in range(n)]
        derived = [i for i, node in enumerate(self.lib.nodes) if node.parents]
        c = self.counts
        return np.array([1., c['opportunities'] / self.config.attempts,
                         min(1., (n - self.initial_nodes) / max(1, self.config.attempts)),
                         c['known'] / max(1, c['opportunities']),
                         sum(d[i] >= 2 for i in derived) / max(1, len(derived)),
                         sum(d[i] >= self.config.target_xmin for i in derived) / max(1, len(derived)),
                         float(np.mean(self.sizes)) / self.config.formula_size,
                         max(d, default=0) / max(1, sum(d))])

    def choose_pair(self, mode):
        available = self.available_ids()
        allowed = set(available)
        major_ids = [i for i in self.major if i in allowed]
        n = len(self.lib.nodes)
        if not major_ids:
            return None
        candidates = []
        for _ in range(self.config.pair_pool):
            minor = available[int(self.rng.integers(len(available)))]
            major = major_ids[int(self.rng.integers(len(major_ids)))]
            self.counts['cheap_pairs_scored'] += 1
            if (minor, major) in self.seen_pairs:
                continue
            a, b = self.lib.nodes[minor], self.lib.nodes[major]
            # Different rigid constructor roots cannot unify. Variable roots
            # stay eligible, and repeated variables/$d await the actual kernel.
            if a.formula[0] != 'v' and b.formula[1][0] != 'v' and a.formula[0] != b.formula[1][0]:
                self.counts['rigid_root_mismatch'] += 1
                continue
            if mode == 'uniform':
                score = 0.
            elif mode == 'reuse':
                score = math.log1p(self.lib.reuse[minor]) + math.log1p(self.lib.reuse[major])
            elif mode == 'bridge':
                score = 2. * (self.families[minor] != self.families[major])
                score += .2 * math.log1p(self.lib.reuse[minor] + self.lib.reuse[major])
            elif mode == 'contract':
                score = math.log1p(self.sizes[minor] + self.sizes[major]) - math.log1p(size(b.formula[2]))
            elif mode == 'frontier':
                score = 3. * max(minor, major) / n
            else:
                raise ValueError('Unknown generation mechanism')
            candidates.append((score, minor, major))
        if not candidates:
            return None
        logits = np.array([x[0] for x in candidates]); p = np.exp(logits - logits.max()); p /= p.sum()
        _, minor, major = candidates[int(self.rng.choice(len(candidates), p=p))]
        return minor, major

    def available_ids(self):
        return list(range(len(self.lib.nodes)))

    def before_block(self, block):
        """Extension hook for the existing lifecycle/fragment mechanisms."""

    def after_block(self, block, first_new):
        """Extension hook; default direct control has no lifecycle policy."""

    def accepted(self, nid, parents):
        """Extension hook: certified admission has already completed."""

    def attempt(self, mode):
        """One charged opportunity; zero or one committed theorem, no archive."""
        if mode not in MODES:
            raise ValueError('Unknown policy action')
        self.counts['opportunities'] += 1
        self.counts['action_' + mode] += 1
        gen = self.rng.random() < self.config.gen_probability
        try:
            if gen:
                available = self.available_ids()
                parent = available[int(self.rng.integers(len(available)))]
                choices = sorted(v for v in variables(self.lib.nodes[parent].formula) if v[1] == 'setvar')
                choices.append(variable('setvar', 'fresh.generalisation'))
                x = choices[int(self.rng.integers(len(choices)))]
                n = self.lib.nodes[parent]
                proposal = (('wal', x, n.formula), set(n.disjoint), 'ax-gen',
                            {'ph': n.formula, 'x': x},
                            [(parent, {v[2]: v for v in variables(n.formula)})], n.origins)
                self.counts['gen_calls'] += 1
            else:
                pair = self.choose_pair(mode)
                if pair is None:
                    self.counts['empty_parent_pool'] += 1
                    return -.02
                self.seen_pairs.add(pair)
                self.counts['mp_calls'] += 1
                proposal = self.lib.propose_mp(*pair)
            self.counts['valid_proposals'] += 1
        except ValueError:
            self.counts['invalid_or_disjoint'] += 1
            return -.03
        if size(proposal[0]) > self.config.formula_size:
            self.counts['size_rejected'] += 1
            return -.03
        if 1 + max(self.lib.nodes[p].height for p, _ in proposal[4]) > self.config.proof_height:
            self.counts['height_rejected'] += 1
            return -.03
        body, dv = statement(proposal)
        if (body, dv) in self.lib.known:
            self.counts['known'] += 1
            return -.02
        if self.config.subsumption:
            for i in self.index.candidates(body):
                self.counts['subsumption_checks'] += 1
                n = self.lib.nodes[i]
                if instance_witness(n.formula, n.disjoint, body, dv) is not None:
                    self.counts['subsumed'] += 1
                    return -.01
        parents = set(p for p, _ in proposal[4])
        new_body = body not in self.bodies
        # Capped, diminishing first-proof expansion credit: a proxy only.
        gain = sum(min(3., math.log1p(self.tree_steps[p])) / (1 + self.lib.reuse[p])
                   for p in parents if self.lib.nodes[p].parents) / max(1, len(parents))
        newly_reused = sum(bool(self.lib.nodes[p].parents) and self.lib.reuse[p] == 1 for p in parents)
        nid, fresh = self.lib.accept(proposal)
        if not fresh:
            raise AssertionError('Pre-admission key differs from certificate key')
        self._register(self.lib.nodes[nid]); self.bodies.add(body)
        self.counts['admitted'] += 1
        self.counts['new_formula_bodies'] += new_body
        self.counts['new_mp_bodies'] += bool(new_body and not gen)
        self.counts['derived_second_uses'] += newly_reused
        self.accepted(nid, parents)
        if gen:
            # Unary padding cannot earn novelty/expansion reward by itself.
            return -.01
        return -.01 + float(new_body) * (.10 + .15 * gain + .10 * newly_reused)

    def run(self, *, policy='fixed_mix', actor=None):
        if policy not in ('uniform', 'reuse', 'fixed_mix', 'rl') or (policy == 'rl' and actor is None):
            raise ValueError('Invalid generation policy')
        start = time.process_time(); trace = []; trajectory = []
        initial_loss = profile_loss(list(self.lib.reuse.values()), alpha=self.config.target_alpha,
                                    xmin=self.config.target_xmin)
        while self.counts['opportunities'] < self.config.attempts and len(self.lib.nodes) < self.config.max_nodes:
            self.before_block(len(trace) + 1)
            first_new = len(self.lib.nodes)
            x = self.features()
            if policy == 'rl':
                action, probabilities = actor.choose(x, self.rng)
            else:
                action = MODES.index(policy) if policy in MODES else len(trace) % len(MODES)
                probabilities = np.eye(len(MODES))[action]
            before = self.counts['admitted']; reward = 0.
            loss_before = profile_loss([self.lib.reuse[i] for i in range(len(self.lib.nodes))],
                                      alpha=self.config.target_alpha, xmin=self.config.target_xmin)
            opportunities = 0
            for _ in range(min(self.config.block, self.config.attempts - self.counts['opportunities'])):
                if len(self.lib.nodes) >= self.config.max_nodes:
                    break
                reward += self.attempt(MODES[action]); opportunities += 1
            self.after_block(len(trace) + 1, first_new)
            loss_after = profile_loss([self.lib.reuse[i] for i in range(len(self.lib.nodes))],
                                     alpha=self.config.target_alpha, xmin=self.config.target_xmin)
            feedback = reward / max(1, opportunities) + self.config.topology_weight * (loss_before - loss_after)
            trajectory.append(dict(features=x.tolist(), action=action,
                                   probabilities=probabilities.tolist(), reward=feedback))
            trace.append(dict(block=len(trace), action=MODES[action], opportunities=opportunities,
                              admitted=self.counts['admitted'] - before, reward=feedback,
                              nodes=len(self.lib.nodes), profile_loss=loss_after))
        d = [self.lib.reuse[i] for i in range(len(self.lib.nodes))]
        derived = [n.id for n in self.lib.nodes if n.parents]
        metrics = dict(nodes=len(d), initial_nodes=self.initial_nodes,
                       distinct_formula_bodies=len(self.bodies),
                       derived_reused_at_least_twice=sum(d[i] >= 2 for i in derived),
                       derived_reused_once=sum(d[i] >= 1 for i in derived),
                       zero_outdegree=sum(k == 0 for k in d), maximum_outdegree=max(d, default=0),
                       edges=sum(d), degrees_out=d,
                       initial_profile_loss=initial_loss,
                       profile_loss=profile_loss(d, alpha=self.config.target_alpha, xmin=self.config.target_xmin),
                       maximum_height=max(n.height for n in self.lib.nodes))
        result = dict(config=asdict(self.config), policy=policy, counters=dict(self.counts),
                      cpu_seconds=time.process_time() - start, metrics=metrics, trace=trace,
                      stop='node_budget' if len(d) >= self.config.max_nodes else 'opportunity_budget',
                      materialised_discard_archive=0, posthoc_core_extraction=False)
        return result, trajectory


def clone(lib):
    return Library.restore(lib.basis, lib.json_nodes())
