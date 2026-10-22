"""Exact marginal gains, inherited dictionaries, and endogenous stopping."""
from collections import Counter
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hpc/article1_skeleton'))
from compression_overlay import encoding_state, marginal_gain, compress_above, continuation_working_set
from fragment_policy import costs


def dag(seed, n=25):
    rng = random.Random(seed)
    return {i: (i, None, 'A' if i < 3 else 'MP',
                () if i < 3 else tuple(rng.randrange(i) for _ in range(rng.randint(1, 2))))
            for i in range(n)}


def test_fast_marginals_equal_existing_exact_costs_with_overlap():
    for seed in range(30):
        records = dag(seed); roots = list(range(10, 25)) + [24, 24]
        base = [4, 8, 13]
        state = encoding_state(records, roots, base)
        before = costs(records, roots, base)['encoded_symbols']
        assert state['encoded_symbols'] == before
        for i in state['support']:
            if i not in base:
                assert marginal_gain(state, i) == before - costs(records, roots, base + [i])['encoded_symbols']


def test_additional_compression_keeps_all_existing_definitions():
    records = dag(42)
    selected, report = compress_above(records, list(range(3, 25)), [5, 12])
    assert {5, 12} <= set(selected)
    assert report['additional_saved_symbols'] >= 0
    assert report['extra_limit'] is None
    assert report['largest_remaining_eligible_gain'] <= 0
    assert all(row['marginal_saved_symbols'] > 0 for row in report['trace'])


def test_size_follows_content_not_a_fixed_number():
    plain = {i: (i, None, 'A', ()) for i in range(10)}
    a, _ = compress_above(plain, list(plain))
    b, _ = compress_above(dag(91, 50), list(range(3, 50)))
    assert not a and len(b) > 0


def test_fixed_size_is_only_an_explicit_control_condition():
    selected, r = compress_above(dag(42, 50), list(range(3, 50)), extra_limit=0)
    assert selected == []
    assert r['stop'] == 'explicit_ablation_limit_with_positive_gain_remaining'


def test_activity_cap_does_not_become_a_dictionary_threshold():
    selected = continuation_working_set([3, 4, 5, 6], [7, 8], [0, 1], Counter({4: 9, 6: 8}), 2)
    assert selected == [0, 1, 4, 6]
