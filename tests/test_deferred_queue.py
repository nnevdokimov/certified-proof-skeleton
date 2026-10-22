"""Small deterministic queue checks without the historical generation drivers."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hpc/article1_skeleton"))
from deferred_gate import OpportunityQueue


def test_fifo_bound_duplicate_and_done_history():
    queue = OpportunityQueue(2)
    for pair in [(1, 2), (1, 2), (2, 3), (3, 4)]:
        queue.add(pair)
    assert len(queue.pairs) == 2
    assert queue.counters["overflow_rejected"] == 1
    assert queue.pop_ready({1, 2}, 1, 2) == [(1, 2)]
    queue.add((1, 2))
    assert queue.counters["replayed_offers_ignored"] == 1
    assert list(queue.pairs) == [(2, 3)]


def test_inactive_pairs_and_parent_reservations():
    queue = OpportunityQueue(8)
    queue.add((1, 2))
    queue.add((3, 4))
    assert queue.pop_ready({1}, 2, 8) == []
    assert queue.reservations({1}, {2: 2}, 3, 2, 4) == [3, 4]
    assert queue.reservations({1}, {2: 2}, 4, 2, 2) == [1, 2]


def test_summary_deterministic_and_history_bounded():
    queue = OpportunityQueue(1)
    for i in range(20):
        queue.add((i, i + 1))
        queue.pop_ready({i, i + 1}, 1, 1)
    assert len(queue.done) == 4
    assert queue.summary() == queue.summary()
    assert queue.summary(False)["queue_sha256"] is None
