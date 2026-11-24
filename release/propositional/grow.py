"""Grow a fresh bank from three axioms with the published native constructor."""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "hpc/article1_skeleton"), str(ROOT / "src")]

import bank
from metamath_native.basis import Assertion
from metamath_native.online_skeleton import OnlineConfig
from metamath_native.proofs import Library
from native_cumulative_policy import CumulativeGrowth


def prefix(term):
    if term[0] == "v":
        return term[2][2:] + "."
    return {"wi": "i", "wn": "n"}[term[0]] + "".join(prefix(t) for t in term[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the trace and bank")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--attempts", type=int, required=True)
    args = parser.parse_args()
    if args.attempts < 1 or args.output.exists():
        parser.error("Use a positive attempt count and a new output directory.")
    basis = json.loads((HERE / "basis.json").read_text())
    model = json.loads((HERE / "gate.json").read_text())
    lib = Library([Assertion(**item) for item in basis])
    lib.initialise()
    if [node.assertion for node in lib.nodes] != ["ax-1", "ax-2", "ax-3"]:
        raise ValueError("Unexpected propositional basis")
    cfg = OnlineConfig(gen_probability=0, attempts=args.attempts, max_nodes=args.attempts + 3)
    engine = CumulativeGrowth(lib, cfg, args.seed, model=model,
                              fragment_selection="transfer", fragment_slots=None)
    engine.run(policy="fixed_mix")

    with TemporaryDirectory(dir=args.output.parent) as temp:
        staging = Path(temp) / "bank"
        staging.mkdir()
        trace = staging / "growth.trace.gz"
        with gzip.open(trace, "wt") as handle:
            for node in lib.nodes:
                premise = ",".join(str(parent) for parent, _ in reversed(node.parents))
                print("K", node.id, premise or node.assertion, prefix(node.formula),
                      sep="\t", file=handle)
        if len(lib.nodes) > 3:
            bank.main(trace, staging / "bank.json", f"ours{args.seed}")
        else:
            print("No derived statements at this budget. Saved the axiom-only trace.")
        os.rename(staging, args.output)


if __name__ == "__main__":
    main()
