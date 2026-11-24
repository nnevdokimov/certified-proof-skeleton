"""Replay the frozen certificates and check the external proof scores."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "hpc/article1_skeleton"))

import bank
from certificate import axioms, mp


def decode(code):
    """Read a formula in the release's prefix notation."""
    tokens = iter(code)

    def read():
        tag = next(tokens)
        if tag.isdigit():
            digits = tag
            for char in tokens:
                if char == ".":
                    return ("v", int(digits))
                if not char.isdigit():
                    break
                digits += char
            raise ValueError(f"Invalid variable in {code}")
        if tag == "n":
            return ("n", read())
        if tag == "i":
            return ("i", read(), read())
        raise ValueError(f"Unknown formula tag {tag}")

    result = read()
    if next(tokens, None) is not None:
        raise ValueError(f"Trailing symbols in {code}")
    return result


def matches(schema, target):
    bindings = {}

    def visit(pattern, term):
        if pattern[0] == "v":
            return bindings.setdefault(pattern[1], term) == term
        return (pattern[0] == term[0] and
                all(visit(a, b) for a, b in zip(pattern[1:], term[1:])))
    return visit(schema, target)


def replay(graph, basis):
    formulas, digests = {}, {}
    for node, parents in sorted(graph.items()):
        if node in basis:
            assert not parents
            formulas[node] = basis[node]
            payload = b"ax" + bytes([node + 1])
        else:
            major, minor = parents
            formulas[node] = mp(formulas[minor], formulas[major])
            payload = b"d" + digests[major] + digests[minor]
        digests[node] = hashlib.sha256(payload).digest()
    return formulas, digests


def traverse(graph, roots, cuts):
    """Count each surviving inference once and report the anchor IDs used."""
    pending = list(roots)
    visited, used = set(), set()
    count = 0
    while pending:
        node = pending.pop()
        if not graph[node] or node in visited:
            continue
        visited.add(node)
        if node in cuts and node not in roots:
            used.add(cuts[node])
        else:
            count += 1
            pending.extend(graph[node])
    return count, sorted(used)


def modes(anchors, summaries):
    for name in summaries:
        if name == "full":
            yield name, anchors
        elif name == "without_identity":
            yield name, [a for a in anchors if a["formula"] != "i0.0."]
        elif name.startswith("k") and name[1:].isdigit():
            yield name, anchors[:int(name[1:])]
        else:
            raise ValueError(f"Unknown bank view {name}")


def check_panel(panel, graph, formulas, digests, data):
    proofs = panel["proofs"]
    for row in proofs:
        root = row["root"]
        assert digests[root].hex() == row["sha256"], row["label"]
        assert matches(formulas[root], decode(row["target"])), row["label"]
        assert traverse(graph, {root}, {})[0] == row["original_steps"], row["label"]
    original = sum(row["original_steps"] for row in proofs)
    hits = {}
    for index, name in enumerate(data["bank_order"]):
        saved_bank = data["banks"][name]
        certificate = {int(n): p for n, p in saved_bank["proof_dag"].items()}
        summaries = panel["per_bank_summary"][name]
        for mode, anchors in modes(saved_bank["anchors"], summaries):
            cuts = {}
            for a in reversed(anchors):
                code = a["formula"]
                if code not in hits:
                    pattern = decode(code)
                    hits[code] = [n for n, f in formulas.items() if graph[n] and matches(pattern, f)]
                cuts.update(dict.fromkeys(hits[code], a["id"]))
            saved = shortened = 0
            for row in proofs:
                remaining, used = traverse(graph, {row["root"]}, cuts)
                gain = row["original_steps"] - remaining
                if mode == "full":
                    assert (gain, used) == (row["steps_saved"][index], row["used_anchor_ids"][index]), (row["label"], name)
                saved += gain
                shortened += gain > 0
            cost, _ = traverse(certificate, {a["id"] for a in anchors}, {})
            summary = summaries[mode]
            assert [summary[k] for k in ("saved", "shortened", "anchors", "definition_cost", "net")] == [
                saved, shortened, len(anchors), cost, saved - cost], (name, mode)
            assert math.isclose(summary["pct"], 100 * saved / original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reselect", action="store_true", help="also regenerate every bank from its growth trace")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without Python's -O flag so the checks execute.")
    data = json.loads((HERE / "skeleton-data.json").read_text())
    basis = {i: formula for i, formula in enumerate(axioms("prop").values())}
    assert {str(i): bank.encoded(f) for i, f in basis.items()} == data["axioms"]
    for name in data["bank_order"]:
        item = data["banks"][name]
        trace = HERE / "traces" / (name + ".trace.gz")
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == item["trace_sha256"], name
        graph = {int(n): p for n, p in item["proof_dag"].items()}
        forms, _ = replay(graph, basis)
        for a in item["anchors"]:
            assert bank.encoded(forms[a["id"]]) == a["formula"], (name, a["id"])
        assert sum(bool(p) for p in graph.values()) == item["definition_steps"], name
        if args.reselect:
            archive, meta = bank.read_archive(trace, "ours" if name.startswith("ours") else "sgcd")
            result = dict(run=name, **meta, **bank.select(archive), trace_sha256=item["trace_sha256"])
            assert result == item, name
    for name, panel in data["panels"].items():
        graph = {int(n): p for n, p in data["proof_graphs"][panel["graph"]].items()}
        check_panel(panel, graph, *replay(graph, basis), data)
        print(f"{name}: {len(panel['proofs'])} proofs checked")
    print(f"{len(data['bank_order'])} certified banks checked" + (" and regenerated" if args.reselect else ""))


if __name__ == "__main__":
    main()
