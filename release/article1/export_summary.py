#!/usr/bin/env python3
"""Export audited aggregates without searching, scoring goals, or fitting graphs.

Requires the original local result packages. This is not a scientific runner.
Only files in --out are written; --check performs a byte-for-byte export check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import mean


RUN = "output/metamath_holdout_confirmation_20261021_v3"
REV = "output/article1_adaptive_revision_20261021_v1"
REF = "output/metamath_native_metrics_20261020_v1"
SEAL = "eb73763e8a0730126498918edcd9bd58876497c1f150e01b3ddfda2288c26080"
METHODS = ["core", "fixed", "random", "adaptive", "fixed_all"]
SEEDS = list(range(2026102201, 2026102217))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def export(root):
    sources = {}

    def record(relative, expected=None):
        path = root / relative
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Source hash mismatch: " + relative)
        sources[relative] = digest
        return path

    def read(relative, expected=None):
        return json.loads(record(relative, expected).read_text())

    complete = read(RUN + "/PIPELINE_COMPLETE.json")
    seal = read(RUN + "/SEAL.json", SEAL)
    protocol = read(RUN + "/PROTOCOL.json", seal["protocol_sha256"])
    review = read(RUN + "/independent_review/INDEPENDENT_REVIEW.json", complete["audit_sha256"])
    receipt = read(RUN + "/EVALUATION_RECEIPT.json")
    expected_summary = next(x["sha256"] for x in receipt["files"] if x["path"] == "SUMMARY.json")
    summary = read(RUN + "/evaluation/results/SUMMARY.json", expected_summary)
    gate = read(REV + "/review/SCIENTIFIC_GATE.json")
    assert complete["status"] == "scored_and_independently_audited"
    assert review["status"] == gate["status"] == "passed"
    assert all(x["seal_sha256"] == SEAL for x in [complete, receipt, review, summary, gate])
    assert summary["status"] == review["producer_conclusion"] == "no_confirmed_adaptive_superiority"
    assert summary["all_horizons_complete"] and not review["confirmation_supported"]
    assert protocol["seeds"] == SEEDS and protocol["methods"] == METHODS
    assert summary["targets"] == review["audited_targets"] == 963
    assert len(summary["rows"]) == review["audited_cells"] == 160
    assert summary["source_blocks"] == review["audited_source_blocks"] == 16
    assert gate["evaluation_receipt_sha256"] == sources[RUN + "/EVALUATION_RECEIPT.json"]
    assert gate["pipeline_sha256"] == sources[RUN + "/PIPELINE_COMPLETE.json"]
    assert gate["independent_review_sha256"] == complete["audit_sha256"]

    # Verify receipt identities, not re-run the original proof or scoring audit.
    source_receipts = {}
    for row in gate["source_receipts"]:
        p = RUN + "/construction/" + str(row["seed"]) + "/RECEIPT.json"
        r = read(p, row["receipt_sha256"])
        assert r["status"] == "complete_before_target_opening" and r["seal_sha256"] == SEAL
        assert len(r["files"]) == row["files"] == 95
        source_receipts[row["seed"]] = {f["path"]: f["sha256"] for f in r["files"]}

    cells = {(r["seed"], r["method"], r["policy"]): r for r in summary["rows"]}
    assert len(cells) == 160
    assert set(cells) == {(s, m, p) for s in SEEDS for m in METHODS for p in protocol["policies"]}
    means = []
    for method in METHODS:
        rows = [cells[s, method, p] for s in SEEDS for p in protocol["policies"]]
        row = dict(method=method, source_blocks=16, continuations=32,
                   initial_coverage_mean=mean(r["initial_coverage"] for r in rows))
        for horizon in ("future", "total"):
            row[horizon] = {key + "_mean": mean(r[horizon][key] for r in rows)
                            for key in ("new_residual", "exact_new_residual", "normalised_auc", "residual_fraction")}
        means.append(row)
    # Balanced two-policy averages equal the mean of the 16 source-block means.
    for control, horizons in summary["comparisons"].items():
        for horizon, result in horizons.items():
            got = next(r for r in means if r["method"] == "adaptive")[horizon]["new_residual_mean"]
            got -= next(r for r in means if r["method"] == control)[horizon]["new_residual_mean"]
            assert abs(got - result["mean"]) < 1e-12
    comparisons = {m: {h: {k: v for k, v in r.items() if k != "differences"}
                       for h, r in horizons.items()}
                   for m, horizons in summary["comparisons"].items()}
    holdout = dict(
        schema="article1-compact-holdout/v1", status=summary["status"],
        recommended_configuration="fixed", confirmation_supported=False,
        source_blocks=16, continuations=160, held_out_targets=963,
        all_horizons_complete=True, methods=METHODS, policies=protocol["policies"],
        adaptive_distinct_selections=summary["adaptive_distinct_selections"],
        method_means=means, adaptive_versus_control=comparisons,
        holm_adjusted_p=summary["holm_adjusted_p"], success_rule=protocol["success"],
        protocol={k: protocol[k] for k in (
            "primary", "primary_total", "inference", "timing_scope", "representation",
            "fixed_control", "quota_scope", "future_cpu", "total_cpu", "formula_size",
            "proof_height", "source_attempts", "source_initial_nodes", "quota", "maximum_mp")},
        technical_amendment={k: protocol["technical_repair"][k] for k in (
            "kind", "interpretation", "previous_seal_sha256", "previous_targets_opened", "authorised_change")},
        audit=review, interpretation=[
            "Fixed is the reproducible working configuration, not a proven optimal rule set or universal winner.",
            "The registered adaptive-superiority criterion did not pass; this does not establish equivalence.",
            "The 160 continuations form 16 paired source blocks, not 160 independent theories.",
            "The held-out corpus shares Metamath ancestry with development data; it is not all of Metamath or ZFC.",
            "These are future-utility outcomes, distinct from initial coverage and descriptive network metrics."])

    network_protocol = read(REV + "/networks/PROTOCOL.json")
    network_summary = read(REV + "/networks/SUMMARY.json")
    assert network_summary["status"] == "complete_descriptive_only"
    assert network_summary["graph_metrics_do_not_measure_future_quality"] is True
    profiles = []
    for row in network_summary["networks"]:
        name = row["name"]
        p = read(REV + "/networks/results/" + name + ".json")
        assert p["graph_sha256"] == row["graph_sha256"]
        record(REV + "/networks/graphs/" + name + ".npz", row["graph_sha256"])
        initial = row["source"]
        expected = source_receipts[row["seed"]][row["method"] + "/initial.json.gz"]
        record(initial, expected)
        assert expected == row["source_sha256"]
        tail = p["tails"]["reuse_out"]
        item = {k: row[k] for k in ("name", "method", "seed", "source_sha256", "graph_sha256",
                "nodes", "arcs", "roots", "derived_anchors", "hidden_support",
                "maximum_certificate_height", "clustering", "transitivity", "path", "diameter",
                "components", "modularity", "modularity_null", "sigma", "gini", "assortativity", "max_core")}
        item.update(tail_alpha=tail["power_law"]["alpha"], tail_xmin=tail["power_law"]["xmin"],
                    tail_n=tail["power_law"]["n_tail"], tail_p=tail["bootstrap_gof"]["p"],
                    tail_p_mc_ci95=tail["bootstrap_gof"]["binomial_ci95"],
                    zero_out_degree=tail["n_zero"],
                    giant_component_fraction=p["observed"]["paths"]["gcc_fraction"])
        profiles.append(item)
    assert len(profiles) == 32
    assert {(p["method"], p["seed"]) for p in profiles} == {(m, s) for m in ("fixed", "adaptive") for s in SEEDS}
    representative = next(p for p in profiles if p["name"] == "fixed_2026102201")
    assert (representative["nodes"], representative["arcs"]) == (841, 1573)
    representative_detail = read(REV + "/networks/results/fixed_2026102201.json")
    ref_audit = read(REF + "/FINAL_AUDIT.json")
    assert ref_audit["status"] == "verified"
    read(REF + "/PROTOCOL.json", ref_audit["protocol_sha256"])
    reference_profiles = []
    for frozen in network_protocol["unchanged_panels"]:
        key = frozen["key"]
        data = read(REF + "/results/" + key + ".json", ref_audit["results"][key])
        assert data["graph_sha256"] == frozen["graph_sha256"]
        reference_profiles.append({k: data[k] for k in (
            "key", "label", "nodes", "arcs", "graph_sha256", "paths", "clustering", "transitivity",
            "modularity", "modularity_null_mean", "undirected_assortativity", "max_core", "maximum_in_degree",
            "dag", "sigma_DP_exact_observed_sampled_null", "sigma_ER_exact_observed_sampled_null",
            "sigma_convention", "power_law", "normalized_alternatives")})
    network = dict(
        schema="article1-compact-network-profiles/v1", status="descriptive_only",
        recommended_method="fixed", representative="fixed_2026102201", profiles=profiles,
        reference_profiles=reference_profiles,
        representative_small_world=representative_detail["small_world"],
        representative_tail_alternatives=representative_detail["tails"]["reuse_out"]["normalized_alternatives"],
        protocol={k: network_protocol[k] for k in (
            "representative_source", "selection", "observations", "nulls", "louvain_starts",
            "seed", "tail_bootstrap", "pure_power_law", "search_or_experiment_changed")},
        interpretation=[
            "Graphs use the matched axiom basis but differ in size, depth, proof granularity, contexts, and budget.",
            "H3 is the complete finite slice of the declared operator, not infinite ZFC closure.",
            "Paths refer to the giant connected component of an undirected simple projection; clustering includes low-degree zeros.",
            "Sigma compares clustering and paths with degree-preserving nulls; short paths alone do not establish small-world structure.",
            "Tail fits are finite conditional iid diagnostics; non-rejection does not establish an asymptotic scale-free law or unique model.",
            "Ten null graphs do not support a rank-test p-value below 1/11.",
            "The fixed representative uses the first source, not the source with the best target coverage.",
            "Descriptive network metrics do not measure future proof utility or establish method superiority."])
    provenance = dict(
        schema="article1-compact-provenance/v1", seal_sha256=SEAL,
        protocol_sha256=sources[RUN + "/PROTOCOL.json"],
        source_files=[dict(path=p, sha256=sources[p]) for p in sorted(sources)],
        verification_scope=[
            "Hash-check final pipeline/audit links, sealed protocol, evaluation-summary receipt, 16 source-receipt identities, and scientific-gate links.",
            "Hash-check 32 initial checkpoints against their source receipts and all 32 graph arrays against existing graph profiles.",
            "Hash-check three refined reference results against their final audit.",
            "Copy existing results and calculate balanced descriptive means only; do not search, score goals, refit metrics, or repeat the independent audit.",
            "This compact release omits raw goals, trajectories, proof archives, manuscript files, and personal metadata."])
    return {"holdout_results.json": holdout, "network_profiles.json": network, "provenance.json": provenance}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = export(args.source_root.resolve())
    encoded = {name: encode(value) for name, value in outputs.items()}
    for name, payload in encoded.items():
        if re.search(rb"/Users/|/home/|[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", payload):
            raise ValueError("Private path or email in exported file: " + name)
    if sum(map(len, encoded.values())) > 150_000:
        raise ValueError("Compact export grew beyond the intended size")
    for name, payload in encoded.items():
        path = args.out / name
        if args.check:
            if path.read_bytes() != payload:
                raise ValueError("Export differs: " + name)
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(json.dumps(dict(status="checked" if args.check else "exported",
                          files={n: len(p) for n, p in encoded.items()},
                          total_bytes=sum(map(len, encoded.values()))), sort_keys=True))


if __name__ == "__main__":
    main()
