"""Keep the default check source-only; opt into archived-input integration tests."""
import pytest


SOURCE_ONLY = {
    "test_compression_overlay.py": None,
    "test_deferred_queue.py": None,
    "test_metamath_native.py": {
        "test_type_mismatch_fails",
        "test_setvar_does_not_unify_with_class_coercion",
        "test_occurs_check",
        "test_metavariable_disjointness_rejects_capture",
        "test_disjointness_tracks_variables_inside_quantifiers",
        "test_recursive_types_do_not_trust_constructor_tag",
        "test_wrong_arity_fails",
        "test_variable_cannot_inject_metamath_tokens",
        "test_resource_and_target_contract",
    },
    "test_native_holdout.py": {
        "test_binary_grammar_complete_without_duplicates",
        "test_poi_parser_alpha_canonicalises_and_keeps_negation",
        "test_poi_parser_fails_closed",
        "test_definition_expansion_prevents_false_holdout",
        "test_holdout_removes_exposed_and_deduplicates_without_size_tuning",
        "test_unsafe_archive_member_never_extracts",
        "test_poi_named_constants_are_excluded_not_reinterpreted",
        "test_real_adaptation_changes_with_source_not_human_goal",
        "test_quota_controls_count_distinct_delivered_rules_and_keep_short_set",
        "test_seal_rejects_any_frozen_change",
        "test_no_holdout_open_before_all_planned_blocks",
        "test_blind_seal_check_does_not_read_targets_but_opening_checks_them",
        "test_late_results_do_not_earn_cpu_credit",
        "test_paired_sign_flip_and_multiplicity_not_pseudoreplication",
    },
    "test_native_cumulative_policy.py": {
        "test_propositional_adapter_never_erases_quantifiers_or_types",
    },
    "test_native_rule_synthesis.py": {
        "test_partitions_do_not_duplicate_boundary_equalities",
        "test_joint_unification_detects_occurs_cycle",
    },
    "test_native_quality_evaluation.py": set(),
}


def pytest_collection_modifyitems(items):
    for item in items:
        filename = item.path.name
        # Do not change unrelated tests in the preserved local research archive.
        if filename not in SOURCE_ONLY:
            continue
        names = SOURCE_ONLY[filename]
        name = getattr(item, "originalname", None) or item.name
        if names is not None and name not in names:
            item.add_marker(pytest.mark.integration)
