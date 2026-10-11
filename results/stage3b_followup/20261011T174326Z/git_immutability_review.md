# Git and immutability review — Stage 3B follow-up

Run `20261011T174326Z`. No commit, push, reset, stash or remote change was performed.

## Protected Stage 1/2/3A/3B paths
```
```
(empty above = byte-identical; the two permitted diagnostic source edits are in `src/`, not here.)

## Permitted diagnostic source edits
```
 M src/ossfs_dd/selector.py
 M src/ossfs_dd_stage3b/trace.py
```

## Staging dry-run (`git add -A --dry-run`) — no data/binaries/traces
```
  add 'src/ossfs_dd/selector.py'
  add 'src/ossfs_dd_stage3b/trace.py'
  add 'configs/stage3b_followup_v1.yaml'
  add 'results/stage3b_followup/20261011T174326Z/FINAL_VERDICT.json'
  add 'results/stage3b_followup/20261011T174326Z/STAGE3B_FOLLOWUP_REPORT.md'
  add 'results/stage3b_followup/20261011T174326Z/arcene_accuracy_by_order.csv'
  add 'results/stage3b_followup/20261011T174326Z/arcene_all_features_baselines.csv'
  add 'results/stage3b_followup/20261011T174326Z/arcene_order_effects.json'
  add 'results/stage3b_followup/20261011T174326Z/arcene_trajectory_summary.csv'
  add 'results/stage3b_followup/20261011T174326Z/checkpoint_prefix_audit.csv'
  add 'results/stage3b_followup/20261011T174326Z/checkpoint_prefix_audit.json'
  add 'results/stage3b_followup/20261011T174326Z/determinism_repeat.json'
  add 'results/stage3b_followup/20261011T174326Z/freeze_chronology.json'
  add 'results/stage3b_followup/20261011T174326Z/git_immutability_review.md'
  add 'results/stage3b_followup/20261011T174326Z/hypothesis_outcomes.json'
  add 'results/stage3b_followup/20261011T174326Z/immutable_sources_sha256.json'
  add 'results/stage3b_followup/20261011T174326Z/operational_gate_results.json'
  add 'results/stage3b_followup/20261011T174326Z/preflight.json'
  add 'results/stage3b_followup/20261011T174326Z/repair_fail_before_pass_after.json'
  add 'results/stage3b_followup/20261011T174326Z/selector_invariance_50.csv'
  add 'results/stage3b_followup/20261011T174326Z/selector_invariance_50.json'
  add 'results/stage3b_followup/20261011T174326Z/svm_c_tol_summary.csv'
  add 'results/stage3b_followup/20261011T174326Z/svm_paired_comparisons.csv'
  add 'results/stage3b_followup/20261011T174326Z/svm_status_and_hypothesis.json'
  add 'results/stage3b_followup/20261011T174326Z/test_and_command_exits.json'
  add 'results/stage3b_followup/20261011T174326Z/true_removal_ledger_audit.csv'
  add 'results/stage3b_followup/20261011T174326Z/true_removal_ledger_audit.json'
  add 'results/stage3b_followup/baseline_inventory.json'
  add 'results/stage3b_followup/last_pytest.json'
  add 'scripts/stage3b_followup_determinism.py'
  add 'scripts/stage3b_followup_preflight.py'
  add 'scripts/stage3b_followup_report.py'
  add 'scripts/stage3b_followup_run_a.py'
  add 'scripts/stage3b_followup_run_b.py'
  add 'src/ossfs_dd_stage3b_followup/__init__.py'
  add 'src/ossfs_dd_stage3b_followup/analysis.py'
  add 'src/ossfs_dd_stage3b_followup/experiment_a.py'
  add 'src/ossfs_dd_stage3b_followup/experiment_b.py'
  add 'src/ossfs_dd_stage3b_followup/instrument.py'
  add 'tests/test_stage3b_followup_repair_d1.py'
  add 'tests/test_stage3b_followup_repair_d2.py'
```

## Ignored paths (must never be staged)
```
  !! .pytest_cache/
  !! .ruff_cache/
  !! .venv/
  !! configs/paper_reference_discrete.csv
  !! results/stage3a/figures/
  !! results/stage3a/logs/
  !! results/stage3a/runs/
```
