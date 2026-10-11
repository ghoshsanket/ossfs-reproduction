"""Stage-3A plots.

Matplotlib runs headless (Agg) and is imported lazily so the sweep workers
never pay for it. Every plot states in its own caption what it is not: the
published values are descriptive context, and the runtime panel carries no
comparability claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .paper import PAPER_DOI  # noqa: E402

CLF_TITLE = {"knn": "KNN (k=5)", "svm": "SVM (linear, C=1)", "cart": "CART (Gini)"}
VARIANT_TITLE = {
    "outcomes_raw_primary": "Raw features (primary)",
    "outcomes_standardized_sensitivity": "Standardized (sensitivity)",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_vs_paper(
    aggregate: dict[str, Any], paper: dict[str, Any], variant: str, out: Path
) -> Path:
    """Our mean accuracy against the published reference, per dataset."""
    datasets = list(aggregate["variants"][variant]["datasets"])
    classifiers = ["knn", "svm", "cart"]
    rows = {(r["dataset"], r["classifier"]): r for r in paper["rows"]}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.36
    for ax, dataset in zip(axes, datasets, strict=True):
        ours, ref, x = [], [], np.arange(len(classifiers))
        for clf in classifiers:
            row = rows.get((dataset, clf), {})
            ours.append(row.get("ours_mean_accuracy") or 0.0)
            ref.append(row.get("paper_accuracy") or 0.0)
        ax.bar(x - width / 2, ours, width, label="This reproduction", color="#2f6f9f")
        ax.bar(x + width / 2, ref, width, label=f"Published (DOI {PAPER_DOI})", color="#b0b0b0")
        ax.set_xticks(x)
        ax.set_xticklabels([CLF_TITLE[c] for c in classifiers], fontsize=8)
        ax.set_title(dataset)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy (mean over seeds of fold means)")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        f"Stage 3A accuracy vs published reference — {VARIANT_TITLE[variant]}\n"
        "Descriptive comparison only; no statistical test against a published mean.",
        fontsize=10,
    )
    _save(fig, out)
    return out


def feature_counts(
    aggregate: dict[str, Any], paper: dict[str, Any], out: Path
) -> Path:
    rows = {(r["dataset"], r["classifier"]): r for r in paper["rows"]}
    datasets = list(aggregate["variants"]["outcomes_raw_primary"]["datasets"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.36
    for ax, dataset in zip(axes, datasets, strict=True):
        entry = aggregate["variants"]["outcomes_raw_primary"]["datasets"][dataset]
        x = np.arange(3)
        ours = [entry["selected_count_mean"]] * 3
        ref = [rows.get((dataset, c), {}).get("paper_mean_features") or 0.0 for c in
               ("knn", "svm", "cart")]
        ax.bar(x - width / 2, ours, width, label="This reproduction", color="#2f6f9f")
        ax.bar(x + width / 2, ref, width, label="Published", color="#b0b0b0")
        ax.set_xticks(x)
        ax.set_xticklabels(["KNN", "SVM", "CART"])
        ax.set_title(f"{dataset}\n(total {entry['n_features_total']})")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean selected features")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Selected-feature count vs published reference "
        "(one shared subset serves all three classifiers)",
        fontsize=10,
    )
    _save(fig, out)
    return out


def per_seed_spread(aggregate: dict[str, Any], out: Path) -> Path:
    """Per-seed mean accuracy, showing the spread the single mean hides."""
    block = aggregate["variants"]["outcomes_raw_primary"]["datasets"]
    datasets = list(block)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets, strict=True):
        data, labels = [], []
        for clf, stats in block[dataset]["classifiers"].items():
            vals = [v for v in stats["per_seed_mean"].values() if v is not None]
            data.append(vals)
            labels.append(CLF_TITLE[clf])
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.3)
        for i, vals in enumerate(data, start=1):
            ax.scatter([i] * len(vals), vals, s=8, alpha=0.6, color="#2f6f9f", zorder=3)
    axes[0].set_ylabel("Per-seed mean accuracy")
    fig.suptitle(
        "Spread of per-seed mean accuracy (10 seeds x 5 folds, raw features)\n"
        "Bootstrap intervals over these seeds are descriptive only.",
        fontsize=10,
    )
    _save(fig, out)
    return out


def runtime_context(aggregate: dict[str, Any], paper: dict[str, Any], out: Path) -> Path:
    """Selection runtime next to the published runtime, explicitly incomparable."""
    ours = paper["our_runtime_seconds_mean"]
    ref = paper["paper_runtime_seconds"]
    datasets = [d for d in ours if d in ref]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(datasets))
    ax.bar(x - 0.2, [ours[d] for d in datasets], 0.4, label="This reproduction (sklearn)",
           color="#2f6f9f")
    ax.bar(x + 0.2, [ref[d] for d in datasets], 0.4, label="Published (MATLAB)", color="#b0b0b0")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean selection seconds (log scale)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Selection runtime — NOT COMPARABLE\n"
        "Different hardware, language and timing protocol; no speedup or parity is claimed.",
        fontsize=10,
    )
    _save(fig, out)
    return out


def render_all(aggregate: dict[str, Any], paper_by_variant: dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for variant, paper in paper_by_variant.items():
        paths.append(
            accuracy_vs_paper(
                aggregate, paper, variant, out_dir / f"accuracy_vs_paper_{variant}.png"
            )
        )
    paths.append(
        feature_counts(aggregate, paper_by_variant["outcomes_raw_primary"],
                        out_dir / "feature_counts.png")
    )
    paths.append(per_seed_spread(aggregate, out_dir / "per_seed_spread.png"))
    paths.append(
        runtime_context(aggregate, paper_by_variant["outcomes_raw_primary"],
                        out_dir / "runtime_context.png")
    )
    return paths
