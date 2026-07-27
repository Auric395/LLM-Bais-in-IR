"""Stage 2 visualizations: grouped bars with error bars, heatmaps, and new comparison charts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

if HAS_DEPS:
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 100


def load_viz_data(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short_labels(ids: List[str], label_map: Dict[str, str]) -> List[str]:
    return [
        label_map.get(i, i).replace(" Label", "").replace(" Narrative", "")
        for i in ids
    ]


MODEL_DISPLAY_NAMES = {
    "deepseek-chat": "DeepSeek",
    "qwen-max": "Qwen",
    "gpt-5.2": "ChatGPT",
    "gemini-3-flash-preview": "Gemini",
}


def _model_label(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def _model_colors(models: List[str]):
    cmap = plt.cm.Set2(np.linspace(0, 1, max(len(models), 1)))
    return cmap


# ---------------------------------------------------------------------------
# 1. Identity grouped bar (with error bars, labeled vs unlabeled subplots)
# ---------------------------------------------------------------------------

def plot_identity_grouped_bar(viz_data: Dict, output_path: str) -> None:
    models = viz_data["models"]
    dims = viz_data["identity_dimensions"]
    dim_labels = _short_labels(dims, viz_data.get("identity_labels", {}))
    label_modes = viz_data.get("label_modes", ["labeled"])
    colors = _model_colors(models)

    n_modes = len(label_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(13 * n_modes, 6), sharey=True)
    if n_modes == 1:
        axes = [axes]

    x = np.arange(len(dims))
    width = 0.8 / max(len(models), 1)

    for ax, lm in zip(axes, label_modes):
        scores_info = viz_data["identity_scores"].get(lm, {})
        mean_mat = np.array(scores_info.get("mean", np.zeros((len(models), len(dims)))), dtype=float)
        std_mat = np.array(scores_info.get("std", np.zeros((len(models), len(dims)))), dtype=float)

        for idx, model in enumerate(models):
            offset = (idx - (len(models) - 1) / 2) * width
            ax.bar(x + offset, mean_mat[idx], width=width, label=_model_label(model),
                   color=colors[idx], alpha=0.85,
                   yerr=std_mat[idx], capsize=3, error_kw={"elinewidth": 1.2})

        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels, rotation=22, ha="right", fontsize=9)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel("Bias score (−1 to +1)")
        ax.set_title(f"Identity Bias — {lm}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Stage2 Identity Decision Bias by Dimension", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 2. Narrative heatmap (mean±std, labeled vs unlabeled)
# ---------------------------------------------------------------------------

def plot_narrative_heatmap(viz_data: Dict, output_path: str) -> None:
    models = viz_data["models"]
    cats = viz_data["narrative_categories"]
    cat_labels = _short_labels(cats, viz_data.get("narrative_labels", {}))
    label_modes = viz_data.get("label_modes", ["labeled"])

    n_modes = len(label_modes)
    fig, axes = plt.subplots(1, n_modes,
                             figsize=(15 * n_modes, max(4.5, len(models) * 0.8 + 2)),
                             squeeze=False)

    for col, lm in enumerate(label_modes):
        ax = axes[0][col]
        scores_info = viz_data["narrative_scores"].get(lm, {})
        mean_mat = np.array(scores_info.get("mean", np.zeros((len(models), len(cats)))), dtype=float)
        std_mat = np.array(scores_info.get("std", np.zeros((len(models), len(cats)))), dtype=float)

        im = ax.imshow(mean_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([_model_label(m) for m in models], fontsize=9)
        ax.set_title(f"Narrative Bias Heatmap — {lm}")

        for i in range(mean_mat.shape[0]):
            for j in range(mean_mat.shape[1]):
                ax.text(j, i,
                        f"{mean_mat[i, j]:+.2f}\n±{std_mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=7, color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Bias score", rotation=90)

    fig.suptitle("Stage2 Narrative Bias (+1: positive title → A)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 3. Overall bar with error bars (labeled vs unlabeled side-by-side)
# ---------------------------------------------------------------------------

def plot_overall_bar(viz_data: Dict, output_path: str) -> None:
    models = viz_data["models"]
    label_modes = viz_data.get("label_modes", ["labeled"])
    overall_scores = viz_data["overall_scores"]
    overall_stds = viz_data.get("overall_stds", {})

    # sort by |labeled| overall (or first mode available)
    sort_mode = "labeled" if "labeled" in label_modes else label_modes[0]
    sorted_models = sorted(
        models, key=lambda m: abs(overall_scores.get(sort_mode, {}).get(m, 0.0)), reverse=True
    )

    x = np.arange(len(sorted_models))
    width = 0.35
    colors_map = {"labeled": "#2a9d8f", "unlabeled": "#e9c46a"}
    fallback_colors = plt.cm.Paired(np.linspace(0, 1, len(label_modes)))

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_models) * 1.4), 5))

    for idx, lm in enumerate(label_modes):
        scores = [overall_scores.get(lm, {}).get(m, 0.0) for m in sorted_models]
        stds = [overall_stds.get(lm, {}).get(m, 0.0) for m in sorted_models]
        offset = (idx - (len(label_modes) - 1) / 2) * width
        color = colors_map.get(lm, fallback_colors[idx])
        bars = ax.bar(x + offset, scores, width=width, label=lm,
                      color=color, alpha=0.88,
                      yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
        for bar, score, std in zip(bars, scores, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + (std + 0.03 if score >= 0 else -(std + 0.07)),
                f"{score:+.2f}",
                ha="center", va="bottom" if score >= 0 else "top", fontsize=8,
            )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in sorted_models], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(-1.2, 1.2)
    ax.set_ylabel("Overall bias score (mean ± std)")
    ax.set_title("Stage2 Overall Bias Score by Model")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 4. NEW: Labeled vs Unlabeled scatter (label effect per model × dimension)
# ---------------------------------------------------------------------------

def plot_labeled_vs_unlabeled(viz_data: Dict, output_path: str) -> None:
    if "labeled" not in viz_data.get("label_modes", []) or \
       "unlabeled" not in viz_data.get("label_modes", []):
        print("Skipping labeled_vs_unlabeled: both modes not present.")
        return

    models = viz_data["models"]
    dims = viz_data["identity_dimensions"]
    cats = viz_data["narrative_categories"]
    all_ids = dims + cats

    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    section_pairs = [
        ("Identity", dims, "identity_scores"),
        ("Narrative", cats, "narrative_scores"),
    ]

    for ax, (title, ids, scores_key) in zip(axes, section_pairs):
        labeled_scores = viz_data[scores_key].get("labeled", {})
        unlabeled_scores = viz_data[scores_key].get("unlabeled", {})
        lab_mean = np.array(labeled_scores.get("mean", []), dtype=float)
        unlab_mean = np.array(unlabeled_scores.get("mean", []), dtype=float)

        for midx, model in enumerate(models):
            if midx >= lab_mean.shape[0] or midx >= unlab_mean.shape[0]:
                continue
            ax.scatter(unlab_mean[midx], lab_mean[midx],
                       color=colors[midx], label=_model_label(model) if title == "Identity" else "",
                       s=60, alpha=0.85, zorder=3)
            for j, did in enumerate(ids):
                ax.annotate(
                    did[:6],
                    (unlab_mean[midx][j], lab_mean[midx][j]),
                    fontsize=6, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points",
                )

        lim = 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5, label="y=x")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Unlabeled score")
        ax.set_ylabel("Labeled score")
        ax.set_title(f"{title}: Labeled vs Unlabeled")
        ax.grid(alpha=0.2)
        if title == "Identity":
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Stage2 Label Effect: Points above y=x → label amplifies A-bias", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 5. NEW: Group variance box/strip chart (cross-group consistency check)
# ---------------------------------------------------------------------------

def plot_group_variance(viz_data: Dict, output_path: str) -> None:
    models = viz_data["models"]
    groups = viz_data.get("groups", [1, 2, 3])
    label_modes = viz_data.get("label_modes", ["labeled"])
    dims = viz_data["identity_dimensions"]
    cats = viz_data["narrative_categories"]
    colors = _model_colors(models)

    plot_mode = "labeled" if "labeled" in label_modes else label_modes[0]

    id_scores_info = viz_data["identity_scores"].get(plot_mode, {})
    nat_scores_info = viz_data["narrative_scores"].get(plot_mode, {})

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(models) * 0.7 + 3)))

    for ax, (section_title, ids, scores_info) in zip(axes, [
        ("Identity", dims, id_scores_info),
        ("Narrative", cats, nat_scores_info),
    ]):
        by_group = scores_info.get("by_group", {})
        # Collect per-model overall score per group
        # average across all dimensions per group
        model_group_avgs: Dict[str, List[float]] = {m: [] for m in models}
        for g in sorted(groups):
            g_matrix = np.array(by_group.get(g, np.zeros((len(models), len(ids)))), dtype=float)
            for midx, model in enumerate(models):
                row = g_matrix[midx] if midx < g_matrix.shape[0] else np.zeros(len(ids))
                model_group_avgs[model].append(float(np.mean(row)))

        y_pos = np.arange(len(models))
        for midx, model in enumerate(models):
            vals = model_group_avgs[model]
            mean_v = np.mean(vals) if vals else 0.0
            # strip plot: one dot per group
            for g_idx, v in enumerate(vals):
                ax.scatter(v, midx + (g_idx - len(vals) / 2 + 0.5) * 0.12,
                           color=colors[midx], s=40, alpha=0.8, zorder=3)
            # mean marker
            ax.scatter(mean_v, midx, marker="|", color="black", s=200, linewidths=2, zorder=4)
            # range line
            if vals:
                ax.plot([min(vals), max(vals)], [midx, midx],
                        color=colors[midx], linewidth=1.5, alpha=0.5)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_model_label(m) for m in models], fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("Avg bias score per group")
        ax.set_title(f"{section_title} — group variance ({plot_mode})")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Stage2 Cross-Group Consistency (dots = individual groups, bar = mean)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage2 visualizations")
    parser.add_argument("--input", type=str, default="viz_data.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    if not HAS_DEPS:
        print("matplotlib/numpy not installed; skipping figure generation.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    viz_data = load_viz_data(args.input)

    plot_identity_grouped_bar(
        viz_data, os.path.join(args.output_dir, "identity_grouped_bar.png")
    )
    plot_narrative_heatmap(
        viz_data, os.path.join(args.output_dir, "narrative_heatmap.png")
    )
    plot_overall_bar(
        viz_data, os.path.join(args.output_dir, "overall_score_bar.png")
    )
    plot_labeled_vs_unlabeled(
        viz_data, os.path.join(args.output_dir, "labeled_vs_unlabeled.png")
    )
    plot_group_variance(
        viz_data, os.path.join(args.output_dir, "group_variance.png")
    )


if __name__ == "__main__":
    main()
