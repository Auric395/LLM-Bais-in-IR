"""Stage 2: Analyze results across 3 groups and labeled/unlabeled modes."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_results(results_dir: str) -> List[Dict]:
    results = []
    if not os.path.isdir(results_dir):
        return results
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith("result_stage2_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, encoding="utf-8") as f:
                results.append(json.load(f))
    return results


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def analyze_results(results: List[Dict]) -> Dict:
    """
    Aggregate results by (model, label_mode) across groups.

    Returns:
        analysis dict with per-model stats including cross-group mean/std
        and labeled_vs_unlabeled delta.
    """
    # keyed by (model, label_mode, group_id) → result
    result_map: Dict[Tuple, Dict] = {}
    for r in results:
        key = (r["model"], r.get("label_mode", "labeled"), r.get("group_id", 1))
        # keep latest if duplicate
        if key not in result_map or r["timestamp"] > result_map[key]["timestamp"]:
            result_map[key] = r

    # collect all models and label_modes
    models = sorted({k[0] for k in result_map})
    label_modes = sorted({k[1] for k in result_map})
    groups = sorted({k[2] for k in result_map})

    # collect dimension/category names
    identity_dim_ids: List[str] = []
    identity_label_map: Dict[str, str] = {}
    narrative_cat_ids: List[str] = []
    narrative_label_map: Dict[str, str] = {}

    for r in result_map.values():
        for dim in r.get("identity_dimensions", []):
            did = dim["dimension_id"]
            if did not in identity_dim_ids:
                identity_dim_ids.append(did)
            identity_label_map[did] = dim.get("label_name", did)
        for cat in r.get("narrative_categories", []):
            cid = cat["category_id"]
            if cid not in narrative_cat_ids:
                narrative_cat_ids.append(cid)
            narrative_label_map[cid] = cat.get("topic", cid)

    # ----------------------------------------------------------------
    # Build per-(model, label_mode): scores by group
    # ----------------------------------------------------------------
    analysis: Dict = {
        "models": models,
        "label_modes": label_modes,
        "groups": groups,
        "identity_dim_ids": identity_dim_ids,
        "narrative_cat_ids": narrative_cat_ids,
        "identity_label_map": identity_label_map,
        "narrative_label_map": narrative_label_map,
        "by_model_mode": {},
    }

    for model in models:
        for lm in label_modes:
            key_ml = f"{model}||{lm}"
            # scores per dimension per group
            id_scores_by_group: Dict[str, Dict[int, float]] = defaultdict(dict)
            nat_scores_by_group: Dict[str, Dict[int, float]] = defaultdict(dict)
            overall_by_group: Dict[int, float] = {}

            for g in groups:
                r = result_map.get((model, lm, g))
                if r is None:
                    continue
                overall_by_group[g] = r.get("overall_score", 0.0)
                for dim in r.get("identity_dimensions", []):
                    id_scores_by_group[dim["dimension_id"]][g] = dim["score"]
                for cat in r.get("narrative_categories", []):
                    nat_scores_by_group[cat["category_id"]][g] = cat["score"]

            # compute mean ± std across groups
            def group_stats(scores_by_group: Dict[str, Dict[int, float]]) -> Dict:
                out = {}
                for dim_id, g_scores in scores_by_group.items():
                    vals = list(g_scores.values())
                    m, s = _mean_std(vals)
                    out[dim_id] = {
                        "by_group": g_scores,
                        "mean": m,
                        "std": s,
                    }
                return out

            overall_vals = list(overall_by_group.values())
            overall_mean, overall_std = _mean_std(overall_vals)

            analysis["by_model_mode"][key_ml] = {
                "model": model,
                "label_mode": lm,
                "identity": group_stats(id_scores_by_group),
                "narrative": group_stats(nat_scores_by_group),
                "overall_by_group": overall_by_group,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
            }

    # ----------------------------------------------------------------
    # Labeled vs unlabeled delta
    # ----------------------------------------------------------------
    labeled_vs_unlabeled: Dict[str, Dict] = {}
    if "labeled" in label_modes and "unlabeled" in label_modes:
        for model in models:
            labeled_key = f"{model}||labeled"
            unlabeled_key = f"{model}||unlabeled"
            if labeled_key not in analysis["by_model_mode"]:
                continue
            if unlabeled_key not in analysis["by_model_mode"]:
                continue
            lab = analysis["by_model_mode"][labeled_key]
            unlab = analysis["by_model_mode"][unlabeled_key]

            id_delta = {}
            for did in identity_dim_ids:
                lab_mean = lab["identity"].get(did, {}).get("mean", 0.0)
                unlab_mean = unlab["identity"].get(did, {}).get("mean", 0.0)
                id_delta[did] = round(lab_mean - unlab_mean, 6)

            nat_delta = {}
            for cid in narrative_cat_ids:
                lab_mean = lab["narrative"].get(cid, {}).get("mean", 0.0)
                unlab_mean = unlab["narrative"].get(cid, {}).get("mean", 0.0)
                nat_delta[cid] = round(lab_mean - unlab_mean, 6)

            labeled_vs_unlabeled[model] = {
                "overall_delta": round(lab["overall_mean"] - unlab["overall_mean"], 6),
                "identity_delta": id_delta,
                "narrative_delta": nat_delta,
            }
    analysis["labeled_vs_unlabeled"] = labeled_vs_unlabeled

    # ----------------------------------------------------------------
    # Overall ranking by |overall_mean| (labeled mode preferred)
    # ----------------------------------------------------------------
    rank_mode = "labeled" if "labeled" in label_modes else label_modes[0]
    ranking = []
    for model in models:
        key_ml = f"{model}||{rank_mode}"
        entry = analysis["by_model_mode"].get(key_ml, {})
        ranking.append({
            "model": model,
            "overall_mean": entry.get("overall_mean", 0.0),
            "overall_std": entry.get("overall_std", 0.0),
            "identity_mean": _mean_std(
                [v for g_s in entry.get("identity", {}).values()
                 for v in g_s.get("by_group", {}).values()]
            )[0] if entry.get("identity") else 0.0,
            "narrative_mean": _mean_std(
                [v for g_s in entry.get("narrative", {}).values()
                 for v in g_s.get("by_group", {}).values()]
            )[0] if entry.get("narrative") else 0.0,
        })
    ranking.sort(key=lambda x: abs(x["overall_mean"]), reverse=True)
    analysis["overall_ranking"] = ranking

    return analysis


def generate_report(analysis: Dict, output_path: str) -> str:
    lines: List[str] = []
    sep = "=" * 80
    lines += [sep, "Model Bias Experiment - Stage 2 Analysis Report", sep, ""]

    lines += ["## Overall Ranking (by |overall_mean|, labeled mode)", "-" * 80]
    lines.append(
        f"{'Rank':<5} {'Model':<30} {'Overall±std':>14} {'Identity':>10} {'Narrative':>10}"
    )
    lines.append("-" * 80)
    for rank, entry in enumerate(analysis["overall_ranking"], 1):
        lines.append(
            f"{rank:<5} {entry['model']:<30} "
            f"{entry['overall_mean']:>+7.4f}±{entry['overall_std']:.4f} "
            f"{entry['identity_mean']:>+10.4f} {entry['narrative_mean']:>+10.4f}"
        )

    for lm in analysis["label_modes"]:
        lines += ["", f"## Identity Scores ({lm} mode)", "-" * 80]
        for did in analysis["identity_dim_ids"]:
            label_name = analysis["identity_label_map"].get(did, did)
            lines.append(f"\n### {did} ({label_name})")
            lines.append(f"{'Model':<30} {'Mean':>8} {'Std':>8} {'Interpretation'}")
            lines.append("-" * 60)
            rows = []
            for model in analysis["models"]:
                key_ml = f"{model}||{lm}"
                entry = analysis["by_model_mode"].get(key_ml, {})
                stats = entry.get("identity", {}).get(did, {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                rows.append((model, mean, std))
            rows.sort(key=lambda x: x[1], reverse=True)
            for model, mean, std in rows:
                interp = "Favors A" if mean > 0.1 else "Favors B" if mean < -0.1 else "Neutral"
                lines.append(f"{model:<30} {mean:>+8.4f} {std:>8.4f} {interp}")

    for lm in analysis["label_modes"]:
        lines += ["", f"## Narrative Scores ({lm} mode)", "-" * 80]
        for cid in analysis["narrative_cat_ids"]:
            topic = analysis["narrative_label_map"].get(cid, cid)
            lines.append(f"\n### {cid} ({topic})")
            lines.append(f"{'Model':<30} {'Mean':>8} {'Std':>8} {'Interpretation'}")
            lines.append("-" * 60)
            rows = []
            for model in analysis["models"]:
                key_ml = f"{model}||{lm}"
                entry = analysis["by_model_mode"].get(key_ml, {})
                stats = entry.get("narrative", {}).get(cid, {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                rows.append((model, mean, std))
            rows.sort(key=lambda x: x[1], reverse=True)
            for model, mean, std in rows:
                interp = (
                    "Pos→A" if mean > 0.1 else "Pos→B" if mean < -0.1 else "Neutral"
                )
                lines.append(f"{model:<30} {mean:>+8.4f} {std:>8.4f} {interp}")

    if analysis["labeled_vs_unlabeled"]:
        lines += ["", "## Label Effect (labeled_mean - unlabeled_mean)", "-" * 80]
        lines.append(f"{'Model':<30} {'Overall Δ':>12}")
        lines.append("-" * 45)
        for model, delta_data in sorted(
            analysis["labeled_vs_unlabeled"].items(),
            key=lambda x: abs(x[1]["overall_delta"]),
            reverse=True,
        ):
            lines.append(f"{model:<30} {delta_data['overall_delta']:>+12.4f}")

    lines += [
        "", sep,
        "Score: +1 = always picks A (Western/labeled camp); -1 = always picks B.",
        "Std: across 3 same-distribution data groups.",
        sep,
    ]

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return text


def export_for_visualization(analysis: Dict, output_path: str) -> Dict:
    models = analysis["models"]
    label_modes = analysis["label_modes"]
    identity_dims = analysis["identity_dim_ids"]
    narrative_cats = analysis["narrative_cat_ids"]
    groups = analysis["groups"]

    def score_matrix(dim_or_cat_ids, section_key):
        """Returns {label_mode: {mean: matrix, std: matrix, by_group: {g: matrix}}}"""
        out: Dict = {}
        for lm in label_modes:
            means = []
            stds = []
            by_group = {g: [] for g in groups}
            for model in models:
                key_ml = f"{model}||{lm}"
                entry = analysis["by_model_mode"].get(key_ml, {})
                row_mean, row_std = [], []
                for did in dim_or_cat_ids:
                    stats = entry.get(section_key, {}).get(did, {})
                    row_mean.append(stats.get("mean", 0.0))
                    row_std.append(stats.get("std", 0.0))
                    for g in groups:
                        by_group[g].append(
                            stats.get("by_group", {}).get(g, 0.0)
                        )
                means.append(row_mean)
                stds.append(row_std)
            # reshape by_group rows: group → [model × dim] matrix
            n_dims = len(dim_or_cat_ids)
            bg_matrix = {}
            for g in groups:
                flat = by_group[g]
                bg_matrix[g] = [flat[i * n_dims:(i + 1) * n_dims] for i in range(len(models))]
            out[lm] = {"mean": means, "std": stds, "by_group": bg_matrix}
        return out

    viz_data = {
        "models": models,
        "label_modes": label_modes,
        "groups": groups,
        "identity_dimensions": identity_dims,
        "narrative_categories": narrative_cats,
        "identity_labels": analysis["identity_label_map"],
        "narrative_labels": analysis["narrative_label_map"],
        "identity_scores": score_matrix(identity_dims, "identity"),
        "narrative_scores": score_matrix(narrative_cats, "narrative"),
        "overall_scores": {
            lm: {
                model: analysis["by_model_mode"].get(f"{model}||{lm}", {}).get("overall_mean", 0.0)
                for model in models
            }
            for lm in label_modes
        },
        "overall_stds": {
            lm: {
                model: analysis["by_model_mode"].get(f"{model}||{lm}", {}).get("overall_std", 0.0)
                for model in models
            }
            for lm in label_modes
        },
        "labeled_vs_unlabeled": analysis.get("labeled_vs_unlabeled", {}),
        "overall_ranking": analysis["overall_ranking"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)
    return viz_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage2 experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result_stage2_*.json files")
    parser.add_argument("--output-report", type=str, default="analysis_report.txt")
    parser.add_argument("--output-viz", type=str, default="viz_data.json")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No stage2 result files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result file(s)")
    analysis = analyze_results(results)
    generate_report(analysis, args.output_report)
    export_for_visualization(analysis, args.output_viz)
    print(f"\nSaved report: {args.output_report}")
    print(f"Saved viz data: {args.output_viz}")


if __name__ == "__main__":
    main()
