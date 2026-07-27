#!/usr/bin/env python3
"""
生成模型—判断模型分解回归（正式版）

本脚本复现并扩展了 Jupyter 笔记本中的分析：
1. 读取 stage2/results/merged_by_judge/ 下的合并结果；
2. 展开到 iteration 级别，计算方向性偏见得分；
3. 运行加权最小二乘（WLS）固定效应回归与 GEE 稳健性检验；
4. 输出回归表格、emmeans、交互矩阵、标签效应、方向性比例等 CSV；
5. 绘制调整后效应图、交互矩阵图等。

输出目录：generator_judge_decomposition/figures/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "stage2" / "results" / "merged_by_judge"
OUTPUT_DIR = SCRIPT_DIR / "figures"

MODEL_FILES = {
    "DeepSeek": "deepseek-chat_results.json",
    "Gemini": "gemini-3-flash-preview_results.json",
    "ChatGPT": "gpt-5.2_results.json",
    "Qwen": "qwen-max_results.json",
}

MODEL_ORDER = ["DeepSeek", "Qwen", "ChatGPT", "Gemini"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_merged_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged_by_judge files and expand to iteration-level DataFrames."""
    identity_rows, narrative_rows = [], []
    for judge_label, filename in MODEL_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        for block in obj["results"]:
            generator_label = _display_name(block["generator_model"])
            label_mode = block["label_mode"]
            labeled = int(label_mode == "labeled")
            rr = block["result"]

            for dim in rr.get("identity_dimensions", []):
                for item in dim.get("items", []):
                    for it in item.get("iterations", []):
                        identity_rows.append({
                            "judge": judge_label,
                            "generator": generator_label,
                            "label_mode": label_mode,
                            "labeled": labeled,
                            "dimension": dim["dimension_id"],
                            "item_id": item["item_id"],
                            "task": f"{dim['dimension_id']}_{item['item_id']}",
                            "choice": it.get("choice"),
                            "response": it.get("response", ""),
                        })

            for cat in rr.get("narrative_categories", []):
                for item in cat.get("items", []):
                    for it in item.get("iterations", []):
                        narrative_rows.append({
                            "judge": judge_label,
                            "generator": generator_label,
                            "label_mode": label_mode,
                            "labeled": labeled,
                            "category": cat["category_id"],
                            "item_id": item["item_id"],
                            "task": f"{cat['category_id']}_{item['item_id']}",
                            "positive_for": it.get("positive_for"),
                            "response": it.get("response", ""),
                        })

    return pd.DataFrame(identity_rows), pd.DataFrame(narrative_rows)


def _display_name(model: str) -> str:
    mapping = {
        "deepseek-chat": "DeepSeek",
        "qwen-max": "Qwen",
        "gpt-5.2": "ChatGPT",
        "gemini-3-flash-preview": "Gemini",
    }
    return mapping.get(model, model)


def prepare(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Add empty/directional/west_binary/signed columns."""
    d = df.copy()
    outcome = "choice" if stage == "identity" else "positive_for"
    d["is_empty"] = d["response"].fillna("").astype(str).str.strip().eq("")
    d["directional"] = d[outcome].isin(["A", "B"])
    d["west_binary"] = np.where(d[outcome] == "A", 1,
                                np.where(d[outcome] == "B", 0, np.nan))
    d["signed"] = np.where(d[outcome] == "A", 1,
                           np.where(d[outcome] == "B", -1, 0))
    d["base_stimulus"] = d["generator"] + "__" + d["task"]
    return d


def audit(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "total": len(d),
        "technical_empty": int(d["is_empty"].sum()),
        "nonempty": int((~d["is_empty"]).sum()),
        "directional": int(d["directional"].sum()),
        "substantive_non_directional": int((~d["is_empty"] & ~d["directional"]).sum()),
        "directional_rate": d["directional"].sum() / (~d["is_empty"]).sum(),
    })


def aggregate_cells(d: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Aggregate to (judge, generator, label_mode, task) cells."""
    outcome = "choice" if stage == "identity" else "positive_for"
    gcols = ["judge", "generator", "label_mode", "labeled", "task", "base_stimulus"]
    rows = []
    for keys, g in d.groupby(gcols, observed=True):
        nonempty = g[~g["is_empty"]]
        a = int((nonempty[outcome] == "A").sum())
        b = int((nonempty[outcome] == "B").sum())
        valid = a + b
        rows.append(dict(zip(gcols, keys)) | {
            "n_nonempty": len(nonempty),
            "valid_n": valid,
            "direction_rate": valid / len(nonempty) if len(nonempty) else np.nan,
            "direction_score": (a - b) / valid if valid else np.nan,
            "signed_mean": (a - b) / len(nonempty) if len(nonempty) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regression formulas
# ---------------------------------------------------------------------------
RHS = (
    "C(generator, Treatment('DeepSeek')) + "
    "C(judge, Treatment('DeepSeek')) + labeled + C(task) + "
    "C(generator, Treatment('DeepSeek')):labeled + "
    "C(judge, Treatment('DeepSeek')):labeled"
)
RHS_INTER = RHS + " + C(generator, Treatment('DeepSeek')):C(judge, Treatment('DeepSeek'))"


def fit_wls(cells: pd.DataFrame, interaction: bool = False):
    d = cells.dropna(subset=["direction_score"]).copy()
    rhs = RHS_INTER if interaction else RHS
    return smf.wls(
        f"direction_score ~ {rhs}", d, weights=d["valid_n"]
    ).fit(cov_type="cluster", cov_kwds={"groups": d["base_stimulus"]})


def fit_gee(d: pd.DataFrame, interaction: bool = False):
    use = d[d["directional"]].copy()
    rhs = RHS_INTER if interaction else RHS
    return smf.gee(
        f"west_binary ~ {rhs}", groups="base_stimulus", data=use,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence()
    ).fit(maxiter=100)


def fit_direction_rate_wls(cells: pd.DataFrame):
    """WLS regression for the proportion of directional responses in a cell.

    Uses the same right-hand side as the main WLS model (generator, judge,
    label, generator×label, judge×label, task fixed effects) so that the
    auxiliary test is directly comparable to the main directional-bias model.
    """
    d = cells.dropna(subset=["direction_rate"]).copy()
    return smf.wls(
        f"direction_rate ~ {RHS}", d, weights=d["n_nonempty"]
    ).fit(cov_type="cluster", cov_kwds={"groups": d["base_stimulus"]})


def fit_signed_mean_wls(cells: pd.DataFrame, interaction: bool = False):
    """WLS regression treating non-directional substantive responses as 0.

    This "net bias" model uses signed_mean = (N^W - N^NW) / N_nonempty as the
    outcome, so it simultaneously reflects directional bias and the propensity
    to form a directional choice.
    """
    d = cells.dropna(subset=["signed_mean"]).copy()
    rhs = RHS_INTER if interaction else RHS
    return smf.wls(
        f"signed_mean ~ {rhs}", d, weights=d["n_nonempty"]
    ).fit(cov_type="cluster", cov_kwds={"groups": d["base_stimulus"]})


def term_table(res) -> pd.DataFrame:
    tab = res.wald_test_terms(skip_single=False).table.copy()
    tab["statistic"] = tab["statistic"].map(lambda x: float(np.asarray(x).squeeze()))
    tab["pvalue"] = tab["pvalue"].map(lambda x: float(np.asarray(x).squeeze()))
    return tab


def interaction_test(res) -> sm.stats.contrast.wald:
    names = [n for n in res.params.index if "C(generator" in n and ":C(judge" in n]
    R = np.zeros((len(names), len(res.params)))
    ix = {n: i for i, n in enumerate(res.params.index)}
    for r, n in enumerate(names):
        R[r, ix[n]] = 1
    return res.wald_test(R, scalar=True)


# ---------------------------------------------------------------------------
# EMMeans and contrasts
# ---------------------------------------------------------------------------
def emmeans(res, rhs_formula: str, cells: pd.DataFrame, focus_var: str, focus_levels: List[str]) -> pd.DataFrame:
    """Estimated marginal means for focus_var, averaging over a balanced grid."""
    factor_cols = ["judge", "generator", "labeled", "task"]
    unique_vals = {c: cells[c].unique() for c in factor_cols}

    grids = []
    for level in focus_levels:
        others = {c: unique_vals[c] for c in factor_cols if c != focus_var}
        import itertools
        for combo in itertools.product(*others.values()):
            row = {focus_var: level}
            for c, v in zip(others.keys(), combo):
                row[c] = v
            grids.append(row)
    grid_df = pd.DataFrame(grids)

    X_grid = patsy.dmatrix(rhs_formula, grid_df, return_type="dataframe")

    rows = []
    for level in focus_levels:
        mask = grid_df[focus_var] == level
        X_level = X_grid[mask]
        c = X_level.mean(axis=0).values
        pred = float(c @ res.params.values)
        var = float(c @ res.cov_params().values @ c)
        se = np.sqrt(var)
        rows.append({
            "model": level,
            "estimate": pred,
            "se": se,
            "ci_low": pred - 1.96 * se,
            "ci_high": pred + 1.96 * se,
        })
    return pd.DataFrame(rows)


def interaction_matrix(res, rhs_formula: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Adjusted predictions for each generator × judge combination."""
    factor_cols = ["judge", "generator", "labeled", "task"]
    unique_vals = {c: cells[c].unique() for c in factor_cols}

    import itertools
    grids = []
    for combo in itertools.product(*[unique_vals[c] for c in factor_cols]):
        grids.append(dict(zip(factor_cols, combo)))
    grid_df = pd.DataFrame(grids)

    X_grid = patsy.dmatrix(rhs_formula, grid_df, return_type="dataframe")

    rows = []
    for gen in unique_vals["generator"]:
        for judge in unique_vals["judge"]:
            mask = (grid_df["generator"] == gen) & (grid_df["judge"] == judge)
            X_sub = X_grid[mask]
            c = X_sub.mean(axis=0).values
            pred = float(c @ res.params.values)
            var = float(c @ res.cov_params().values @ c)
            se = np.sqrt(var)
            rows.append({
                "generator": gen,
                "judge": judge,
                "estimate": pred,
                "ci_low": pred - 1.96 * se,
                "ci_high": pred + 1.96 * se,
            })
    return pd.DataFrame(rows)


def label_effects_by_judge(res, rhs_formula: str, cells: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Labeled - unlabeled effect for each judge, averaging over generator/task."""
    factor_cols = ["judge", "generator", "labeled", "task"]
    unique_vals = {c: cells[c].unique() for c in factor_cols}

    import itertools
    grids = []
    for combo in itertools.product(*[unique_vals[c] for c in factor_cols]):
        grids.append(dict(zip(factor_cols, combo)))
    grid_df = pd.DataFrame(grids)

    X_grid = patsy.dmatrix(rhs_formula, grid_df, return_type="dataframe")

    rows = []
    for judge in unique_vals["judge"]:
        mask_labeled = (grid_df["judge"] == judge) & (grid_df["labeled"] == 1)
        mask_unlabeled = (grid_df["judge"] == judge) & (grid_df["labeled"] == 0)
        c_labeled = X_grid[mask_labeled].mean(axis=0).values
        c_unlabeled = X_grid[mask_unlabeled].mean(axis=0).values
        c_diff = c_labeled - c_unlabeled
        pred = float(c_diff @ res.params.values)
        var = float(c_diff @ res.cov_params().values @ c_diff)
        se = np.sqrt(var)
        rows.append({
            "judge": judge,
            "stage": stage,
            "label_effect": pred,
            "se": se,
            "ci_low": pred - 1.96 * se,
            "ci_high": pred + 1.96 * se,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Direction rates
# ---------------------------------------------------------------------------
def direction_rate_tables(cells: pd.DataFrame, by: str) -> pd.DataFrame:
    """Average direction_rate by generator or judge."""
    grouped = cells.groupby(by).agg(
        direction_rate_mean=("direction_rate", "mean"),
        direction_rate_std=("direction_rate", "std"),
        n=("direction_rate", "count"),
    ).reset_index()
    grouped.columns = [by, "direction_rate_mean", "direction_rate_std", "n"]
    return grouped


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_adjusted_effects(emm_gen: pd.DataFrame, emm_judge: pd.DataFrame,
                          stage: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    gen_vals = [emm_gen[emm_gen["model"] == m]["estimate"].values[0] for m in MODEL_ORDER]
    gen_se = [emm_gen[emm_gen["model"] == m]["se"].values[0] for m in MODEL_ORDER]
    judge_vals = [emm_judge[emm_judge["model"] == m]["estimate"].values[0] for m in MODEL_ORDER]
    judge_se = [emm_judge[emm_judge["model"] == m]["se"].values[0] for m in MODEL_ORDER]

    ax.errorbar(x - width/2, gen_vals, yerr=1.96*np.array(gen_se), fmt="o",
                label="Generator-source effect", capsize=4, markersize=8)
    ax.errorbar(x + width/2, judge_vals, yerr=1.96*np.array(judge_se), fmt="s",
                label="Judge-model effect", capsize=4, markersize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Adjusted directional bias score")
    ax.set_title(f"{stage} decisions: adjusted generator and judge effects")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved figure: {save_path}")


def plot_interaction_matrix(mat: pd.DataFrame, save_path: Path):
    pivot = mat.pivot(index="generator", columns="judge", values="estimate")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlabel("Judge Model")
    ax.set_ylabel("Generator Model")
    ax.set_title("Identity decisions: adjusted generator × judge bias scores")

    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            val = pivot.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved figure: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    identity, narrative = load_merged_results()
    identity = prepare(identity, "identity")
    narrative = prepare(narrative, "narrative")

    print("\nData audit:")
    audit_df = pd.DataFrame({"Identity": audit(identity), "Narrative": audit(narrative)}).T
    print(audit_df)
    audit_df.to_csv(OUTPUT_DIR / "table_01_data_audit.csv")

    print("\nAggregating cells...")
    id_cells = aggregate_cells(identity, "identity")
    nar_cells = aggregate_cells(narrative, "narrative")
    print(f"  Identity cells: {id_cells.shape}")
    print(f"  Narrative cells: {nar_cells.shape}")

    # WLS regressions
    print("\nFitting WLS regressions...")
    id_main = fit_wls(id_cells)
    id_inter = fit_wls(id_cells, True)
    nar_main = fit_wls(nar_cells)
    nar_inter = fit_wls(nar_cells, True)

    # Main-effect Wald tables
    block_tests = []
    for stage, res in [("Identity", id_main), ("Narrative", nar_main)]:
        tab = term_table(res).loc[[
            "C(generator, Treatment('DeepSeek'))",
            "C(judge, Treatment('DeepSeek'))",
            "C(generator, Treatment('DeepSeek')):labeled",
            "C(judge, Treatment('DeepSeek')):labeled",
        ]].copy()
        tab["stage"] = stage
        tab["term"] = tab.index
        block_tests.append(tab.reset_index(drop=True))
    block_tests_df = pd.concat(block_tests, ignore_index=True)
    block_tests_df = block_tests_df[["stage", "term", "statistic", "df_constraint", "pvalue"]]
    block_tests_df.to_csv(OUTPUT_DIR / "table_02_block_tests.csv", index=False)
    print("\nWLS main-effect tests:")
    print(block_tests_df)

    # Interaction tests
    print("\nWLS generator × judge interaction tests:")
    id_inter_test = interaction_test(id_inter)
    nar_inter_test = interaction_test(nar_inter)
    print(f"  Identity: {id_inter_test}")
    print(f"  Narrative: {nar_inter_test}")

    # EMMeans
    print("\nComputing EMMeans...")
    id_gen_emm = emmeans(id_main, RHS, id_cells, "generator", MODEL_ORDER)
    id_judge_emm = emmeans(id_main, RHS, id_cells, "judge", MODEL_ORDER)
    nar_gen_emm = emmeans(nar_main, RHS, nar_cells, "generator", MODEL_ORDER)
    nar_judge_emm = emmeans(nar_main, RHS, nar_cells, "judge", MODEL_ORDER)

    id_gen_emm.to_csv(OUTPUT_DIR / "table_03_identity_generator_emmeans.csv", index=False)
    id_judge_emm.to_csv(OUTPUT_DIR / "table_04_identity_judge_emmeans.csv", index=False)
    nar_gen_emm.to_csv(OUTPUT_DIR / "table_05_narrative_generator_emmeans.csv", index=False)
    nar_judge_emm.to_csv(OUTPUT_DIR / "table_06_narrative_judge_emmeans.csv", index=False)

    # Interaction matrix
    print("\nComputing interaction matrices...")
    id_inter_mat = interaction_matrix(id_inter, RHS_INTER, id_cells)
    id_inter_mat.to_csv(OUTPUT_DIR / "table_07_identity_generator_judge_matrix.csv", index=False)

    # Label effects by judge
    print("\nComputing label effects by judge...")
    id_label_eff = label_effects_by_judge(id_main, RHS, id_cells, "identity")
    nar_label_eff = label_effects_by_judge(nar_main, RHS, nar_cells, "narrative")
    id_label_eff.to_csv(OUTPUT_DIR / "table_08_identity_label_effects_by_judge.csv", index=False)
    nar_label_eff.to_csv(OUTPUT_DIR / "table_09_narrative_label_effects_by_judge.csv", index=False)

    # Direction rates
    print("\nComputing direction rates...")
    nar_dir_gen = direction_rate_tables(nar_cells, "generator")
    nar_dir_judge = direction_rate_tables(nar_cells, "judge")
    nar_dir_gen.to_csv(OUTPUT_DIR / "table_10_narrative_direction_rate_by_generator.csv", index=False)
    nar_dir_judge.to_csv(OUTPUT_DIR / "table_11_narrative_direction_rate_by_judge.csv", index=False)

    # Auxiliary WLS for narrative direction rate (reproduces appendix Wald tests)
    print("\nFitting narrative direction-rate regression...")
    nar_dir_res = fit_direction_rate_wls(nar_cells)
    nar_dir_wald = term_table(nar_dir_res).loc[[
        "C(generator, Treatment('DeepSeek'))",
        "C(judge, Treatment('DeepSeek'))",
    ]].copy()
    nar_dir_wald["term"] = nar_dir_wald.index.map({
        "C(generator, Treatment('DeepSeek'))": "Generator model",
        "C(judge, Treatment('DeepSeek'))": "Judge model",
    })
    nar_dir_wald = nar_dir_wald.reset_index(drop=True)[["term", "statistic", "df_constraint", "pvalue"]]
    nar_dir_wald.to_csv(OUTPUT_DIR / "table_13_narrative_direction_rate_wald.csv", index=False)
    print(nar_dir_wald)

    # Exclude military-topic robustness for narrative
    print("\nFitting narrative model excluding military topics...")
    non_military_tasks = [t for t in nar_cells["task"].unique() if "military" not in t]
    nar_cells_no_military = nar_cells[nar_cells["task"].isin(non_military_tasks)]
    nar_no_mil_main = fit_wls(nar_cells_no_military)
    nar_no_mil_inter = fit_wls(nar_cells_no_military, True)
    no_mil_tab = term_table(nar_no_mil_main).loc[[
        "C(generator, Treatment('DeepSeek'))",
        "C(judge, Treatment('DeepSeek'))",
    ]].copy()
    no_mil_tab["term"] = no_mil_tab.index.map({
        "C(generator, Treatment('DeepSeek'))": "Generator model",
        "C(judge, Treatment('DeepSeek'))": "Judge model",
    })
    no_mil_tab = no_mil_tab.reset_index(drop=True)[["term", "statistic", "df_constraint", "pvalue"]]
    no_mil_inter = interaction_test(nar_no_mil_inter)
    no_mil_inter_row = pd.DataFrame([{
        "term": "Generator × judge",
        "statistic": float(np.asarray(no_mil_inter.statistic).squeeze()),
        "df_constraint": int(np.asarray(no_mil_inter.df_denom).squeeze()),
        "pvalue": float(np.asarray(no_mil_inter.pvalue).squeeze()),
    }])
    no_mil_df = pd.concat([no_mil_tab, no_mil_inter_row], ignore_index=True)
    no_mil_df.to_csv(OUTPUT_DIR / "table_15_narrative_no_military.csv", index=False)
    print(no_mil_df)

    # Net-bias robustness (treat non-directional substantive responses as 0)
    print("\nFitting net-bias robustness regressions...")
    id_net_main = fit_signed_mean_wls(id_cells)
    id_net_inter = fit_signed_mean_wls(id_cells, True)
    nar_net_main = fit_signed_mean_wls(nar_cells)
    nar_net_inter = fit_signed_mean_wls(nar_cells, True)

    net_bias_rows = []
    for stage, res_main, res_inter in [
        ("Identity", id_net_main, id_net_inter),
        ("Narrative", nar_net_main, nar_net_inter),
    ]:
        tab = term_table(res_main).loc[[
            "C(generator, Treatment('DeepSeek'))",
            "C(judge, Treatment('DeepSeek'))",
        ]]
        for effect, row in tab.iterrows():
            net_bias_rows.append({
                "stage": stage,
                "effect": effect.replace("C(generator, Treatment('DeepSeek'))", "Generator model")
                                     .replace("C(judge, Treatment('DeepSeek'))", "Judge model"),
                "chi2": row["statistic"],
                "df": row["df_constraint"],
                "p": row["pvalue"],
            })
        inter = interaction_test(res_inter)
        net_bias_rows.append({
            "stage": stage,
            "effect": "Generator × judge",
            "chi2": float(np.asarray(inter.statistic).squeeze()),
            "df": int(np.asarray(inter.df_denom).squeeze()),
            "p": float(np.asarray(inter.pvalue).squeeze()),
        })
    net_bias_df = pd.DataFrame(net_bias_rows)
    net_bias_df.to_csv(OUTPUT_DIR / "table_14_net_bias_robustness.csv", index=False)
    print("\nNet-bias robustness:")
    print(net_bias_df)

    # GEE robustness
    print("\nFitting GEE robustness checks...")
    id_gee = fit_gee(identity)
    id_gee_inter = fit_gee(identity, True)
    nar_gee = fit_gee(narrative)
    nar_gee_inter = fit_gee(narrative, True)

    gee_rows = []
    for stage, res, res_inter in [
        ("Identity", id_gee, id_gee_inter),
        ("Narrative", nar_gee, nar_gee_inter),
    ]:
        tab = term_table(res).loc[[
            "C(generator, Treatment('DeepSeek'))",
            "C(judge, Treatment('DeepSeek'))",
        ]]
        for effect, row in tab.iterrows():
            gee_rows.append({
                "stage": stage,
                "effect": effect.replace("C(generator, Treatment('DeepSeek'))", "Generator model")
                                     .replace("C(judge, Treatment('DeepSeek'))", "Judge model"),
                "chi2": row["statistic"],
                "df": row["df_constraint"],
                "p": row["pvalue"],
            })
        inter = interaction_test(res_inter)
        gee_rows.append({
            "stage": stage,
            "effect": "Generator × judge",
            "chi2": float(np.asarray(inter.statistic).squeeze()),
            "df": int(np.asarray(inter.df_denom).squeeze()),
            "p": float(np.asarray(inter.pvalue).squeeze()),
        })
    gee_df = pd.DataFrame(gee_rows)
    gee_df.to_csv(OUTPUT_DIR / "table_12_gee_robustness.csv", index=False)
    print("\nGEE robustness:")
    print(gee_df)

    # Figures
    print("\nGenerating figures...")
    plot_adjusted_effects(id_gen_emm, id_judge_emm, "Identity",
                          OUTPUT_DIR / "figure_01_identity_adjusted_effects.png")
    plot_adjusted_effects(nar_gen_emm, nar_judge_emm, "Narrative",
                          OUTPUT_DIR / "figure_02_narrative_adjusted_effects.png")
    plot_interaction_matrix(id_inter_mat, OUTPUT_DIR / "figure_03_identity_interaction_matrix.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
