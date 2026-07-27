#!/usr/bin/env bash
# =============================================================================
# Stage 2 Modified Scoring Re-analysis Pipeline
# =============================================================================
# 一键运行修改计分逻辑后的重新分析与可视化，生成：
#   - stage2/reanalysis_modified_report.txt
#   - stage2/reanalysis_modified_viz_data.json
#   - stage2/figures/modified_*.png
#   - stage2/figures/identity_*.png
#
# 用法：
#   bash stage2/run_reanalysis_modified.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE2_DIR="${ROOT_DIR}/stage2"
FIGURES_DIR="${STAGE2_DIR}/figures"

echo "============================================================"
echo "Stage 2 Modified Scoring Re-analysis"
echo "============================================================"

echo ""
echo "[1/3] Re-analyzing results with modified scoring logic..."
python3 "${STAGE2_DIR}/reanalyze_modified_scoring.py"

echo ""
echo "[2/3] Generating modified scoring visualizations..."
python3 "${STAGE2_DIR}/visualize_modified_scoring.py"

echo ""
echo "[3/3] Generating identity label effect visualizations..."
python3 "${STAGE2_DIR}/visualize_identity_label_effect.py"

echo ""
echo "============================================================"
echo "Done."
echo "Report:    ${STAGE2_DIR}/reanalysis_modified_report.txt"
echo "Viz data:  ${STAGE2_DIR}/reanalysis_modified_viz_data.json"
echo "Figures:   ${FIGURES_DIR}/"
echo "============================================================"
