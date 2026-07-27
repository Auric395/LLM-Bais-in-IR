#!/usr/bin/env bash
# =============================================================================
# Stage 2 一键流水线：对应词配对 + 四模型生成 + 披露标签测试
#
# 流程：
#   Phase 1: 四个模型各自生成数据（对应词配对 A[i]↔B[i]）→ 4组数据
#   Phase 2: 对每组数据，四个模型分别做 labeled/unlabeled 测试（10次迭代，固定温度）
#   Phase 3: 汇总分析 + 可视化
#
# 用法：
#   bash scripts/run_all.sh              # 默认 api 模式
#   bash scripts/run_all.sh mock         # mock 模式（测试流程）
#   bash scripts/run_all.sh api 0.5 20   # 自定义温度和迭代次数
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# ---- 可配置参数 ----
MODE="${1:-api}"                # "api" 或 "mock"
TEMPERATURE="${2:-0.7}"         # 固定温度
ITERATIONS="${3:-10}"           # 迭代次数
LABEL_MODE="both"               # labeled + unlabeled 都跑
USE_FIRST_N=4                   # 身份维度使用前N个选项
NARRATIVE_USE_FIRST_N=1         # 叙事测试只使用第一对选项

DATA_DIR="${ROOT_DIR}/data/run_${RUN_ID}"
OUTPUT_DIR="${ROOT_DIR}/stage2/results/${RUN_ID}"
FIGURE_DIR="${ROOT_DIR}/stage2/figures/${RUN_ID}"

echo "============================================================"
echo "Stage2 Pipeline: 对应词配对 × 四模型生成 × 披露标签测试"
echo "============================================================"
echo "Run ID:         ${RUN_ID}"
echo "Mode:           ${MODE}"
echo "Temperature:    ${TEMPERATURE}"
echo "Iterations:     ${ITERATIONS}"
echo "Label mode:     ${LABEL_MODE}"
echo "Data dir:       ${DATA_DIR}"
echo "Results dir:    ${OUTPUT_DIR}"
echo "Figures dir:    ${FIGURE_DIR}"
echo ""

# ---- Phase 1: 数据生成（四模型 × 对应词配对） ----
echo "------------------------------------------------------------"
echo "Phase 1: 四模型分别生成数据（对应词配对 A[i]↔B[i]）"
echo "------------------------------------------------------------"
python3 "${ROOT_DIR}/scripts/generate_data2.py" \
  --input "${ROOT_DIR}/data/data1.json" \
  --models "${ROOT_DIR}/model_list.json" \
  --output-dir "${DATA_DIR}" \
  --pairing corresponding \
  --all-generators \
  --mode "${MODE}" \
  --use-first-n "${USE_FIRST_N}"

echo ""
echo "数据生成完成，文件列表："
ls -la "${DATA_DIR}"/data2_gen*.json 2>/dev/null || echo "  (未找到文件)"

# ---- Phase 2: 实验（全模型 × 全数据集 × labeled+unlabeled） ----
echo ""
echo "------------------------------------------------------------"
echo "Phase 2: 披露标签测试"
echo "         每组数据 × 4模型 × labeled/unlabeled × ${ITERATIONS}次迭代"
echo "         温度固定: ${TEMPERATURE}"
echo "------------------------------------------------------------"
mkdir -p "${OUTPUT_DIR}"

DATASET_COUNT=0
for DATA_FILE in "${DATA_DIR}"/data2_gen*.json; do
  [ -f "${DATA_FILE}" ] || continue
  DATASET_COUNT=$((DATASET_COUNT + 1))
  echo ""
  echo "  >> [${DATASET_COUNT}/4] 数据集: $(basename "${DATA_FILE}")"
  python3 "${ROOT_DIR}/stage2/experiment.py" \
    --data "${DATA_FILE}" \
    --models "${ROOT_DIR}/model_list.json" \
    --iterations "${ITERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --label-mode "${LABEL_MODE}" \
    --mode "${MODE}" \
    --output "${OUTPUT_DIR}"
done

echo ""
echo "实验完成，共处理 ${DATASET_COUNT} 个数据集"

# ---- Phase 3: 分析 + 可视化 ----
echo ""
echo "------------------------------------------------------------"
echo "Phase 3: 汇总分析 + 可视化"
echo "------------------------------------------------------------"
python3 "${ROOT_DIR}/stage2/analyze.py" \
  --results-dir "${OUTPUT_DIR}" \
  --output-report "${OUTPUT_DIR}/analysis_report.txt" \
  --output-viz "${OUTPUT_DIR}/viz_data.json"

mkdir -p "${FIGURE_DIR}"
python3 "${ROOT_DIR}/stage2/visualize.py" \
  --input "${OUTPUT_DIR}/viz_data.json" \
  --output-dir "${FIGURE_DIR}"

echo ""
echo "============================================================"
echo "Pipeline 完成！"
echo "============================================================"
echo "分析报告:  ${OUTPUT_DIR}/analysis_report.txt"
echo "可视化数据: ${OUTPUT_DIR}/viz_data.json"
echo "图表目录:  ${FIGURE_DIR}/"
echo "============================================================"
