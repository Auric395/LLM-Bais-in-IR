#!/usr/bin/env bash
# Stage 2 two-phase pipeline:
#   Phase 1: Generate data (DeepSeek-chat, 3 groups)
#   Phase 2: Run experiment on each group (all models, labeled+unlabeled, 20 iterations)
#   Phase 3: Analyze + visualize
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# ---- defaults ----
NUM_GROUPS=3
ITEMS_PER_DIM=1
ITERATIONS=20
TEMPERATURE=0.7
LABEL_MODE="both"
MODE="api"
MODEL_INDEX=""
SKIP_GENERATE=false
USE_FIRST_N=4
NARRATIVE_USE_FIRST_N=1
OUTPUT_DIR="${ROOT_DIR}/stage2/results/${RUN_ID}"
FIGURE_DIR="${ROOT_DIR}/stage2/figures/${RUN_ID}"
DATA_DIR="${ROOT_DIR}/data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-groups)     NUM_GROUPS="$2"; shift 2 ;;
    --items-per-dim)  ITEMS_PER_DIM="$2"; shift 2 ;;
    --iterations)     ITERATIONS="$2"; shift 2 ;;
    --temperature)    TEMPERATURE="$2"; shift 2 ;;
    --label-mode)     LABEL_MODE="$2"; shift 2 ;;
    --mode)           MODE="$2"; shift 2 ;;
    --model-index)    MODEL_INDEX="$2"; shift 2 ;;
    --skip-generate)  SKIP_GENERATE=true; shift ;;
    --output)         OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Supported: --num-groups --items-per-dim --iterations --temperature"
      echo "           --label-mode --mode --model-index --skip-generate"
      echo "           --output --data-dir"
      exit 1 ;;
  esac
done

echo "============================================================"
echo "Stage2 Model Bias Experiment"
echo "============================================================"
echo "Mode:           ${MODE}"
echo "Num groups:     ${NUM_GROUPS}"
echo "Items per dim:  ${ITEMS_PER_DIM}"
echo "Iterations:     ${ITERATIONS}"
echo "Temperature:    ${TEMPERATURE}"
echo "Label mode:     ${LABEL_MODE}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Data dir:       ${DATA_DIR}"
echo ""

# ---- Phase 1: Data generation ----
if [[ "${SKIP_GENERATE}" == "false" ]]; then
  echo "------------------------------------------------------------"
  echo "Phase 1: Generating data with DeepSeek-chat"
  echo "------------------------------------------------------------"
  python3 "${ROOT_DIR}/scripts/generate_data2.py" \
    --input    "${ROOT_DIR}/data/data1.json" \
    --models   "${ROOT_DIR}/model_list.json" \
    --output-dir "${DATA_DIR}" \
    --num-groups "${NUM_GROUPS}" \
    --items-per-dim "${ITEMS_PER_DIM}" \
    --mode "${MODE}" \
    --use-first-n "${USE_FIRST_N}" \
    --narrative-use-first-n "${NARRATIVE_USE_FIRST_N}"
  echo "Data generation complete."
else
  echo "Skipping data generation (--skip-generate)"
fi

# ---- Phase 2: Experiment ----
echo ""
echo "------------------------------------------------------------"
echo "Phase 2: Running experiments"
echo "------------------------------------------------------------"
mkdir -p "${OUTPUT_DIR}"

EXP_ARGS=(
  python3 "${ROOT_DIR}/stage2/experiment.py"
  --models      "${ROOT_DIR}/model_list.json"
  --iterations  "${ITERATIONS}"
  --temperature "${TEMPERATURE}"
  --label-mode  "${LABEL_MODE}"
  --mode        "${MODE}"
  --output      "${OUTPUT_DIR}"
)
if [[ -n "${MODEL_INDEX}" ]]; then
  EXP_ARGS+=(--model-index "${MODEL_INDEX}")
fi

for GROUP in $(seq 1 "${NUM_GROUPS}"); do
  DATA_FILE="${DATA_DIR}/data2_group${GROUP}.json"
  if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Warning: ${DATA_FILE} not found, skipping group ${GROUP}"
    continue
  fi
  echo ""
  echo "  >> Group ${GROUP}: ${DATA_FILE}"
  "${EXP_ARGS[@]}" --data "${DATA_FILE}"
done

# ---- Phase 3: Analysis + visualization ----
echo ""
echo "------------------------------------------------------------"
echo "Phase 3: Analysis and visualization"
echo "------------------------------------------------------------"
python3 "${ROOT_DIR}/stage2/analyze.py" \
  --results-dir    "${OUTPUT_DIR}" \
  --output-report  "${OUTPUT_DIR}/analysis_report.txt" \
  --output-viz     "${OUTPUT_DIR}/viz_data.json"

mkdir -p "${FIGURE_DIR}"
python3 "${ROOT_DIR}/stage2/visualize.py" \
  --input      "${OUTPUT_DIR}/viz_data.json" \
  --output-dir "${FIGURE_DIR}"

echo ""
echo "============================================================"
echo "Stage2 completed"
echo "Report:    ${OUTPUT_DIR}/analysis_report.txt"
echo "Viz data:  ${OUTPUT_DIR}/viz_data.json"
echo "Figures:   ${FIGURE_DIR}"
echo "============================================================"
