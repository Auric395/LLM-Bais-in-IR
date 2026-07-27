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
#   bash scripts/run_all.sh                                    # 默认全流程
#   bash scripts/run_all.sh --mode mock                        # mock 模式
#   bash scripts/run_all.sh --skip-gen --data-dir data/run_20260225_154609
#                                                              # 跳过生成，复用已有数据
#   bash scripts/run_all.sh --skip-gen --data-dir data/run_20260225_154609 \
#       --apis deepseek-chat,qwen-max                          # 只用指定模型测试
#   bash scripts/run_all.sh --skip-gen --data-dir data/run_20260225_154609 \
#       --gen-data data2_gen1_deepseek-chat.json,data2_gen2_qwen-max.json
#                                                              # 只测试指定的生成数据
#
# 参数：
#   --mode api|mock          运行模式（默认: api）
#   --temperature FLOAT      温度参数（默认: 0.7）
#   --iterations INT         迭代次数（默认: 10）
#   --skip-gen               跳过数据生成阶段，需配合 --data-dir 使用
#   --data-dir DIR           指定已有数据目录（隐含 --skip-gen）
#   --apis MODEL1,MODEL2     逗号分隔，仅用这些模型做测试（默认: 全部）
#   --gen-data FILE1,FILE2   逗号分隔，仅测试这些生成数据文件（默认: 目录下全部）
#   --skip-analysis          跳过分析和可视化阶段
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# ---- 默认参数 ----
MODE="api"
TEMPERATURE="0.7"
ITERATIONS="10"
LABEL_MODE="both"
SKIP_GEN=false
SKIP_ANALYSIS=false
DATA_DIR=""
SELECTED_APIS=""
SELECTED_GEN_DATA=""

# ---- 解析命令行参数 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --temperature)
      TEMPERATURE="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;
    --skip-gen)
      SKIP_GEN=true; shift ;;
    --data-dir)
      DATA_DIR="$2"; SKIP_GEN=true; shift 2 ;;
    --apis)
      SELECTED_APIS="$2"; shift 2 ;;
    --gen-data)
      SELECTED_GEN_DATA="$2"; shift 2 ;;
    --skip-analysis)
      SKIP_ANALYSIS=true; shift ;;
    -h|--help)
      head -30 "$0" | tail -25
      exit 0 ;;
    *)
      # 兼容旧的位置参数: run_all.sh [mode] [temperature] [iterations]
      if [[ -z "${MODE_SET:-}" ]]; then
        MODE="$1"; MODE_SET=1; shift
      elif [[ -z "${TEMP_SET:-}" ]]; then
        TEMPERATURE="$1"; TEMP_SET=1; shift
      elif [[ -z "${ITER_SET:-}" ]]; then
        ITERATIONS="$1"; ITER_SET=1; shift
      else
        echo "错误: 未知参数 '$1'" >&2; exit 1
      fi
      ;;
  esac
done

# ---- 确定数据目录 ----
if [[ "${SKIP_GEN}" == true ]]; then
  if [[ -z "${DATA_DIR}" ]]; then
    echo "错误: --skip-gen 需要配合 --data-dir 指定已有数据目录" >&2
    exit 1
  fi
  # 支持相对路径和绝对路径
  if [[ "${DATA_DIR}" != /* ]]; then
    DATA_DIR="${ROOT_DIR}/${DATA_DIR}"
  fi
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "错误: 数据目录不存在: ${DATA_DIR}" >&2
    exit 1
  fi
else
  DATA_DIR="${ROOT_DIR}/data/run_${RUN_ID}"
fi

OUTPUT_DIR="${ROOT_DIR}/stage2/results/${RUN_ID}"
FIGURE_DIR="${ROOT_DIR}/stage2/figures/${RUN_ID}"

# ---- 构建测试用模型列表 ----
MODELS_FILE="${ROOT_DIR}/model_list.json"
if [[ -n "${SELECTED_APIS}" ]]; then
  # 生成临时的模型列表文件，只包含选中的模型
  FILTERED_MODELS_FILE="${ROOT_DIR}/.tmp_model_list_${RUN_ID}.json"
  python3 -c "
import json, sys
with open('${MODELS_FILE}') as f:
    models = json.load(f)
selected = [s.strip() for s in '${SELECTED_APIS}'.split(',')]
filtered = [m for m in models if m['name'] in selected]
missing = set(selected) - {m['name'] for m in filtered}
if missing:
    print(f'警告: 以下模型未在 model_list.json 中找到: {missing}', file=sys.stderr)
if not filtered:
    print('错误: 没有匹配的模型', file=sys.stderr)
    sys.exit(1)
with open('${FILTERED_MODELS_FILE}', 'w') as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)
print(f'已选择 {len(filtered)} 个模型: {[m[\"name\"] for m in filtered]}')
"
  MODELS_FILE="${FILTERED_MODELS_FILE}"
  trap "rm -f '${FILTERED_MODELS_FILE}'" EXIT
fi

echo "============================================================"
echo "Stage2 Pipeline: 对应词配对 × 模型生成 × 披露标签测试"
echo "============================================================"
echo "Run ID:         ${RUN_ID}"
echo "Mode:           ${MODE}"
echo "Temperature:    ${TEMPERATURE}"
echo "Iterations:     ${ITERATIONS}"
echo "Label mode:     ${LABEL_MODE}"
echo "Skip gen:       ${SKIP_GEN}"
echo "Data dir:       ${DATA_DIR}"
echo "Results dir:    ${OUTPUT_DIR}"
echo "Figures dir:    ${FIGURE_DIR}"
if [[ -n "${SELECTED_APIS}" ]]; then
  echo "测试模型:       ${SELECTED_APIS}"
fi
if [[ -n "${SELECTED_GEN_DATA}" ]]; then
  echo "指定数据文件:   ${SELECTED_GEN_DATA}"
fi
echo ""

# ---- Phase 1: 数据生成（四模型 × 对应词配对） ----
if [[ "${SKIP_GEN}" == true ]]; then
  echo "------------------------------------------------------------"
  echo "Phase 1: [已跳过] 使用已有数据: ${DATA_DIR}"
  echo "------------------------------------------------------------"
  echo "已有数据文件："
  ls -la "${DATA_DIR}"/data2_gen*.json 2>/dev/null || echo "  (未找到 data2_gen*.json 文件)"
else
  echo "------------------------------------------------------------"
  echo "Phase 1: 四模型分别生成数据（对应词配对 A[i]↔B[i]）"
  echo "------------------------------------------------------------"
  python3 "${ROOT_DIR}/scripts/generate_data2.py" \
    --input "${ROOT_DIR}/data/data1.json" \
    --models "${MODELS_FILE}" \
    --output-dir "${DATA_DIR}" \
    --pairing corresponding \
    --all-generators \
    --mode "${MODE}"

  echo ""
  echo "数据生成完成，文件列表："
  ls -la "${DATA_DIR}"/data2_gen*.json 2>/dev/null || echo "  (未找到文件)"
fi

# ---- Phase 2: 实验（全模型 × 全数据集 × labeled+unlabeled） ----
echo ""
echo "------------------------------------------------------------"
echo "Phase 2: 披露标签测试"
echo "         每组数据 × 测试模型 × labeled/unlabeled × ${ITERATIONS}次迭代"
echo "         温度固定: ${TEMPERATURE}"
echo "------------------------------------------------------------"
mkdir -p "${OUTPUT_DIR}"

# 确定要测试的数据文件列表
DATA_FILES=()
if [[ -n "${SELECTED_GEN_DATA}" ]]; then
  # 使用指定的数据文件
  IFS=',' read -ra GEN_DATA_NAMES <<< "${SELECTED_GEN_DATA}"
  for NAME in "${GEN_DATA_NAMES[@]}"; do
    NAME="$(echo "${NAME}" | xargs)"  # trim whitespace
    FULL_PATH="${DATA_DIR}/${NAME}"
    if [[ -f "${FULL_PATH}" ]]; then
      DATA_FILES+=("${FULL_PATH}")
    else
      echo "警告: 数据文件不存在，跳过: ${FULL_PATH}" >&2
    fi
  done
  if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
    echo "错误: 指定的数据文件均不存在" >&2
    exit 1
  fi
else
  # 使用目录下所有 data2_gen*.json
  for f in "${DATA_DIR}"/data2_gen*.json; do
    [[ -f "$f" ]] && DATA_FILES+=("$f")
  done
fi

DATASET_COUNT=0
TOTAL=${#DATA_FILES[@]}
for DATA_FILE in "${DATA_FILES[@]}"; do
  DATASET_COUNT=$((DATASET_COUNT + 1))
  echo ""
  echo "  >> [${DATASET_COUNT}/${TOTAL}] 数据集: $(basename "${DATA_FILE}")"
  python3 "${ROOT_DIR}/stage2/experiment.py" \
    --data "${DATA_FILE}" \
    --models "${MODELS_FILE}" \
    --iterations "${ITERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --label-mode "${LABEL_MODE}" \
    --mode "${MODE}" \
    --output "${OUTPUT_DIR}"
done

echo ""
echo "实验完成，共处理 ${DATASET_COUNT} 个数据集"

# ---- Phase 3: 分析 + 可视化 ----
if [[ "${SKIP_ANALYSIS}" == true ]]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Phase 3: [已跳过] 分析和可视化"
  echo "------------------------------------------------------------"
else
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
fi

echo ""
echo "============================================================"
echo "Pipeline 完成！"
echo "============================================================"
echo "结果目录:   ${OUTPUT_DIR}"
if [[ "${SKIP_ANALYSIS}" != true ]]; then
  echo "分析报告:   ${OUTPUT_DIR}/analysis_report.txt"
  echo "可视化数据: ${OUTPUT_DIR}/viz_data.json"
  echo "图表目录:   ${FIGURE_DIR}/"
fi
echo "============================================================"
