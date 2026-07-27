#!/bin/bash
# Stage 1 实验运行脚本

# 默认参数
TRIALS=5
DATA="../data/data1.json"
MODELS="../model_list.json"
OUTPUT="results"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --model-index)
            MODEL_INDEX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Model Bias Experiment - Stage 1"
echo "========================================"
echo "Trials per entity: $TRIALS"
echo "Data file: $DATA"
echo "Models file: $MODELS"
echo ""

# 运行实验
if [ -n "$MODEL_INDEX" ]; then
    echo "Running for model index: $MODEL_INDEX"
    python experiment.py --data "$DATA" --models "$MODELS" --trials "$TRIALS" --output "$OUTPUT" --model-index "$MODEL_INDEX"
else
    echo "Running for all models"
    python experiment.py --data "$DATA" --models "$MODELS" --trials "$TRIALS" --output "$OUTPUT"
fi

# 分析结果
echo ""
echo "Analyzing results..."
python analyze.py --results-dir "$OUTPUT" --output-report "$OUTPUT/analysis_report.txt" --output-viz "$OUTPUT/viz_data.json"

# 生成可视化
echo ""
echo "Generating visualizations..."
python visualize.py --input "$OUTPUT/viz_data.json" --output-dir figures

echo ""
echo "========================================"
echo "Experiment completed!"
echo "========================================"
