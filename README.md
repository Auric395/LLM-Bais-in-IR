# LLM-Bais-in-IR

本仓库为论文《中立表象：大语言模型的国际政治偏见与意义生产》的实验代码、数据与补充材料。

## 项目结构

```
.
├── README.md                          # 本文件
├── model_list.json                    # 实验使用的模型列表
├── data/                              # 实验材料与刺激文本
│   ├── data1.json
│   ├── data2.json
│   └── run_20260225_165913/           # stage2 使用的生成材料（4个模型生成）
├── stage1/                            # 阶段一：身份与叙事偏见测试
│   ├── experiment.py                  # 主实验脚本
│   ├── analyze.py                     # 结果分析
│   ├── visualize.py                   # 可视化
│   ├── requirements.txt
│   ├── run.sh
│   ├── results/                       # stage1 实验结果
│   └── figures/                       # stage1 可视化图表
├── stage2/                            # 阶段二：生成—判断分离实验
│   ├── experiment.py
│   ├── analyze.py
│   ├── visualize.py
│   ├── merge_results_by_judge.py      # 按判断模型合并原始结果
│   ├── reanalyze_modified_scoring.py  # 修正评分后的重分析
│   ├── requirements.txt
│   ├── run.sh
│   ├── results/merged_by_judge/       # stage2 合并后的判断结果（4个文件）
│   └── figures/                       # stage2 可视化图表
├── generator_judge_decomposition/     # 生成模型—判断模型分解回归
│   ├── run_generator_judge_decomposition.py  # 正式分析脚本
│   ├── 生成模型—判断模型分解回归_复现笔记本.ipynb  # 复现笔记本
│   ├── paper_ready_section_generator_judge_decomposition.md
│   ├── 生成模型—判断模型分解回归分析报告.docx
│   └── figures/                       # 分解回归表格与图表
├── scripts/                           # 辅助脚本
│   ├── generate_data2.py
│   ├── run_all.sh
│   └── run_stage2.sh
├── achieve/                           # 研究计划与进展记录
├── 实验与结果.docx                    # 论文正文（实验与结果部分）
├── 生成模型分解回归附录.docx         # 生成模型分解回归附录（最终版）
└── 附录.docx                          # 通用实验附录
```

## 数据说明

- `data/run_20260225_165913/` 包含 4 个生成模型（DeepSeek、Qwen、ChatGPT、Gemini）生成的阶段二实验材料。
- `stage2/results/merged_by_judge/` 包含按判断模型合并后的最终判断数据，每模型一个 JSON 文件，是分解回归的输入。
- `stage1/results/` 包含阶段一 4 个模型的完整判断结果。

## 主要分析流程

1. **阶段一**：运行 `stage1/run.sh` 或 `stage1/experiment.py` 完成身份与叙事偏见测试。
2. **阶段二**：运行 `stage2/run.sh` 完成生成—判断分离实验，再用 `stage2/merge_results_by_judge.py` 合并结果。
3. **分解回归**：运行 `generator_judge_decomposition/run_generator_judge_decomposition.py`，输出回归表格与图表至 `figures/`。

## 环境依赖

各阶段目录下均有 `requirements.txt`，建议分别安装：

```bash
pip install -r stage1/requirements.txt
pip install -r stage2/requirements.txt
```

分解回归脚本额外依赖 `statsmodels`、`pandas`、`matplotlib`。
