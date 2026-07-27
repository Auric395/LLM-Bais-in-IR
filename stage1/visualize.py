"""
Stage 1: 可视化脚本
生成雷达图展示各模型在不同类别上的偏见分布
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_viz_data(filepath: str) -> Dict:
    """加载可视化数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


MODEL_LABELS = {
    'deepseek-chat': 'DeepSeek',
    'qwen-max': 'Qwen',
    'gpt-5.2': 'ChatGPT',
    'gemini-3-flash-preview': 'Gemini',
}


def plot_radar_chart(viz_data: Dict, output_path: str):
    """
    绘制雷达图：展示各模型在不同类别上的偏见分布
    """
    models = viz_data['models']
    samples = viz_data['samples']
    matrix = np.array(viz_data['scores_matrix'])

    # 雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(samples), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        values = matrix[i].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=MODEL_LABELS.get(model, model), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(samples, fontsize=10)
    ax.set_ylim(-1, 1)
    ax.set_title('Model Bias Radar Chart', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate radar chart for Stage 1 results')
    parser.add_argument('--input', type=str, default='viz_data.json', help='Input visualization data file')
    parser.add_argument('--output-dir', type=str, default='figures', help='Output directory for figures')

    args = parser.parse_args()

    viz_data = load_viz_data(args.input)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    plot_radar_chart(viz_data, os.path.join(args.output_dir, 'radar_chart.png'))


if __name__ == '__main__':
    main()
