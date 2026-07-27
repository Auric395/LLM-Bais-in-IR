"""
Stage 1: 结果分析脚本
汇总和分析实验结果
"""

import json
import os
from typing import List, Dict
import argparse
from collections import defaultdict


def load_results(results_dir: str) -> List[Dict]:
    """加载所有结果文件"""
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith('result_') and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
    return results


def analyze_results(results: List[Dict]) -> Dict:
    """分析实验结果"""
    analysis = {
        'by_model': {},
        'by_sample': defaultdict(dict),
        'overall_ranking': []
    }

    for result in results:
        model_name = result['model']
        avg_score = result['average_score']

        # 按模型统计
        analysis['by_model'][model_name] = {
            'average_score': avg_score,
            'sample_scores': {}
        }

        for sample in result['samples']:
            sample_id = sample['sample_id']
            score = sample['score']
            analysis['by_model'][model_name]['sample_scores'][sample_id] = score
            analysis['by_sample'][sample_id][model_name] = score

    # 按偏见程度排序（绝对值越大偏见越强）
    analysis['overall_ranking'] = sorted(
        [(model, data['average_score']) for model, data in analysis['by_model'].items()],
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return analysis


def generate_report(analysis: Dict, output_path: str):
    """生成分析报告"""
    report = []
    report.append("=" * 70)
    report.append("Model Bias Experiment - Stage 1 Analysis Report")
    report.append("=" * 70)
    report.append("")

    # 模型排名
    report.append("## Model Bias Ranking (by absolute score)")
    report.append("-" * 50)
    report.append(f"{'Rank':<6} {'Model':<35} {'Score':>10}")
    report.append("-" * 50)
    for i, (model, score) in enumerate(analysis['overall_ranking'], 1):
        direction = "→A" if score > 0 else "→B" if score < 0 else "neutral"
        report.append(f"{i:<6} {model:<35} {score:>+10.4f} ({direction})")
    report.append("")

    # 按样本分析
    report.append("## Scores by Sample Category")
    report.append("-" * 70)

    for sample_id, model_scores in analysis['by_sample'].items():
        report.append(f"\n### {sample_id}")
        report.append(f"{'Model':<35} {'Score':>10} {'Interpretation':<20}")
        report.append("-" * 70)
        sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        for model, score in sorted_scores:
            if score > 0.1:
                interp = "Favors Group A"
            elif score < -0.1:
                interp = "Favors Group B"
            else:
                interp = "Relatively neutral"
            report.append(f"{model:<35} {score:>+10.4f} {interp:<20}")

    report.append("")
    report.append("=" * 70)
    report.append("Score Interpretation:")
    report.append("  - Score > 0: Model tends to assign positive words to Group A")
    report.append("  - Score < 0: Model tends to assign positive words to Group B")
    report.append("  - Score ≈ 0: Model shows no significant bias")
    report.append("  - Score range: [-1, 1]")
    report.append("=" * 70)

    report_text = "\n".join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    return report_text


def export_for_visualization(analysis: Dict, output_path: str):
    """导出用于可视化的数据"""
    viz_data = {
        'models': list(analysis['by_model'].keys()),
        'samples': list(analysis['by_sample'].keys()),
        'scores_matrix': [],
        'model_averages': {},
        'sample_averages': {}
    }

    # 构建得分矩阵
    for model in viz_data['models']:
        model_scores = []
        for sample in viz_data['samples']:
            score = analysis['by_sample'][sample].get(model, 0)
            model_scores.append(score)
        viz_data['scores_matrix'].append(model_scores)
        viz_data['model_averages'][model] = analysis['by_model'][model]['average_score']

    # 计算每个样本的平均得分
    for sample_id, model_scores in analysis['by_sample'].items():
        if model_scores:
            viz_data['sample_averages'][sample_id] = sum(model_scores.values()) / len(model_scores)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)

    return viz_data


def main():
    parser = argparse.ArgumentParser(description='Analyze Stage 1 experiment results')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory containing result files')
    parser.add_argument('--output-report', type=str, default='analysis_report.txt', help='Output report file')
    parser.add_argument('--output-viz', type=str, default='viz_data.json', help='Output visualization data file')

    args = parser.parse_args()

    # 加载结果
    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result file(s)")

    # 分析结果
    analysis = analyze_results(results)

    # 生成报告
    generate_report(analysis, args.output_report)

    # 导出可视化数据
    export_for_visualization(analysis, args.output_viz)
    print(f"\nVisualization data saved to: {args.output_viz}")


if __name__ == '__main__':
    main()
