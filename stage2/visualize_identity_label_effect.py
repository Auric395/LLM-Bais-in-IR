#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Identity Label Effect Visualization
根据 identity 测试结果绘制标签效应图
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# 设置字体
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
with open("stage2/reanalysis_modified_viz_data.json", "r", encoding="utf-8") as f:
    viz_data = json.load(f)

MODELS = ['deepseek-chat', 'qwen-max', 'gpt-5.2', 'gemini-3-flash-preview']
IDENTITY_DIMS = viz_data["identity_dims"]

# 模型颜色配置
MODEL_COLORS = {
    'deepseek-chat': '#1f77b4',  # 蓝色
    'qwen-max': '#2ca02c',       # 绿色
    'gemini-3-flash-preview': '#d62728',  # 红色
    'gpt-5.2': '#ff7f0e'         # 橙色
}

# 模型显示名称（图注中不显示具体型号）
MODEL_LABELS = {
    'deepseek-chat': 'DeepSeek',
    'qwen-max': 'Qwen',
    'gpt-5.2': 'ChatGPT',
    'gemini-3-flash-preview': 'Gemini',
}

# 维度标签（用于显示）
IDENTITY_LABELS = ['Alliance\n(Identity)', 'Political\n(Ideology)', 'Civilization', 
                   'Religion', 'Ethnicity', 'Status']

# 维度完整名称
IDENTITY_FULL_NAMES = {
    'identity': 'Alliance Label (Country/Bloc)',
    'ideology': 'Political System Label',
    'civilization': 'Civilization Label',
    'religion': 'Religion Label',
    'ethnicity': 'Ethnic Label',
    'status': 'Development and International Status Label'
}


def plot_identity_label_effect_line(save_path='stage2/figures/identity_label_effect_line.png'):
    """
    图1: 身份测试标签效应折线图
    展示 labeled - unlabeled 的差异
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(IDENTITY_DIMS))
    
    for model in MODELS:
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        effect = labeled - unlabeled
        
        ax.plot(x, effect, 'o-', linewidth=2.5, markersize=10,
                label=MODEL_LABELS[model], color=MODEL_COLORS[model], markeredgecolor='white', 
                markeredgewidth=1.5)
        
        # 在每个点添加数值标签
        for i, (xi, yi) in enumerate(zip(x, effect)):
            offset = 8 if yi >= 0 else -12
            ax.annotate(f'{yi:+.2f}',
                       xy=(xi, yi),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va='bottom' if yi >= 0 else 'top',
                       fontsize=8, color=MODEL_COLORS[model], fontweight='bold')
    
    ax.set_xlabel('Identity Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Label Effect (Labeled - Unlabeled)', fontsize=13, fontweight='bold')
    ax.set_title('Identity Test: Label Effect Analysis\n(Positive = Label Enhances Bias toward Group A)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(IDENTITY_LABELS, fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
    ax.axhspan(-0.2, 0.2, alpha=0.1, color='gray', label='Neutral Zone')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(-0.8, 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份测试标签效应折线图已保存：{save_path}")
    plt.close()


def plot_identity_label_effect_bar(save_path='stage2/figures/identity_label_effect_bar.png'):
    """
    图2: 身份测试标签效应分组柱状图
    每个维度展示4个模型的标签效应
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(IDENTITY_DIMS))
    width = 0.2
    
    for idx, model in enumerate(MODELS):
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        effect = labeled - unlabeled
        
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, effect, width, label=MODEL_LABELS[model],
                     color=MODEL_COLORS[model], alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, effect):
            height = bar.get_height()
            if abs(height) > 0.05:
                ax.annotate(f'{height:+.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height > 0 else -10),
                           textcoords="offset points",
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Identity Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Label Effect (Labeled - Unlabeled)', fontsize=13, fontweight='bold')
    ax.set_title('Identity Test: Label Effect by Dimension\n(Grouped by Model)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(IDENTITY_LABELS, fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhspan(-0.2, 0.2, alpha=0.1, color='gray')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(-0.8, 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份测试标签效应柱状图已保存：{save_path}")
    plt.close()


def plot_identity_label_effect_heatmap(save_path='stage2/figures/identity_label_effect_heatmap.png'):
    """
    图3: 身份测试标签效应热力图
    展示每个模型在每个维度的标签效应
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 构建标签效应矩阵
    effect_matrix = []
    for model in MODELS:
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        effect = labeled - unlabeled
        effect_matrix.append(effect)
    
    effect_matrix = np.array(effect_matrix)
    
    # 绘制热力图
    im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.7, vmax=0.7)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(IDENTITY_DIMS)))
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_xticklabels(IDENTITY_LABELS, fontsize=10)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    
    # 添加数值标签
    for i in range(len(MODELS)):
        for j in range(len(IDENTITY_DIMS)):
            value = effect_matrix[i, j]
            text_color = 'white' if abs(value) > 0.35 else 'black'
            ax.text(j, i, f'{value:+.2f}', ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Label Effect (Labeled - Unlabeled)', fontsize=11, fontweight='bold')
    cbar.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
    
    ax.set_title('Identity Test: Label Effect Heatmap\n(Red = Label Enhances Bias toward A, Blue = Toward B)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份测试标签效应热力图已保存：{save_path}")
    plt.close()


def plot_identity_comparison_by_model(save_path='stage2/figures/identity_comparison_by_model.png'):
    """
    图4: 按模型分组的身份测试对比图
    每个子图展示一个模型的 labeled vs unlabeled 对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x = np.arange(len(IDENTITY_DIMS))
    width = 0.35
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        
        labeled_data = viz_data["data"][model]["labeled"]["identity"]
        unlabeled_data = viz_data["data"][model]["unlabeled"]["identity"]
        
        labeled_scores = labeled_data["means"]
        labeled_stds = labeled_data["stds"]
        unlabeled_scores = unlabeled_data["means"]
        unlabeled_stds = unlabeled_data["stds"]
        
        # 计算标签效应
        effects = [l - u for l, u in zip(labeled_scores, unlabeled_scores)]
        
        bars1 = ax.bar(x - width/2, labeled_scores, width, yerr=labeled_stds,
                       label='Labeled', color='#2ca02c', alpha=0.8, 
                       edgecolor='black', linewidth=0.5,
                       capsize=4, error_kw={'linewidth': 1})
        bars2 = ax.bar(x + width/2, unlabeled_scores, width, yerr=unlabeled_stds,
                       label='Unlabeled', color='#ff7f0e', alpha=0.8, 
                       edgecolor='black', linewidth=0.5,
                       capsize=4, error_kw={'linewidth': 1})
        
        # 添加标签效应标注
        for i, (b1, b2, eff) in enumerate(zip(bars1, bars2, effects)):
            x_pos = (b1.get_x() + b2.get_x() + b2.get_width()) / 2
            y_pos = max(b1.get_height(), b2.get_height()) + 0.15
            color = '#d62728' if eff > 0 else '#1f77b4'
            ax.annotate(f'Δ={eff:+.2f}', xy=(x_pos, y_pos),
                       ha='center', va='bottom', fontsize=7, 
                       color=color, fontweight='bold')
        
        ax.set_xlabel('Identity Dimension', fontsize=10, fontweight='bold')
        ax.set_ylabel('Bias Score', fontsize=10, fontweight='bold')
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=12, fontweight='bold',
                    color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels(IDENTITY_LABELS, fontsize=8)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.suptitle('Identity Test: Labeled vs Unlabeled Comparison by Model\n'
                 '(Δ = Label Effect, Red = Label enhances bias toward A, Blue = toward B)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份测试按模型对比图已保存：{save_path}")
    plt.close()


def plot_identity_overall_effect(save_path='stage2/figures/identity_overall_label_effect.png'):
    """
    图5: 身份测试整体标签效应
    展示每个模型在身份测试上的整体标签效应
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 计算每个模型在身份测试上的平均标签效应
    model_effects = []
    for model in MODELS:
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        # 计算平均效应
        avg_effect = np.mean(labeled - unlabeled)
        model_effects.append(avg_effect)
    
    # 左图：整体标签效应柱状图
    ax1 = axes[0]
    colors = ['#d62728' if e > 0 else '#1f77b4' for e in model_effects]
    bars = ax1.bar([MODEL_LABELS[m] for m in MODELS], model_effects, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar, effect in zip(bars, model_effects):
        height = bar.get_height()
        ax1.annotate(f'{height:+.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Average Label Effect', fontsize=12, fontweight='bold')
    ax1.set_title('Identity Test: Overall Label Effect by Model\n'
                  '(Average across all identity dimensions)', 
                  fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhspan(-0.1, 0.1, alpha=0.15, color='gray', label='Neutral Zone')
    ax1.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9, rotation=15, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim(-0.5, 0.3)
    
    # 右图：各维度标签效应雷达图风格的极坐标图
    ax2 = axes[1]
    
    x = np.arange(len(IDENTITY_DIMS))
    width = 0.15
    
    for idx, model in enumerate(MODELS):
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        effect = labeled - unlabeled
        
        offset = (idx - 1.5) * width
        ax2.bar(x + offset, effect, width, label=MODEL_LABELS[model],
               color=MODEL_COLORS[model], alpha=0.8, 
               edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Identity Dimension', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Label Effect', fontsize=11, fontweight='bold')
    ax2.set_title('Identity Test: Label Effect by Dimension\n(All Models Comparison)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(IDENTITY_LABELS, fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_ylim(-0.8, 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份测试整体标签效应图已保存：{save_path}")
    plt.close()


def plot_identity_label_effect_by_dimension(save_path='stage2/figures/identity_label_effect_by_dimension.png'):
    """
    图6: 身份测试各维度标签效应（所有模型对比）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(IDENTITY_DIMS))
    width = 0.2

    for idx, model in enumerate(MODELS):
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        effect = labeled - unlabeled

        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, effect, width, label=MODEL_LABELS[model],
                     color=MODEL_COLORS[model], alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        # 添加数值标签
        for bar, val in zip(bars, effect):
            height = bar.get_height()
            if abs(height) > 0.05:
                ax.annotate(f'{height:+.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height > 0 else -10),
                           textcoords="offset points",
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=7, fontweight='bold')

    ax.set_xlabel('Identity Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Label Effect (Labeled - Unlabeled)', fontsize=13, fontweight='bold')
    ax.set_title('Identity Test: Label Effect by Dimension\n(All Models Comparison)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(IDENTITY_LABELS, fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhspan(-0.2, 0.2, alpha=0.1, color='gray', label='Neutral Zone')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(-0.8, 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份测试各维度标签效应图（所有模型对比）已保存：{save_path}")
    plt.close()


def generate_all_visualizations():
    """生成所有身份测试标签效应可视化图表"""
    import os
    os.makedirs('stage2/figures', exist_ok=True)
    
    print("=" * 60)
    print("Stage 2 Identity Test Label Effect Visualization")
    print("=" * 60)
    
    # 设置全局样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # 生成所有图表
    plot_identity_label_effect_line()
    plot_identity_label_effect_bar()
    plot_identity_label_effect_heatmap()
    plot_identity_comparison_by_model()
    plot_identity_overall_effect()
    plot_identity_label_effect_by_dimension()
    
    print("=" * 60)
    print("所有身份测试标签效应图表生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_visualizations()
