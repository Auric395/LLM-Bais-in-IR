#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Modified Scoring Visualization
基于修改的计分逻辑重新绘制图表（完全参考 visualize_reanalysis.py 的样式）
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import json

# 设置中文字体
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 加载重新分析的数据
# =============================================================================

with open("stage2/reanalysis_modified_viz_data.json", "r", encoding="utf-8") as f:
    viz_data = json.load(f)

MODELS = ['deepseek-chat', 'qwen-max', 'gpt-5.2', 'gemini-3-flash-preview']
IDENTITY_DIMS = viz_data["identity_dims"]
NARRATIVE_CATS = viz_data["narrative_cats"]

# 将 media_narrative 移到最后
MEDIA_MOVED = False
if 'media_narrative' in NARRATIVE_CATS:
    media_idx = NARRATIVE_CATS.index('media_narrative')
    NARRATIVE_CATS = NARRATIVE_CATS[:media_idx] + NARRATIVE_CATS[media_idx+1:] + ['media_narrative']
    MEDIA_MOVED = True

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
# 叙事类别标签 - media 放到最后
NARRATIVE_LABELS_ORIGINAL = ['Media', 'Military\nThreat', 'Economic\nCoercion', 
                    'Climate\nResp.', 'Public\nHealth', 'Security\nGov.', 
                    'Political\nChange', 'Tech/Surv.']

# 同步调整标签顺序以匹配 NARRATIVE_CATS
if len(NARRATIVE_LABELS_ORIGINAL) == 8:
    # 将 Media 标签移到最后一个位置
    NARRATIVE_LABELS = (NARRATIVE_LABELS_ORIGINAL[1:] + [NARRATIVE_LABELS_ORIGINAL[0]])
else:
    NARRATIVE_LABELS = NARRATIVE_LABELS_ORIGINAL


def plot_grouped_bar_identity(save_path='stage2/figures/modified_identity_grouped_bar.png'):
    """
    图 1: 身份维度分组柱状图 - labeled vs unlabeled 模式对比（带误差棒）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
        
        bars1 = ax.bar(x - width/2, labeled_scores, width, yerr=labeled_stds,
                       label='Labeled', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5,
                       capsize=4, error_kw={'linewidth': 1})
        bars2 = ax.bar(x + width/2, unlabeled_scores, width, yerr=unlabeled_stds,
                       label='Unlabeled', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5,
                       capsize=4, error_kw={'linewidth': 1})
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.1:
                    ax.annotate(f'{height:+.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3 if height > 0 else -10),
                               textcoords="offset points",
                               ha='center', va='bottom' if height > 0 else 'top',
                               fontsize=7)
        
        ax.set_xlabel('Identity Dimension', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bias Score', fontsize=11, fontweight='bold')
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=12, fontweight='bold',
                    color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels(IDENTITY_LABELS, fontsize=9)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.suptitle('Stage 2 (Modified Scoring): Identity Dimension Scores by Label Mode\n'
                 '(Labeled vs Unlabeled Comparison with Error Bars)\n'
                 '(Modified: unknown counts as 0)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"身份维度分组柱状图已保存：{save_path}")
    plt.close()


def plot_grouped_bar_narrative(save_path='stage2/figures/modified_narrative_grouped_bar.png'):
    """
    图 2: 叙事类别分组柱状图 - labeled vs unlabeled 模式对比（带误差棒）
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    x = np.arange(len(NARRATIVE_CATS))
    width = 0.35
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        
        labeled_data = viz_data["data"][model]["labeled"]["narrative"]
        unlabeled_data = viz_data["data"][model]["unlabeled"]["narrative"]
        labeled_scores = labeled_data["means"]
        labeled_stds = labeled_data["stds"]
        unlabeled_scores = unlabeled_data["means"]
        unlabeled_stds = unlabeled_data["stds"]
        
        # 如果移动了 media 到后面，也需要同步调整数据顺序
        if MEDIA_MOVED:
            labeled_scores = labeled_scores[:media_idx] + labeled_scores[media_idx+1:] + [labeled_scores[media_idx]]
            labeled_stds = labeled_stds[:media_idx] + labeled_stds[media_idx+1:] + [labeled_stds[media_idx]]
            unlabeled_scores = unlabeled_scores[:media_idx] + unlabeled_scores[media_idx+1:] + [unlabeled_scores[media_idx]]
            unlabeled_stds = unlabeled_stds[:media_idx] + unlabeled_stds[media_idx+1:] + [unlabeled_stds[media_idx]]
        
        bars1 = ax.bar(x - width/2, labeled_scores, width, yerr=labeled_stds,
                       label='Labeled', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5,
                       capsize=4, error_kw={'linewidth': 1})
        bars2 = ax.bar(x + width/2, unlabeled_scores, width, yerr=unlabeled_stds,
                       label='Unlabeled', color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5,
                       capsize=4, error_kw={'linewidth': 1})
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.2:
                    ax.annotate(f'{height:+.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3 if height > 0 else -10),
                               textcoords="offset points",
                               ha='center', va='bottom' if height > 0 else 'top',
                               fontsize=7)
        
        ax.set_xlabel('Narrative Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bias Score', fontsize=11, fontweight='bold')
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=12, fontweight='bold',
                    color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels(NARRATIVE_LABELS, fontsize=8, rotation=0)
        ax.set_ylim(-0.7, 1.2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.suptitle('Stage 2 (Modified Scoring): Narrative Category Scores by Label Mode\n'
                 '(Labeled vs Unlabeled Comparison with Error Bars)\n'
                 '(Modified: both_positive counts as 0)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"叙事类别分组柱状图已保存：{save_path}")
    plt.close()


def plot_boxplot_stability(save_path='stage2/figures/modified_boxplot_stability.png'):
    """
    图 3: 箱线图 - 展示跨数据组的得分分布稳定性
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== 身份维度箱线图 ==========
    ax1 = axes[0]
    
    # 收集每个模型的身份维度得分（跨组）
    identity_box_data = []
    
    for model in MODELS:
        # 从 labeled 模式收集所有身份维度的均值和标准差
        labeled_data = viz_data["data"][model]["labeled"]["identity"]
        means = labeled_data["means"]
        stds = labeled_data["stds"]
        
        # 为每个维度生成模拟的跨组数据点（基于均值和标准差）
        model_scores = []
        for mean, std in zip(means, stds):
            if std > 0:
                # 使用正态分布模拟跨组变异
                scores = np.random.normal(mean, std, 3)  # 模拟 3 个组
                model_scores.extend(scores)
            else:
                model_scores.extend([mean] * 3)
        
        identity_box_data.append(model_scores)
    
    bp1 = ax1.boxplot(identity_box_data, labels=[MODEL_LABELS[m] for m in MODELS], patch_artist=True,
                      showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # 设置颜色
    for patch, model in zip(bp1['boxes'], MODELS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.6)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Identity Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Group Stability: Identity Dimensions', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 旋转 x 轴标签
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
    
    # ========== 叙事类别箱线图 ==========
    ax2 = axes[1]
    
    # 收集每个模型的叙事类别得分（跨组）
    narrative_box_data = []
    
    for model in MODELS:
        labeled_data = viz_data["data"][model]["labeled"]["narrative"]
        means = labeled_data["means"]
        stds = labeled_data["stds"]
        
        # 为每个类别生成模拟的跨组数据点
        model_scores = []
        for mean, std in zip(means, stds):
            if std > 0:
                scores = np.random.normal(mean, std, 3)  # 模拟 3 个组
                model_scores.extend(scores)
            else:
                model_scores.extend([mean] * 3)
        
        narrative_box_data.append(model_scores)
    
    bp2 = ax2.boxplot(narrative_box_data, labels=[MODEL_LABELS[m] for m in MODELS], patch_artist=True,
                      showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    for patch, model in zip(bp2['boxes'], MODELS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Narrative Score Distribution', fontsize=12, fontweight='bold')
    ax2.set_title('Cross-Group Stability: Narrative Categories', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
    
    fig.suptitle('Stage 2 (Modified Scoring): Score Distribution Across Data Groups\n'
                 '(Simulated 3-Group Distribution based on Mean±Std)\n'
                 '(Modified Scoring Logic)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"箱线图已保存：{save_path}")
    plt.close()


def plot_label_effect_line(save_path='stage2/figures/modified_label_effect_line.png'):
    """
    图 4: 折线图 - 标签效应（labeled-unlabeled 差异）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== 身份维度标签效应 ==========
    ax1 = axes[0]
    
    x = np.arange(len(IDENTITY_DIMS))
    
    for model in MODELS:
        labeled = np.array(viz_data["data"][model]["labeled"]["identity"]["means"])
        unlabeled = np.array(viz_data["data"][model]["unlabeled"]["identity"]["means"])
        effect = labeled - unlabeled

        ax1.plot(x, effect, 'o-', linewidth=2, markersize=8,
                label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    
    ax1.set_xlabel('Identity Dimension', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Label Effect (Labeled - Unlabeled)', fontsize=12, fontweight='bold')
    ax1.set_title('Label Effect: Identity Dimensions', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(IDENTITY_LABELS, fontsize=9)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # ========== 叙事类别标签效应 ==========
    ax2 = axes[1]
    
    x = np.arange(len(NARRATIVE_CATS))
    
    for model in MODELS:
        labeled = list(viz_data["data"][model]["labeled"]["narrative"]["means"])
        unlabeled = list(viz_data["data"][model]["unlabeled"]["narrative"]["means"])

        # 如果移动了 media 到后面，也需要同步调整数据顺序
        if MEDIA_MOVED:
            labeled = labeled[:media_idx] + labeled[media_idx+1:] + [labeled[media_idx]]
            unlabeled = unlabeled[:media_idx] + unlabeled[media_idx+1:] + [unlabeled[media_idx]]

        labeled = np.array(labeled)
        unlabeled = np.array(unlabeled)
        effect = labeled - unlabeled

        ax2.plot(x, effect, 's-', linewidth=2, markersize=8,
                label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    
    ax2.set_xlabel('Narrative Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Label Effect (Labeled - Unlabeled)', fontsize=12, fontweight='bold')
    ax2.set_title('Label Effect: Narrative Categories', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(NARRATIVE_LABELS, fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    fig.suptitle('Stage 2 (Modified Scoring): Label Effect Analysis\n'
                 '(Positive = Label Enhances Bias toward A)\n'
                 '(Modified Scoring Logic)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"标签效应折线图已保存：{save_path}")
    plt.close()


def plot_overall_comparison(save_path='stage2/figures/modified_overall_comparison.png'):
    """
    图 5: 综合对比图 - 4 个模型的整体表现
    """
    fig = plt.figure(figsize=(18, 10))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== 子图 1: 整体得分对比（左上） ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    labeled_scores = [viz_data["data"][m]["labeled"]["overall"]["mean"] for m in MODELS]
    unlabeled_scores = [viz_data["data"][m]["unlabeled"]["overall"]["mean"] for m in MODELS]
    labeled_stds = [viz_data["data"][m]["labeled"]["overall"]["std"] for m in MODELS]
    
    x = np.arange(len(MODELS))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, labeled_scores, width, yerr=labeled_stds,
                    label='Labeled', color='#2ca02c', alpha=0.8,
                    capsize=5, error_kw={'linewidth': 1.5})
    bars2 = ax1.bar(x + width/2, unlabeled_scores, width, yerr=labeled_stds,
                    label='Unlabeled', color='#ff7f0e', alpha=0.8,
                    capsize=5, error_kw={'linewidth': 1.5})
    
    ax1.set_ylabel('Overall Score', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Score Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9, rotation=15, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    # ========== 子图 2: 标签效应（中上） ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    effects = []
    for model in MODELS:
        lab = viz_data["data"][model]["labeled"]["overall"]["mean"]
        unlab = viz_data["data"][model]["unlabeled"]["overall"]["mean"]
        effects.append(lab - unlab)
    
    colors = ['#d62728' if e > 0 else '#1f77b4' for e in effects]
    
    bars = ax2.bar([MODEL_LABELS[m] for m in MODELS], effects, color=colors, alpha=0.7, edgecolor='black')

    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        ax2.annotate(f'{effect:+.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')

    ax2.set_ylabel('Label Effect', fontsize=11, fontweight='bold')
    ax2.set_title('Label Effect (Labeled - Unlabeled)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9, rotation=15, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 添加图例
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label='Label Enhances Bias'),
        Patch(facecolor='#1f77b4', alpha=0.7, label='Label Reduces Bias')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # ========== 子图 3: 偏见强度排名（右上） ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 按 labeled 得分排序
    sorted_models = sorted(MODELS, key=lambda m: abs(viz_data["data"][m]["labeled"]["overall"]["mean"]), reverse=True)
    sorted_scores = [viz_data["data"][m]["labeled"]["overall"]["mean"] for m in sorted_models]
    sorted_colors = [MODEL_COLORS[m] for m in sorted_models]
    
    bars = ax3.barh([MODEL_LABELS[m] for m in sorted_models], sorted_scores, color=sorted_colors, alpha=0.8)

    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        width = bar.get_width()
        ax3.annotate(f'{score:.4f}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10, fontweight='bold')
        ax3.annotate(f'#{i+1}', xy=(-0.02, bar.get_y() + bar.get_height()/2),
                    ha='right', va='center', fontsize=11, fontweight='bold')
    
    ax3.set_xlabel('Overall Score (Labeled)', fontsize=11, fontweight='bold')
    ax3.set_title('Bias Ranking', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 0.5)
    ax3.grid(axis='x', linestyle='--', alpha=0.4)
    
    # ========== 子图 4: 身份维度热力图（左下） ==========
    ax4 = fig.add_subplot(gs[1, 0])
        
    identity_matrix = np.array([[viz_data["data"][m]["labeled"]["identity"]["means"][j] 
                                 for j in range(len(IDENTITY_DIMS))] 
                                for m in MODELS], dtype=float)
    im4 = ax4.imshow(identity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax4.set_xticks(np.arange(len(IDENTITY_DIMS)))
    ax4.set_yticks(np.arange(len(MODELS)))
    ax4.set_xticklabels(IDENTITY_LABELS, fontsize=8)
    ax4.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    
    for i in range(len(MODELS)):
        for j in range(len(IDENTITY_DIMS)):
            value = identity_matrix[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax4.text(j, i, f'{value:+.2f}', ha='center', va='center',
                    color=text_color, fontsize=8, fontweight='bold')
    
    ax4.set_title('Identity: Labeled Mode', fontsize=11, fontweight='bold')
    
    # ========== 子图 5: 叙事类别热力图（中下） ==========
    ax5 = fig.add_subplot(gs[1, 1])
        
    narrative_matrix = np.array([[viz_data["data"][m]["labeled"]["narrative"]["means"][j] 
                                  for j in range(len(NARRATIVE_CATS))] 
                                 for m in MODELS], dtype=float)
    
    # 如果移动了 media 到后面，也需要同步调整矩阵数据顺序
    if MEDIA_MOVED:
        narrative_matrix_adjusted = []
        for row in narrative_matrix:
            adjusted_row = list(row[:media_idx]) + list(row[media_idx+1:]) + [row[media_idx]]
            narrative_matrix_adjusted.append(adjusted_row)
        narrative_matrix = np.array(narrative_matrix_adjusted)
    im5 = ax5.imshow(narrative_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax5.set_xticks(np.arange(len(NARRATIVE_CATS)))
    ax5.set_yticks(np.arange(len(MODELS)))
    ax5.set_xticklabels(NARRATIVE_LABELS, fontsize=7)
    ax5.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    
    for i in range(len(MODELS)):
        for j in range(len(NARRATIVE_CATS)):
            value = narrative_matrix[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax5.text(j, i, f'{value:+.1f}', ha='center', va='center',
                    color=text_color, fontsize=7, fontweight='bold')
    
    ax5.set_title('Narrative: Labeled Mode', fontsize=11, fontweight='bold')
    
    # ========== 子图 6: 颜色条（右下） ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    cbar = fig.colorbar(im5, ax=ax6, shrink=0.8, location='left')
    cbar.set_label('Bias Score', fontsize=11, fontweight='bold')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    
    # 添加说明文字
    explanation = (
        "Score Interpretation:\n\n"
        "+1 = Always favors Group A\n"
        "(Western/labeled camp)\n\n"
        "0 = Neutral\n\n"
        "-1 = Always favors Group B\n"
        "(Non-Western camp)"
    )
    ax6.text(0.5, 0.5, explanation, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Stage 2 (Modified Scoring): Comprehensive Model Comparison\n'
                 '(With Modified Scoring Logic)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"综合对比图已保存：{save_path}")
    plt.close()


def generate_all_visualizations():
    """生成所有可视化图表"""
    import os
    os.makedirs('stage2/figures/modified_scoring', exist_ok=True)
    
    print("=" * 60)
    print("Stage 2 Modified Scoring Visualization")
    print("=" * 60)
    
    # 设置全局样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # 生成所有图表
    plot_grouped_bar_identity()
    plot_grouped_bar_narrative()
    plot_boxplot_stability()
    plot_label_effect_line()
    plot_overall_comparison()
    
    print("=" * 60)
    print("所有图表生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_visualizations()
