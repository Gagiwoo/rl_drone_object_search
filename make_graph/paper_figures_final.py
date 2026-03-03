"""
논문용 최종 그래프 생성 코드 (타이틀 수정 버전)
- Figure 1: Uniform Distribution, Medium Clustering, Strong Clustering
- Figure 2-4: 기존 타이틀 유지
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ========== 설정 ==========
COMMON_STEPS = 60000
SMOOTHING_WINDOW = 101
SMOOTHING_POLYORDER = 3

# 시나리오별 색상 매핑
SCENARIO_COLORS = {
    'Uniform': '#E74C3C',    # Red
    'Medium': '#F39C12',     # Orange
    'Strong': '#9B59B6'      # Purple
}

# ========== 데이터 로드 ==========
def load_data():
    """CSV 파일 로드"""
    success_df = pd.read_csv('make_graph/wandb_export_2025-12-30T23_20_06.131+09_00.csv')
    ep_len_df = pd.read_csv('make_graph/wandb_export_2025-12-30T23_20_22.746+09_00.csv')
    ep_rew_df = pd.read_csv('make_graph/wandb_export_2025-12-30T23_20_12.948+09_00.csv')
    print("✓ 데이터 로드 완료")
    return success_df, ep_len_df, ep_rew_df

# ========== Smoothing 함수 ==========
def smooth_curve(steps, values, window_length=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER):
    """Savitzky-Golay 필터 적용"""
    mask = ~np.isnan(values)
    steps_clean = steps[mask]
    values_clean = values[mask]

    if len(values_clean) > window_length:
        smoothed = savgol_filter(values_clean, window_length, polyorder)
        return steps_clean, smoothed
    return steps_clean, values_clean

# ========== Figure 1: Success Rate 통합 ==========
def create_figure1_success_rate(success_df):
    """
    Figure 1: 3개 시나리오 Success Rate 비교
    타이틀: Uniform Distribution, Medium Clustering, Strong Clustering
    """
    scenarios = [
        ('Uniform', 'Proposed_Uniform', 'Prior_Uniform', 'Uniform Distribution'),
        ('Medium', 'Proposed_medium2', 'Prior_medium2', 'Medium Clustering'),
        ('Strong', 'Proposed_hard', 'Prior_Strong', 'Strong Clustering')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle('Complete Success Rate Comparison', fontsize=16, fontweight='bold', y=0.96)

    for idx, (scenario_name, proposed_col, prior_col, title_name) in enumerate(scenarios):
        ax = axes[idx]
        color = SCENARIO_COLORS[scenario_name]

        proposed_col_full = f'{proposed_col} - rollout/success_rate'
        prior_col_full = f'{prior_col} - rollout/success_rate'

        df_filtered = success_df[success_df['Step'] <= COMMON_STEPS].copy()
        proposed_data = df_filtered[['Step', proposed_col_full]].dropna()
        prior_data = df_filtered[['Step', prior_col_full]].dropna()

        if len(proposed_data) > 0 and len(prior_data) > 0:
            # 원본 데이터
            ax.plot(proposed_data['Step'], proposed_data[proposed_col_full], 
                   alpha=0.15, color=color, linewidth=0.8)
            ax.plot(prior_data['Step'], prior_data[prior_col_full], 
                   alpha=0.15, color='#3498DB', linewidth=0.8)

            # Smoothing
            prop_steps_smooth, prop_values_smooth = smooth_curve(
                proposed_data['Step'].values, 
                proposed_data[proposed_col_full].values
            )
            prior_steps_smooth, prior_values_smooth = smooth_curve(
                prior_data['Step'].values, 
                prior_data[prior_col_full].values
            )

            # Smoothed 라인
            ax.plot(prop_steps_smooth, prop_values_smooth, 
                   label='Proposed', color=color, linewidth=2.5)
            ax.plot(prior_steps_smooth, prior_values_smooth, 
                   label='Baseline', color='#3498DB', linewidth=2.5, linestyle='--')

        # Subfigure 라벨
        ax.text(-0.15, 1.0, f'({chr(97+idx)})', transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

        ax.set_xlabel('Environment Steps', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_xlim(0, COMMON_STEPS)
        ax.set_ylim(0, 1.0)

        # Legend
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

        ax.grid(True, alpha=0.3, linestyle='--')
        # 타이틀 변경: Uniform Distribution, Medium Clustering, Strong Clustering
        ax.set_title(title_name, fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks([0, 15000, 30000, 45000, 60000])
        ax.set_xticklabels(['0', '15k', '30k', '45k', '60k'])

    plt.tight_layout()
    plt.savefig('Figure1_success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure1_success_rate_comparison.pdf', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 저장: Figure1_success_rate_comparison.png, .pdf")
    plt.close()

# ========== Figure 2-4: Reward & Length (시나리오별) ==========
def create_figure_reward_length(scenario_name, proposed_col, prior_col, 
                                 ep_rew_df, ep_len_df, figure_num):
    """
    Figure 2-4: 시나리오별 Reward & Length 비교
    텍스트 박스 없이 Legend만 표시
    """
    color = SCENARIO_COLORS[scenario_name]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # 제목 설정 (기존 유지)
    title_map = {
        'Uniform': 'Uniform Distribution',
        'Medium': 'Medium Clustering',
        'Strong': 'Strong Clustering'
    }
    fig.suptitle(title_map.get(scenario_name, scenario_name), 
                 fontsize=16, fontweight='bold', y=0.96)

    metrics = [
        ('ep_rew_mean', ep_rew_df, 'Episode Reward', None),
        ('ep_len_mean', ep_len_df, 'Episode Length (steps)', (350, 720))
    ]

    for idx, (metric_name, df, ylabel, ylim) in enumerate(metrics):
        ax = axes[idx]

        proposed_col_full = f'{proposed_col} - rollout/{metric_name}'
        prior_col_full = f'{prior_col} - rollout/{metric_name}'

        df_filtered = df[df['Step'] <= COMMON_STEPS].copy()
        proposed_data = df_filtered[['Step', proposed_col_full]].dropna()
        prior_data = df_filtered[['Step', prior_col_full]].dropna()

        if len(proposed_data) > 0 and len(prior_data) > 0:
            # 원본
            ax.plot(proposed_data['Step'], proposed_data[proposed_col_full], 
                   alpha=0.15, color=color, linewidth=0.8)
            ax.plot(prior_data['Step'], prior_data[prior_col_full], 
                   alpha=0.15, color='#3498DB', linewidth=0.8)

            # Smoothing
            prop_steps_smooth, prop_values_smooth = smooth_curve(
                proposed_data['Step'].values, 
                proposed_data[proposed_col_full].values
            )
            prior_steps_smooth, prior_values_smooth = smooth_curve(
                prior_data['Step'].values, 
                prior_data[prior_col_full].values
            )

            # Smoothed
            ax.plot(prop_steps_smooth, prop_values_smooth, 
                   label='Proposed', color=color, linewidth=2.5)
            ax.plot(prior_steps_smooth, prior_values_smooth, 
                   label='Baseline', color='#3498DB', linewidth=2.5, linestyle='--')

        # Subfigure 라벨
        ax.text(-0.15, 1.0, f'({chr(97+idx)})', transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

        ax.set_xlabel('Environment Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlim(0, COMMON_STEPS)
        if ylim:
            ax.set_ylim(ylim)

        # Legend 위치: Reward는 우하단, Length는 우상단
        if 'rew' in metric_name:
            legend_loc = 'lower right'  # Reward
        else:
            legend_loc = 'upper right'  # Length

        ax.legend(loc=legend_loc, fontsize=10, framealpha=0.95)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks([0, 15000, 30000, 45000, 60000])
        ax.set_xticklabels(['0', '15k', '30k', '45k', '60k'])

    plt.tight_layout()
    filename_base = f'Figure{figure_num}_{scenario_name.lower()}_reward_length'
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename_base}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Figure {figure_num} 저장: {filename_base}.png, .pdf")
    plt.close()

# ========== 통계 테이블 ==========
def print_statistics(success_df):
    """성능 통계 출력"""
    scenarios = [
        ('Uniform', 'Proposed_Uniform', 'Prior_Uniform'),
        ('Medium', 'Proposed_medium2', 'Prior_medium2'),
        ('Strong', 'Proposed_hard', 'Prior_Strong')
    ]

    print("\n" + "="*80)
    print("Performance Comparison at 60,000 Environment Steps")
    print("="*80)
    print(f"{'Scenario':<12} {'Proposed':<18} {'Baseline':<18} {'Improvement':<15}")
    print("-"*80)

    for scenario_name, proposed_col, prior_col in scenarios:
        proposed_col_full = f'{proposed_col} - rollout/success_rate'
        prior_col_full = f'{prior_col} - rollout/success_rate'

        df_filtered = success_df[success_df['Step'] <= COMMON_STEPS].copy()
        proposed_data = df_filtered[['Step', proposed_col_full]].dropna()
        prior_data = df_filtered[['Step', prior_col_full]].dropna()

        if len(proposed_data) > 0 and len(prior_data) > 0:
            cutoff_prop = int(len(proposed_data) * 0.9)
            final_prop_mean = proposed_data[proposed_col_full].iloc[cutoff_prop:].mean()
            final_prop_std = proposed_data[proposed_col_full].iloc[cutoff_prop:].std()

            cutoff_prior = int(len(prior_data) * 0.9)
            final_prior_mean = prior_data[prior_col_full].iloc[cutoff_prior:].mean()
            final_prior_std = prior_data[prior_col_full].iloc[cutoff_prior:].std()

            improvement = ((final_prop_mean - final_prior_mean) / final_prior_mean) * 100

            print(f"{scenario_name:<12} {final_prop_mean:.3f} ± {final_prop_std:.3f}    "
                  f"{final_prior_mean:.3f} ± {final_prior_std:.3f}    "
                  f"{improvement:+.1f}%")

    print("="*80)

# ========== 메인 실행 ==========
if __name__ == "__main__":
    print("="*80)
    print("논문용 최종 그래프 생성 (타이틀 수정 버전)")
    print("="*80)
    print(f"비교 구간: 0~{COMMON_STEPS:,} steps")
    print("="*80)

    # 데이터 로드
    success_df, ep_len_df, ep_rew_df = load_data()

    print("\n[1/5] Figure 1 생성 중 (Success Rate - 3개 시나리오)...")
    create_figure1_success_rate(success_df)

    print("[2/5] Figure 2 생성 중 (Uniform - Reward & Length)...")
    create_figure_reward_length('Uniform', 'Proposed_Uniform', 'Prior_Uniform',
                                ep_rew_df, ep_len_df, 2)

    print("[3/5] Figure 3 생성 중 (Medium - Reward & Length)...")
    create_figure_reward_length('Medium', 'Proposed_medium2', 'Prior_medium2',
                                ep_rew_df, ep_len_df, 3)

    print("[4/5] Figure 4 생성 중 (Strong - Reward & Length)...")
    create_figure_reward_length('Strong', 'Proposed_hard', 'Prior_Strong',
                                ep_rew_df, ep_len_df, 4)

    print("[5/5] 통계 테이블 생성 중...")
    print_statistics(success_df)

    print("\n" + "="*80)
    print("✅ 모든 그래프 생성 완료!")
    print("="*80)
    print("\n📊 생성된 파일:")
    print("  - Figure1_success_rate_comparison.png / .pdf")
    print("  - Figure2_uniform_reward_length.png / .pdf")
    print("  - Figure3_medium_reward_length.png / .pdf")
    print("  - Figure4_strong_reward_length.png / .pdf")
    print("\n✨ 타이틀 변경:")
    print("  Figure 1:")
    print("    - (a) Uniform Distribution (유지)")
    print("    - (b) Medium Clustering (변경)")
    print("    - (c) Strong Clustering (변경)")
    print("\n  Figure 2-4: 기존 타이틀 유지")
    print("\n🎉 논문 투고 준비 완료!")
    print("="*80)
