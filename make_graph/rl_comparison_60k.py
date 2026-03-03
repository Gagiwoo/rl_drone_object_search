"""
강화학습 실험 결과 비교 시각화 코드 (60k 버전)
- 동일 Step 구간(0~60,000)에서 성능 비교
- Smoothing 적용하여 가독성 향상
- 3가지 시나리오(Uniform, Medium, Strong) 비교
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ========== 설정 ==========
COMMON_STEPS = 60000  # 비교할 공통 구간 (원하는 값으로 변경 가능: 40000, 60000, 80000 등)
SMOOTHING_WINDOW = 101  # Smoothing window size
SMOOTHING_POLYORDER = 3  # Polynomial order for Savitzky-Golay filter

# ========== 1. 데이터 로드 ==========
def load_data():
    """WandB에서 export한 CSV 파일 로드"""
    success_df = pd.read_csv('make_graph/wandb_export_2025-12-30T23_20_06.131+09_00.csv')
    ep_len_df = pd.read_csv('make_graph/wandb_export_2025-12-30T23_20_22.746+09_00.csv')
    ep_rew_df = pd.read_csv('make_graph/wandb_export_2025-12-30T23_20_12.948+09_00.csv')

    print("=== 데이터 로드 완료 ===")
    print(f"Success Rate: {success_df.shape}")
    print(f"Episode Length: {ep_len_df.shape}")
    print(f"Episode Reward: {ep_rew_df.shape}")

    return success_df, ep_len_df, ep_rew_df

# ========== 2. Smoothing 함수 ==========
def smooth_curve(steps, values, window_length=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER):
    """
    Savitzky-Golay 필터를 사용한 곡선 평활화

    Args:
        steps: X축 데이터 (training steps)
        values: Y축 데이터 (metric values)
        window_length: Smoothing window 크기
        polyorder: 다항식 차수

    Returns:
        smoothed_steps, smoothed_values
    """
    # NaN 제거
    mask = ~np.isnan(values)
    steps_clean = steps[mask]
    values_clean = values[mask]

    if len(values_clean) > window_length:
        smoothed = savgol_filter(values_clean, window_length, polyorder)
        return steps_clean, smoothed
    return steps_clean, values_clean

# ========== 3. 시나리오별 그래프 생성 ==========
def plot_all_metrics_comparison(success_df, ep_len_df, ep_rew_df):
    """3가지 metric을 모두 포함한 비교 그래프 생성"""

    scenarios = [
        ('Uniform', 'Proposed_Uniform', 'Prior_Uniform'),
        ('Medium', 'Proposed_medium2', 'Prior_medium2'),
        ('Strong', 'Proposed_hard', 'Prior_Strong')
    ]

    metrics = [
        ('success_rate', success_df, 'Task Success Rate', (0, 1)),
        ('ep_rew_mean', ep_rew_df, 'Episode Reward Mean', None),
        ('ep_len_mean', ep_len_df, 'Episode Length Mean', (300, 800))
    ]

    for scenario_name, proposed_col, prior_col in scenarios:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{scenario_name} Scenario - Performance Comparison (0-{COMMON_STEPS:,} steps)', 
                     fontsize=16, fontweight='bold')

        for idx, (metric_name, df, ylabel, ylim) in enumerate(metrics):
            ax = axes[idx]

            # 열 이름 생성
            proposed_col_full = f'{proposed_col} - rollout/{metric_name}'
            prior_col_full = f'{prior_col} - rollout/{metric_name}'

            # 데이터 필터링 (공통 구간)
            df_filtered = df[df['Step'] <= COMMON_STEPS].copy()
            proposed_data = df_filtered[['Step', proposed_col_full]].dropna()
            prior_data = df_filtered[['Step', prior_col_full]].dropna()

            if len(proposed_data) > 0 and len(prior_data) > 0:
                # 원본 데이터 (투명하게)
                ax.plot(proposed_data['Step'], proposed_data[proposed_col_full], 
                       alpha=0.15, color='#E74C3C', linewidth=0.8)
                ax.plot(prior_data['Step'], prior_data[prior_col_full], 
                       alpha=0.15, color='#3498DB', linewidth=0.8)

                # Smoothing 적용
                prop_steps_smooth, prop_values_smooth = smooth_curve(
                    proposed_data['Step'].values, 
                    proposed_data[proposed_col_full].values
                )
                prior_steps_smooth, prior_values_smooth = smooth_curve(
                    prior_data['Step'].values, 
                    prior_data[prior_col_full].values
                )

                # Smoothed 데이터 플롯
                ax.plot(prop_steps_smooth, prop_values_smooth, 
                       label='Proposed Method', color='#E74C3C', linewidth=2.5)
                ax.plot(prior_steps_smooth, prior_values_smooth, 
                       label='Baseline (Prior)', color='#3498DB', linewidth=2.5)

                # 최종 성능 표시
                final_prop = prop_values_smooth[-1] if len(prop_values_smooth) > 0 else np.nan
                final_prior = prior_values_smooth[-1] if len(prior_values_smooth) > 0 else np.nan

                ax.text(0.98, 0.05, 
                       f'Final ({COMMON_STEPS//1000}k):\nProposed: {final_prop:.3f}\nPrior: {final_prior:.3f}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 그래프 설정
            ax.set_xlabel('Training Steps', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(0, COMMON_STEPS)
            if ylim:
                ax.set_ylim(ylim)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_title(ylabel, fontsize=12, fontweight='bold')

        plt.tight_layout()
        filename = f'comparison_{scenario_name}_all_metrics_{COMMON_STEPS//1000}k.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ {scenario_name} 그래프 저장: {filename}")
        plt.close()

# ========== 4. Success Rate 집중 그래프 ==========
def plot_success_rate_comparison(success_df):
    """Success Rate에 집중한 깔끔한 비교 그래프"""

    scenarios = [
        ('Uniform', 'Proposed_Uniform', 'Prior_Uniform'),
        ('Medium', 'Proposed_medium2', 'Prior_medium2'),
        ('Strong', 'Proposed_hard', 'Prior_Strong')
    ]

    colors_proposed = ['#E74C3C', '#F39C12', '#9B59B6']
    colors_prior = ['#3498DB', '#1ABC9C', '#2ECC71']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Task Success Rate Comparison (0-{COMMON_STEPS:,} steps)', 
                 fontsize=18, fontweight='bold', y=1.02)

    for idx, (scenario_name, proposed_col, prior_col) in enumerate(scenarios):
        ax = axes[idx]

        proposed_col_full = f'{proposed_col} - rollout/success_rate'
        prior_col_full = f'{prior_col} - rollout/success_rate'

        df_filtered = success_df[success_df['Step'] <= COMMON_STEPS].copy()
        proposed_data = df_filtered[['Step', proposed_col_full]].dropna()
        prior_data = df_filtered[['Step', prior_col_full]].dropna()

        if len(proposed_data) > 0 and len(prior_data) > 0:
            # 원본 (투명)
            ax.plot(proposed_data['Step'], proposed_data[proposed_col_full], 
                   alpha=0.2, color=colors_proposed[idx], linewidth=0.5)
            ax.plot(prior_data['Step'], prior_data[prior_col_full], 
                   alpha=0.2, color=colors_prior[idx], linewidth=0.5)

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
                   label='Proposed Method', color=colors_proposed[idx], linewidth=3)
            ax.plot(prior_steps_smooth, prior_values_smooth, 
                   label='Baseline (Prior)', color=colors_prior[idx], linewidth=3)

            # 성능 통계
            final_prop = prop_values_smooth[-1]
            final_prior = prior_values_smooth[-1]
            improvement = ((final_prop - final_prior) / final_prior) * 100

            textstr = f'Final ({COMMON_STEPS//1000}k steps):\n'
            textstr += f'Proposed: {final_prop:.3f}\n'
            textstr += f'Prior: {final_prior:.3f}\n'
            textstr += f'Improvement: {improvement:+.1f}%'

            ax.text(0.98, 0.05, textstr,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            edgecolor='black', alpha=0.8))

        ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
        ax.set_xlim(0, COMMON_STEPS)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_title(f'{scenario_name} Distribution', fontsize=14, fontweight='bold', pad=10)

        # X축 눈금을 동적으로 생성
        tick_interval = COMMON_STEPS // 4
        ax.set_xticks([i * tick_interval for i in range(5)])
        ax.set_xticklabels([f'{i * tick_interval // 1000}k' if i > 0 else '0' for i in range(5)])

    plt.tight_layout()
    filename = f'success_rate_comparison_{COMMON_STEPS//1000}k.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Success Rate 통합 그래프 저장: {filename}")
    plt.close()

# ========== 5. 통계 출력 ==========
def print_statistics(success_df):
    """최종 성능 통계 출력"""

    scenarios = [
        ('Uniform', 'Proposed_Uniform', 'Prior_Uniform'),
        ('Medium', 'Proposed_medium2', 'Prior_medium2'),
        ('Strong', 'Proposed_hard', 'Prior_Strong')
    ]

    print("\n" + "="*70)
    print(f"성능 비교 통계 ({COMMON_STEPS:,} steps 기준)")
    print("="*70)

    for scenario_name, proposed_col, prior_col in scenarios:
        proposed_col_full = f'{proposed_col} - rollout/success_rate'
        prior_col_full = f'{prior_col} - rollout/success_rate'

        df_filtered = success_df[success_df['Step'] <= COMMON_STEPS].copy()
        proposed_data = df_filtered[['Step', proposed_col_full]].dropna()
        prior_data = df_filtered[['Step', prior_col_full]].dropna()

        if len(proposed_data) > 0 and len(prior_data) > 0:
            # 최종 10% 구간의 평균
            cutoff_prop = int(len(proposed_data) * 0.9)
            final_prop_mean = proposed_data[proposed_col_full].iloc[cutoff_prop:].mean()
            final_prop_std = proposed_data[proposed_col_full].iloc[cutoff_prop:].std()

            cutoff_prior = int(len(prior_data) * 0.9)
            final_prior_mean = prior_data[prior_col_full].iloc[cutoff_prior:].mean()
            final_prior_std = prior_data[prior_col_full].iloc[cutoff_prior:].std()

            improvement = ((final_prop_mean - final_prior_mean) / final_prior_mean) * 100

            print(f"\n【{scenario_name} Scenario】")
            print(f"  Proposed: {final_prop_mean:.4f} (±{final_prop_std:.4f})")
            print(f"  Prior:    {final_prior_mean:.4f} (±{final_prior_std:.4f})")
            print(f"  Improvement: {improvement:+.2f}%")

    print("\n" + "="*70)

# ========== 메인 실행 ==========
if __name__ == "__main__":
    print("="*70)
    print("강화학습 실험 결과 비교 시각화")
    print("="*70)
    print(f"비교 구간: 0 ~ {COMMON_STEPS:,} steps")
    print("="*70)

    # 데이터 로드
    success_df, ep_len_df, ep_rew_df = load_data()

    # 그래프 생성
    print("\n=== 전체 메트릭 비교 그래프 생성 중... ===")
    plot_all_metrics_comparison(success_df, ep_len_df, ep_rew_df)

    print("\n=== Success Rate 집중 그래프 생성 중... ===")
    plot_success_rate_comparison(success_df)

    # 통계 출력
    print_statistics(success_df)

    print("\n" + "="*70)
    print("✅ 모든 작업 완료!")
    print("="*70)
    print("\n💡 TIP: 다른 구간으로 비교하려면 코드 상단의 COMMON_STEPS 값을 변경하세요.")
    print("   예: COMMON_STEPS = 40000  # 40k까지 비교")
    print("       COMMON_STEPS = 80000  # 80k까지 비교")
