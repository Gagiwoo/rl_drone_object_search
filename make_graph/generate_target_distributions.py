"""
논문용 타겟 분포 시각화 코드 (깔끔한 버전)
3가지 시나리오의 타겟(잡초) 분포를 생성
- 하단 텍스트 박스 제거
- 깔끔한 디자인
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# 시드 설정 (재현 가능하도록)
np.random.seed(42)
random.seed(42)

# 환경 설정
GRID_SIZE = 50  # 50x50 그리드
NUM_TARGETS = 30  # 타겟 개수

def generate_uniform_distribution(grid_size, num_targets):
    """Uniform Distribution: 완전 랜덤 분포"""
    targets = []
    while len(targets) < num_targets:
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        if (x, y) not in targets:
            targets.append((x, y))
    return targets

def generate_medium_clustering(grid_size, num_targets, num_clusters=5):
    """Medium Clustering: 중간 정도 군집 (5개 클러스터)"""
    # 클러스터 중심 생성
    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        x = np.random.randint(10, grid_size-10)
        y = np.random.randint(10, grid_size-10)
        # 기존 중심과 최소 거리 유지
        if all(np.sqrt((x-cx)**2 + (y-cy)**2) > 10 for cx, cy in cluster_centers):
            cluster_centers.append((x, y))

    targets = []
    targets_per_cluster = num_targets // num_clusters
    sigma = 5  # 클러스터 분산

    for center in cluster_centers:
        for _ in range(targets_per_cluster):
            attempts = 0
            while attempts < 100:
                x = int(np.random.normal(center[0], sigma))
                y = int(np.random.normal(center[1], sigma))
                if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
                    targets.append((x, y))
                    break
                attempts += 1

    # 남은 타겟 추가
    while len(targets) < num_targets:
        center = random.choice(cluster_centers)
        x = int(np.random.normal(center[0], sigma))
        y = int(np.random.normal(center[1], sigma))
        if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
            targets.append((x, y))

    return targets

def generate_strong_clustering(grid_size, num_targets, num_clusters=3):
    """Strong Clustering: 강한 군집 (3개 클러스터)"""
    # 클러스터 중심 생성
    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        x = np.random.randint(12, grid_size-12)
        y = np.random.randint(12, grid_size-12)
        # 기존 중심과 최소 거리 유지
        if all(np.sqrt((x-cx)**2 + (y-cy)**2) > 15 for cx, cy in cluster_centers):
            cluster_centers.append((x, y))

    targets = []
    targets_per_cluster = num_targets // num_clusters
    sigma = 3  # 더 작은 분산 (강한 군집)

    for center in cluster_centers:
        for _ in range(targets_per_cluster):
            attempts = 0
            while attempts < 100:
                x = int(np.random.normal(center[0], sigma))
                y = int(np.random.normal(center[1], sigma))
                if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
                    targets.append((x, y))
                    break
                attempts += 1

    # 남은 타겟 추가
    while len(targets) < num_targets:
        center = random.choice(cluster_centers)
        x = int(np.random.normal(center[0], sigma))
        y = int(np.random.normal(center[1], sigma))
        if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
            targets.append((x, y))

    return targets

def plot_distribution(targets, title, ax):
    """분포 시각화 (텍스트 박스 없음)"""
    # 배경 색상 (회색)
    ax.set_facecolor('#CCCCCC')

    # 그리드 범위
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_aspect('equal')

    # 그리드 라인
    for i in range(0, GRID_SIZE+1, 5):
        ax.axhline(y=i, color='white', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i, color='white', linewidth=0.5, alpha=0.5)

    # 타겟 표시 (어두운 회색 사각형)
    for x, y in targets:
        rect = Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                         facecolor='#666666', edgecolor='none')
        ax.add_patch(rect)

    # 축 레이블
    ax.set_xlabel('X coordinate (grid cells)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y coordinate (grid cells)', fontsize=11, fontweight='bold')

    # 타이틀
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Y축 반전 (위에서 아래로)
    ax.invert_yaxis()

    # 축 눈금
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 10, 20, 30, 40, 50])

# Figure 생성
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3가지 분포 생성
print("타겟 분포 생성 중...")

uniform_targets = generate_uniform_distribution(GRID_SIZE, NUM_TARGETS)
print(f"✓ Uniform Distribution: {len(uniform_targets)} targets")

medium_targets = generate_medium_clustering(GRID_SIZE, NUM_TARGETS, num_clusters=5)
print(f"✓ Medium Clustering: {len(medium_targets)} targets (5 clusters)")

strong_targets = generate_strong_clustering(GRID_SIZE, NUM_TARGETS, num_clusters=3)
print(f"✓ Strong Clustering: {len(strong_targets)} targets (3 clusters)")

# 시각화
distributions = [
    (uniform_targets, 'Uniform Distribution', 0),
    (medium_targets, 'Medium Clustering', 1),
    (strong_targets, 'Strong Clustering', 2)
]

for targets, title, idx in distributions:
    # Subfigure 라벨
    axes[idx].text(-0.12, 1.03, f'({chr(97+idx)})', 
                   transform=axes[idx].transAxes,
                   fontsize=14, fontweight='bold', va='top')

    plot_distribution(targets, title, axes[idx])

plt.suptitle('Target Distribution Scenarios', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# 저장
plt.savefig('Figure_target_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_target_distributions.pdf', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("✅ 타겟 분포 Figure 생성 완료!")
print("="*80)
print("\n📊 생성된 파일:")
print("  - Figure_target_distributions.png")
print("  - Figure_target_distributions.pdf")
print("\n📝 분포 특징:")
print("  (a) Uniform Distribution: 타겟이 랜덤하게 분포")
print("  (b) Medium Clustering: 5개 클러스터, 중간 밀집도 (σ=5)")
print("  (c) Strong Clustering: 3개 클러스터, 강한 밀집도 (σ=3)")
print("\n✨ 개선 사항:")
print("  ✓ 하단 텍스트 박스 제거")
print("  ✓ 깔끔한 디자인")
print("  ✓ 논문 Figure에 적합한 스타일")
print("="*80)

plt.show()
