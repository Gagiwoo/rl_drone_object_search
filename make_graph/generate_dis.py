import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random

# 시드 설정
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
    """Medium Clustering: 중간 정도 군집"""
    # 클러스터 중심 생성
    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        x = np.random.randint(10, grid_size-10)
        y = np.random.randint(10, grid_size-10)
        cluster_centers.append((x, y))
    
    targets = []
    targets_per_cluster = num_targets // num_clusters
    sigma = 5  # 클러스터 분산
    
    for center in cluster_centers:
        for _ in range(targets_per_cluster):
            while True:
                x = int(np.random.normal(center[0], sigma))
                y = int(np.random.normal(center[1], sigma))
                if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
                    targets.append((x, y))
                    break
    
    # 남은 타겟 추가
    while len(targets) < num_targets:
        center = random.choice(cluster_centers)
        x = int(np.random.normal(center[0], sigma))
        y = int(np.random.normal(center[1], sigma))
        if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
            targets.append((x, y))
    
    return targets

def generate_strong_clustering(grid_size, num_targets, num_clusters=3):
    """Strong Clustering: 강한 군집"""
    # 클러스터 중심 생성
    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        x = np.random.randint(10, grid_size-10)
        y = np.random.randint(10, grid_size-10)
        cluster_centers.append((x, y))
    
    targets = []
    targets_per_cluster = num_targets // num_clusters
    sigma = 3  # 더 작은 분산 (강한 군집)
    
    for center in cluster_centers:
        for _ in range(targets_per_cluster):
            while True:
                x = int(np.random.normal(center[0], sigma))
                y = int(np.random.normal(center[1], sigma))
                if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
                    targets.append((x, y))
                    break
    
    # 남은 타겟 추가
    while len(targets) < num_targets:
        center = random.choice(cluster_centers)
        x = int(np.random.normal(center[0], sigma))
        y = int(np.random.normal(center[1], sigma))
        if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in targets:
            targets.append((x, y))
    
    return targets

def plot_distribution(targets, title, ax):
    """분포 시각화"""
    # 그리드 배경
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 타겟 표시
    for x, y in targets:
        circle = Circle((x, y), 0.5, color='red', alpha=0.7)
        ax.add_patch(circle)
    
    ax.set_xlabel('X coordinate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y coordinate', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.invert_yaxis()  # Y축 반전 (위에서 아래로)

# Figure 생성
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3가지 분포 생성 및 시각화
distributions = [
    (generate_uniform_distribution(GRID_SIZE, NUM_TARGETS), 
     'Uniform Distribution', 0),
    (generate_medium_clustering(GRID_SIZE, NUM_TARGETS, num_clusters=5), 
     'Medium Clustering', 1),
    (generate_strong_clustering(GRID_SIZE, NUM_TARGETS, num_clusters=3), 
     'Strong Clustering', 2)
]

for targets, title, idx in distributions:
    # Subfigure 라벨
    axes[idx].text(-0.12, 1.03, f'({chr(97+idx)})', 
                   transform=axes[idx].transAxes,
                   fontsize=14, fontweight='bold', va='top')
    
    plot_distribution(targets, title, axes[idx])
    
    # 통계 정보 추가
    num_targets = len(targets)
    axes[idx].text(0.5, -0.08, f'Targets: {num_targets}',
                   transform=axes[idx].transAxes,
                   fontsize=10, ha='center', va='top')

plt.suptitle('Target Distribution Scenarios', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('Figure_target_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_target_distributions.pdf', dpi=300, bbox_inches='tight')
print("✓ Target distribution figures generated!")
plt.show()
