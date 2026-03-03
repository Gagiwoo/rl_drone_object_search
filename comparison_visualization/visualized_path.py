# visualized_path.py
"""
UAV 경로 비교 시각화 스크립트
Baseline vs Proposed Method Comparison
"""

import sys
import os

# 작업 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"✓ Working directory: {script_dir}")

# drone_grid_env 경로 추가
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
print(f"✓ Added to sys.path: {parent_dir}")

import numpy as np
import matplotlib.pyplot as plt
from drone_grid_env.envs.drone_grid_env import DroneGridEnv
import matplotlib

# CustomDQN import
try:
    from custom_dqn_algorithm import CustomDQN
    print("✓ CustomDQN imported successfully")
    USE_CUSTOM_DQN = True
except ImportError:
    print("⚠ CustomDQN not found, using standard DQN")
    from stable_baselines3 import DQN as CustomDQN
    USE_CUSTOM_DQN = False

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 설정
CONFIG = {
    'baseline_model': 'baseline_model.zip',
    'proposed_model': 'proposed_model.zip',
    'config_file': 'env_config_medium.yaml',
    'n_episodes': 5,
    'results_dir': 'results',
    'seed': 42,
}

def verify_and_find_files():
    """모델 파일과 설정 파일을 검증하고 찾기"""
    print("\n" + "="*60)
    print("Verifying files...")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    
    errors = []
    
    # Baseline 모델 확인
    baseline = CONFIG['baseline_model']
    if not os.path.exists(baseline):
        baseline_no_ext = baseline.replace('.zip', '')
        if os.path.exists(baseline_no_ext):
            CONFIG['baseline_model'] = baseline_no_ext
            print(f"✓ Found baseline model: {baseline_no_ext}")
        else:
            errors.append(f"Baseline model not found: {baseline}")
    else:
        print(f"✓ Found baseline model: {baseline}")
    
    # Proposed 모델 확인
    proposed = CONFIG['proposed_model']
    if not os.path.exists(proposed):
        proposed_no_ext = proposed.replace('.zip', '')
        if os.path.exists(proposed_no_ext):
            CONFIG['proposed_model'] = proposed_no_ext
            print(f"✓ Found proposed model: {proposed_no_ext}")
        else:
            errors.append(f"Proposed model not found: {proposed}")
    else:
        print(f"✓ Found proposed model: {proposed}")
    
    # Config 파일 확인
    config = CONFIG['config_file']
    if not os.path.exists(config):
        errors.append(f"Config file not found: {config}")
    else:
        print(f"✓ Found config file: {config}")
    
    if errors:
        print("\n✗ Errors found:")
        for error in errors:
            print(f"  - {error}")
        
        print("\n📁 Available files in current directory:")
        for file in sorted(os.listdir('.')):
            if file.endswith(('.zip', '.yaml', '.yml')) or os.path.isdir(file):
                file_type = "📂" if os.path.isdir(file) else "📄"
                print(f"  {file_type} {file}")
        
        print("\n💡 Please ensure these files are in:")
        print(f"   {os.getcwd()}")
        return False
    
    print("="*60)
    return True

def collect_episode_data(model_path, config_file, n_episodes=5, seed=42, model_name="model"):
    """학습된 모델로 에피소드를 실행하고 경로 데이터를 수집"""
    print(f"\n{'='*60}")
    print(f"Collecting data from {model_name}...")
    print(f"{'='*60}")
    
    # 1. 환경 생성
    try:
        env = DroneGridEnv(config_file=config_file)
        print(f"✓ Environment created: {config_file}")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return None
    
    # 2. 모델 로드
    try:
        if USE_CUSTOM_DQN:
            model = CustomDQN.load(model_path, env=env)
        else:
            model = CustomDQN.load(model_path)
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Model type: {'CustomDQN' if USE_CUSTOM_DQN else 'Standard DQN'}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        env.close()
        return None
    
    episodes_data = []
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        episode_data = {
            'flight_path': [],
            'detected_weeds': [],
            'ground_truth_weeds': None,
            'metrics': {},
            'start_position': None,  # ← 추가: 시작 위치 저장
            'end_position': None     # ← 추가: 끝 위치 저장
        }
        
        # 환경 리셋
        reset_result = env.reset(seed=seed + episode)
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        # Ground truth weeds 찾기
        total_weeds = 0
        actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        if hasattr(actual_env, 'world') and hasattr(actual_env.world, 'object_map'):
            object_map = actual_env.world.object_map
            weed_positions = np.argwhere(object_map > 0)
            if len(weed_positions) > 0:
                episode_data['ground_truth_weeds'] = weed_positions
                total_weeds = len(weed_positions)
                print(f"  ✓ Found {total_weeds} weeds via world.object_map")
        
        if total_weeds == 0 and hasattr(actual_env, 'world'):
            world = actual_env.world
            if hasattr(world, 'prior_knowledge') and world.prior_knowledge is not None:
                prior = world.prior_knowledge
                if hasattr(prior, '__len__') and len(prior) > 0:
                    episode_data['ground_truth_weeds'] = np.array(prior)
                    total_weeds = len(prior)
                    print(f"  ✓ Found {total_weeds} weeds via world.prior_knowledge")
        
        if total_weeds == 0:
            episode_data['ground_truth_weeds'] = np.array([])
            print("  ⚠ Warning: Could not find ground truth weeds")
        
        done = False
        total_steps = 0
        
        # 시작 위치 저장
        drone = None
        if hasattr(actual_env, 'drone'):
            drone = actual_env.drone
        
        if drone is not None and hasattr(drone, 'position'):
            start_pos = drone.position.copy()
            episode_data['flight_path'].append(start_pos)
            episode_data['start_position'] = start_pos  # ← 저장
            print(f"  Start position: {start_pos}")
        
        # 에피소드 실행
        while not done:
            if USE_CUSTOM_DQN:
                action, action_values, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            step_result = env.step(action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            total_steps += 1
            
            if drone is not None and hasattr(drone, 'position'):
                episode_data['flight_path'].append(drone.position.copy())
        
        # 끝 위치 저장
        if len(episode_data['flight_path']) > 0:
            episode_data['end_position'] = episode_data['flight_path'][-1]  # ← 저장
        
        # 발견한 잡초
        found_weeds_count = 0
        
        if drone is not None:
            if hasattr(drone, 'found_object_positions'):
                found_positions = drone.found_object_positions
                if found_positions is not None:
                    if isinstance(found_positions, (list, set)):
                        episode_data['detected_weeds'] = [list(pos) if hasattr(pos, '__iter__') else pos for pos in found_positions]
                        found_weeds_count = len(found_positions)
                    elif isinstance(found_positions, np.ndarray) and len(found_positions) > 0:
                        episode_data['detected_weeds'] = found_positions.tolist()
                        found_weeds_count = len(found_positions)
            
            if found_weeds_count == 0 and hasattr(drone, '_found_objects'):
                found_objects = drone._found_objects
                if isinstance(found_objects, dict):
                    episode_data['detected_weeds'] = list(found_objects.keys())
                    found_weeds_count = len(found_objects)
                elif isinstance(found_objects, (list, tuple, set)):
                    episode_data['detected_weeds'] = list(found_objects)
                    found_weeds_count = len(found_objects)
        
        if found_weeds_count == 0:
            episode_data['detected_weeds'] = []
        
        # 메트릭 계산
        found_percentage = found_weeds_count / total_weeds if total_weeds > 0 else 0
        
        episode_data['metrics'] = {
            'path_length': total_steps,
            'found_weeds': found_weeds_count,
            'total_weeds': total_weeds,
            'found_percentage': found_percentage,
            'task_success': found_percentage >= 0.8
        }
        
        episode_data['flight_path'] = np.array(episode_data['flight_path'])
        
        print(f"  Path length: {total_steps} steps")
        print(f"  Found weeds: {found_weeds_count}/{total_weeds} ({found_percentage:.1%})")
        print(f"  Start: {episode_data['start_position']}, End: {episode_data['end_position']}")  # ← 출력
        
        episodes_data.append(episode_data)
    
    env.close()
    print(f"\n✓ Data collection complete for {model_name}")
    
    return episodes_data

def plot_single_path(ax, data, title, path_color):
    """단일 에피소드의 경로를 시각화"""
    flight_path = data['flight_path']
    detected_weeds = data['detected_weeds']
    gt_weeds = data['ground_truth_weeds']
    metrics = data['metrics']
    
    field_size = 48
    if len(flight_path) > 0:
        field_size = max(int(np.max(flight_path)) + 5, 48)
    
    ax.set_xlim(0, field_size)
    ax.set_ylim(0, field_size)
    ax.set_aspect('equal')
    
    # Weeds 분류
    if len(gt_weeds) > 0:
        detected_set = set()
        if len(detected_weeds) > 0:
            for weed in detected_weeds:
                if isinstance(weed, (list, np.ndarray)):
                    detected_set.add(tuple(weed))
                else:
                    detected_set.add(weed)
        
        undetected = []
        detected_arr = []
        
        for weed in gt_weeds:
            weed_tuple = tuple(weed) if isinstance(weed, (list, np.ndarray)) else weed
            if weed_tuple in detected_set:
                detected_arr.append(weed)
            else:
                undetected.append(weed)
        
        if len(undetected) > 0:
            undetected_arr = np.array(undetected)
            ax.scatter(undetected_arr[:, 0], undetected_arr[:, 1], 
                      c='lightgray', s=80, alpha=0.6, label='Undetected', 
                      edgecolors='gray', linewidths=0.5, zorder=1)
        
        if len(detected_arr) > 0:
            detected_arr = np.array(detected_arr)
            ax.scatter(detected_arr[:, 0], detected_arr[:, 1], 
                      c='black', s=80, marker='o', label='Detected',
                      edgecolors='white', linewidths=0.5, zorder=2)
    
    # Flight path
    if len(flight_path) > 0:
        ax.plot(flight_path[:, 0], flight_path[:, 1], 
               color=path_color, linewidth=1.5, alpha=0.7, 
               label='Flight path', zorder=3)
        
        # Start & End
        ax.scatter(flight_path[0, 0], flight_path[0, 1], 
                  c='green', s=250, marker='*', label='Start', 
                  edgecolors='darkgreen', linewidths=2, zorder=5)
        
        ax.scatter(flight_path[-1, 0], flight_path[-1, 1], 
                  c='orange', s=200, marker='s', label='End', 
                  edgecolors='darkorange', linewidths=2, zorder=5)
    
    # Title
    ax.set_title(f"{title}\n",
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('X position (grid)', fontsize=10)
    ax.set_ylabel('Y position (grid)', fontsize=10)
    
    # ===== 범례 설정 (아이콘 크기 조정) =====
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=7,  # 8 → 7
                      framealpha=0.9, 
                      borderaxespad=0,
                      markerscale=0.6,  # ← 추가: 마커 크기 60%로 축소
                      handlelength=1.5,  # ← 추가: 아이콘 길이 축소
                      handleheight=0.8,  # ← 추가: 아이콘 높이 축소
                      labelspacing=0.3)  # ← 추가: 항목 간 간격 축소
    # =========================================
    
    ax.grid(True, alpha=0.3, linestyle='--')

def plot_overlay_comparison(ax, baseline_data, proposed_data):
    """Baseline과 Proposed 경로를 오버레이하여 비교"""
    baseline_path = baseline_data['flight_path']
    proposed_path = proposed_data['flight_path']
    gt_weeds = baseline_data['ground_truth_weeds']
    
    field_size = 48
    if len(baseline_path) > 0:
        field_size = max(int(np.max(baseline_path)) + 5, 
                        int(np.max(proposed_path)) + 5 if len(proposed_path) > 0 else 48)
    
    ax.set_xlim(0, field_size)
    ax.set_ylim(0, field_size)
    ax.set_aspect('equal')
    
    # Ground truth
    if len(gt_weeds) > 0:
        ax.scatter(gt_weeds[:, 0], gt_weeds[:, 1], 
                  c='lightgray', s=60, alpha=0.5, label='Weeds',
                  edgecolors='gray', linewidths=0.5, zorder=1)
    
    # Paths
    if len(baseline_path) > 0:
        ax.plot(baseline_path[:, 0], baseline_path[:, 1], 
               color='blue', linewidth=2, alpha=0.6, 
               label='Baseline', zorder=3)
    
    if len(proposed_path) > 0:
        ax.plot(proposed_path[:, 0], proposed_path[:, 1], 
               color='red', linewidth=2, alpha=0.6, 
               label='Proposed', zorder=4)
    
    # Metrics
    baseline_len = baseline_data['metrics']['path_length']
    proposed_len = proposed_data['metrics']['path_length']
    reduction = (baseline_len - proposed_len) / baseline_len * 100
    
    baseline_found = baseline_data['metrics']['found_percentage']
    proposed_found = proposed_data['metrics']['found_percentage']
    
    # Title
    ax.set_title(f"Path Comparison\n",
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('X position (grid)', fontsize=10)
    ax.set_ylabel('Y position (grid)', fontsize=10)
    
    # ===== 범례 설정 (아이콘 크기 조정) =====
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=8,  # 9 → 8
                      framealpha=0.9, 
                      borderaxespad=0,
                      markerscale=0.6,  # ← 추가
                      handlelength=1.5,  # ← 추가
                      handleheight=0.8,  # ← 추가
                      labelspacing=0.3)  # ← 추가
    # =========================================
    
    ax.grid(True, alpha=0.3, linestyle='--')

def create_comparison_plot(baseline_episodes, proposed_episodes, 
                          episode_idx=0, save_path=None):
    """Baseline과 Proposed 경로를 비교하는 종합 플롯 생성"""
    # ===== Figure 크기 조정 (범례 공간 확보) =====
    fig = plt.figure(figsize=(24, 6))  # 20 → 24로 증가
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.4)  # wspace 증가
    # ==========================================
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    baseline_data = baseline_episodes[episode_idx]
    proposed_data = proposed_episodes[episode_idx]
    
    plot_single_path(ax1, baseline_data, 
                     'Baseline (van Essen et al.)', 
                     'blue')
    
    plot_single_path(ax2, proposed_data, 
                     'Proposed (CBAM + Multi-scale)', 
                     'red')
    
    plot_overlay_comparison(ax3, baseline_data, proposed_data)
    
    fig.suptitle(f'UAV Path Planning Comparison', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # ===== tight_layout에서 rect로 범례 공간 확보 =====
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    # ==============================================
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()

def print_statistics(baseline_episodes, proposed_episodes):
    """여러 에피소드의 통계 요약 출력"""
    print(f"\n{'='*60}")
    print("PERFORMANCE STATISTICS")
    print(f"{'='*60}")
    
    baseline_lengths = [ep['metrics']['path_length'] for ep in baseline_episodes]
    baseline_found = [ep['metrics']['found_percentage'] for ep in baseline_episodes]
    
    proposed_lengths = [ep['metrics']['path_length'] for ep in proposed_episodes]
    proposed_found = [ep['metrics']['found_percentage'] for ep in proposed_episodes]
    
    print(f"\n📊 Baseline (van Essen et al.):")
    print(f"  Path length: {np.mean(baseline_lengths):.1f} ± {np.std(baseline_lengths):.1f} steps")
    print(f"  Found weeds: {np.mean(baseline_found):.1%} ± {np.std(baseline_found):.1%}")
    
    print(f"\n📊 Proposed (CBAM + Multi-scale):")
    print(f"  Path length: {np.mean(proposed_lengths):.1f} ± {np.std(proposed_lengths):.1f} steps")
    print(f"  Found weeds: {np.mean(proposed_found):.1%} ± {np.std(proposed_found):.1%}")
    
    path_reduction = (np.mean(baseline_lengths) - np.mean(proposed_lengths)) / np.mean(baseline_lengths) * 100
    found_diff = (np.mean(proposed_found) - np.mean(baseline_found)) * 100
    
    print(f"\n✨ Improvement:")
    print(f"  Path reduction: {path_reduction:+.1f}%")
    print(f"  Found difference: {found_diff:+.1f}%p")
    print(f"{'='*60}\n")

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("UAV PATH PLANNING VISUALIZATION")
    print("Baseline vs Proposed Method Comparison")
    print("="*60)
    
    if not verify_and_find_files():
        sys.exit(1)
    
    # 결과 디렉토리 생성
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    baseline_data_path = os.path.join(CONFIG['results_dir'], 'baseline_data.npy')
    proposed_data_path = os.path.join(CONFIG['results_dir'], 'proposed_data.npy')
    
    # ===== 중요: 캐시 삭제 옵션 =====
    regenerate = True  # ← True로 설정하면 항상 새로 생성
    
    if regenerate:
        print("\n💡 Regenerating all data for consistency...")
        if os.path.exists(baseline_data_path):
            os.remove(baseline_data_path)
            print(f"  ✓ Removed old baseline data")
        if os.path.exists(proposed_data_path):
            os.remove(proposed_data_path)
            print(f"  ✓ Removed old proposed data")
    # ===================================
    
    # 1. Baseline 데이터
    if os.path.exists(baseline_data_path) and not regenerate:
        print(f"\n✓ Loading existing baseline data: {baseline_data_path}")
        baseline_episodes = np.load(baseline_data_path, allow_pickle=True)
    else:
        baseline_episodes = collect_episode_data(
            CONFIG['baseline_model'],
            CONFIG['config_file'],
            CONFIG['n_episodes'],
            CONFIG['seed'],
            "Baseline"
        )
        if baseline_episodes is not None:
            np.save(baseline_data_path, baseline_episodes)
            print(f"✓ Baseline data saved: {baseline_data_path}")
    
    # 2. Proposed 데이터
    if os.path.exists(proposed_data_path) and not regenerate:
        print(f"\n✓ Loading existing proposed data: {proposed_data_path}")
        proposed_episodes = np.load(proposed_data_path, allow_pickle=True)
    else:
        proposed_episodes = collect_episode_data(
            CONFIG['proposed_model'],
            CONFIG['config_file'],
            CONFIG['n_episodes'],
            CONFIG['seed'],
            "Proposed"
        )
        if proposed_episodes is not None:
            np.save(proposed_data_path, proposed_episodes)
            print(f"✓ Proposed data saved: {proposed_data_path}")
    
    # 3. 데이터 검증
    if baseline_episodes is None or proposed_episodes is None:
        print("\n✗ Error: Failed to collect data")
        return
    
    # 4. 시작/끝 위치 비교 출력
    print(f"\n{'='*60}")
    print("POSITION COMPARISON")
    print(f"{'='*60}")
    for i in range(len(baseline_episodes)):
        b_start = baseline_episodes[i]['start_position']
        b_end = baseline_episodes[i]['end_position']
        p_start = proposed_episodes[i]['start_position']
        p_end = proposed_episodes[i]['end_position']
        
        print(f"\nEpisode {i+1}:")
        print(f"  Baseline  - Start: {b_start}, End: {b_end}")
        print(f"  Proposed  - Start: {p_start}, End: {p_end}")
        
        if np.array_equal(b_start, p_start):
            print(f"  ✓ Start positions match!")
        else:
            print(f"  ✗ Start positions differ!")
    print(f"{'='*60}")
    
    # 5. 통계 출력
    print_statistics(baseline_episodes, proposed_episodes)
    
    # 6. 시각화 생성
    for i in range(len(baseline_episodes)):
        plot_path = os.path.join(CONFIG['results_dir'], 
                                 f'comparison_episode_{i+1}.png')
        create_comparison_plot(baseline_episodes, proposed_episodes, 
                              episode_idx=i, save_path=plot_path)
    
    print("\n✓ All visualizations complete!")
    print(f"✓ Results saved in: {os.path.abspath(CONFIG['results_dir'])}/")

if __name__ == "__main__":
    main()
