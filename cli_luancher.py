import subprocess
from pathlib import Path
import sys


# 실험 구성 옵션: (메뉴명, config 파일 경로, wandb 베이스태그)
configs = [
    ("기존 방식 (Pooling)", "experiments/distributions/dqn_random.yaml", "baseline-pooling"),
    ("클러스터드 (Pooling)", "experiments/distributions/dqn_clustered.yaml", "baseline-clustered"),
    ("제안 방식 (CAE-latent)", "experiments/distributions/dqn_cae.yaml", "cae-latent"),
]

cae_model_path = 'models/cae_pretrained.pth'  # CAE 모델 파일 경로 (CAE 실험용)
cae_config_file = 'experiments/distributions/dqn_cae.yaml'

print("\n===== RL 실험 자동 시작 =====\n")
# 1. 실험/설정 선택
print("[실험 방식/설정 선택]")
for idx, (title, path, tag) in enumerate(configs):
    print(f"  {idx + 1}. {title} ⇒ {path}")
selected_idx_raw = input("쉼표(,)로 여러개 선택, 예: 1,3 : ")
selected_idxs = [int(i)-1 for i in selected_idx_raw.split(",") if i.strip()]

# 2. 알고리즘 선택
algos = ['custom_dqn', 'custom_ddqn']
print("\n[알고리즘 선택]")
for idx, opt in enumerate(algos):
    print(f"  {idx + 1}. {opt}")
alg_idx = int(input("번호를 입력하세요: ")) - 1
algo = algos[alg_idx]

# 3. wandb 로그 설정
track = input("\nWeights & Biases로 로깅하시겠습니까? (y/n) ").lower() == 'y'
user_tags = input("wandb 태그(추가, 쉼표로): ") if track else ''

# 4. seed 설정
seed = input("Seed를 입력(Enter시 무시): ") or None

# 5. 각 실험별 실행
for idx in selected_idxs:
    conf_file = configs[idx][1]
    base_tag = configs[idx][2]
    tags = base_tag
    if user_tags.strip():
        tags += "," + user_tags.strip()
    
    # CAE config인 경우: pretrained 모델 유무 확인 → 없으면 사전학습 자동
    if conf_file == cae_config_file:
        cae_path = Path(cae_model_path)
        if not cae_path.exists():
            print("[CAE 사전학습] 사전학습된 모델이 없어 pretrain_cae.py를 실행합니다!")
            cae_pretrain_cmd = [
                sys.executable, 'scripts/pretrain_cae.py',
                '--env', 'DroneGridEnv-v0',
                '--n-episodes', '1000',
                '--latent-dim', '64',
                '--output', cae_model_path
            ]
            print(f"실행 명령: {' '.join(cae_pretrain_cmd)}")
            try:
                # cwd(project root) 설정: 현재 .py가 바로 프로젝트 루트라면 아래 추가해도 좋음
                project_root = Path(__file__).parent.resolve()
                result = subprocess.run(
                    cae_pretrain_cmd,
                    check=True,        # 실페 시 예외 발생
                    cwd=project_root,  # 작업 경로 명시(필수)
                    capture_output=True,
                    text=True
                )
                print("CAE 사전학습이 완료됐습니다.")
            except subprocess.CalledProcessError as e:
                print("[ERROR] CAE 사전학습 실패!")
                print("에러 로그:")
                print(e.stderr or e.stdout)
                print("문제를 먼저 해결한 뒤 다시 실행하세요.")
                sys.exit(1)
        else:
            print(f"이미 사전학습된 CAE 모델파일이 있습니다. ({cae_model_path}). 사전학습 생략!")
    # RL 학습 명령
    rl_cmd = [
        sys.executable, 'train_rl.py',
        f'--algo={algo}',
        '--env=DroneGridEnv-v0',
        '--gym-packages=drone_grid_env',
        f'--conf-file={conf_file}'
    ]
    if track:
        rl_cmd += ['--track']
        rl_cmd += [f'--wandb-tags={tags}']
    if seed:
        rl_cmd += [f'--seed={seed}']
    print("\n아래 명령어를 실행합니다:")
    print(' '.join(rl_cmd))
    input("계속하려면 Enter... (다음 실험은 Enter 후 진행)")
    subprocess.run(rl_cmd)
