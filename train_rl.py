from __future__ import annotations

from warnings import filterwarnings

from rl_zoo3.utils import ALGOS

from drone_grid_env import logger

from sb3_custom.custom_dqn_algorithm import CustomDQN

from sb3_custom.double_dqn_algorithm import DoubleDQN

import wandb

filterwarnings("ignore", category=UserWarning)  # Ignore Gymnasium UserWarning

# Register custom algorithms
ALGOS["ddqn"] = DoubleDQN
ALGOS["custom_dqn"] = CustomDQN

logger.set_level(logger.WARN)


def main() -> None:
    # wandb 로그인 - 필수는 아니며, 환경변수에 키가 있으면 생략 가능
    wandb.login(key="dd0edb9b7e000d63f9ccd75ed3eabb3e99633250")
    
    # wandb 초기화 - 프로젝트명과 엔티티 계정명 정확히 입력
    wandb.init(project="test", entity="ksain1-ajou-university")

    from rl_zoo3.train import train as rlzoo3_train

    rlzoo3_train()


if __name__ == "__main__":
    main()


#python train_rl.py --algo custom_dqn --env DroneGridEnv-v0 --conf-file experiments/distributions/dqn_random.yaml --track

#python train_rl.py --algo custom_dqn --env DroneGridEnv-v0 --conf-file experiments/distributions/dqn_spatial_attention.yaml --track
