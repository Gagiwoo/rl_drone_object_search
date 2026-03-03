"""
Custom DQN Algorithm with Q-value Tracking
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn import DQN

from sb3_custom.policies.local_global_softmax_scaling_policy import LocalGlobalSoftmaxScalingPolicy
from sb3_custom.policies.spatial_attention_policy import (
    SpatialAttentionPolicy,
    LearnablePoolingPolicy
)

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    import torch as th
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.type_aliases import GymEnv, Schedule, TrainFreq, TrainFrequencyUnit
    from stable_baselines3.dqn.policies import DQNPolicy


class CustomDQN(DQN):
    """
    Adapted version of DQN that also passes the Q-values to the callbacks.
    """
    
    def __init__(
        self,
        policy: str | type["DQNPolicy"],
        env: "GymEnv" | str,
        learning_rate: float | "Schedule" = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "cuda",
        _init_setup_model: bool = True,
    ) -> None:
        print(f"DEBUG: device parameter = {device}, type = {type(device)}")
        device = "cuda"
        print(f"DEBUG: device after override = {device}")
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )
        
        self._n_actions = (
            self.action_space.n
            if isinstance(self.action_space, spaces.Discrete)
            else self.action_space.shape[0]
        )
    
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Predict action with Q-values.
        
        Returns 3 values: (action, action_values, state)
        """
        allowed_policy_names = [
            'LocalGlobalSoftmaxScalingPolicy',
            'LocalGlobalCAEPolicy',
            'MultiChannelCAEPolicy',
            'SpatialAttentionPolicy',
            'LearnablePoolingPolicy',
        ]
        
        policy_name = type(self.policy).__name__
        
        if policy_name not in allowed_policy_names:
            raise RuntimeError(
                f"CustomDQN requires one of {allowed_policy_names}, "
                f"but got '{policy_name}'!"
            )
        
        # Epsilon-greedy exploration
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
                action_values = np.full(
                    (n_batch, self._n_actions), 1 / self._n_actions, dtype=np.float32
                )
            else:
                action = np.array(self.action_space.sample())
                action_values = np.full(
                    (1, self._n_actions), 1 / self._n_actions, dtype=np.float32
                )
        else:
            action, action_values, state = self.policy.predict(
                observation, state, episode_start, deterministic
            )
        
        return action, action_values, state
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample action from exploration policy.
        
        Override parent's method to return 3 values including action_values.
        
        Returns:
            unscaled_action: Selected action
            action_values: Q-values or uniform probabilities
            buffer_action: Action for replay buffer
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase: random actions
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
            action_values = np.full(
                (n_envs, self._n_actions), 1 / self._n_actions, dtype=np.float32
            )
        else:
            # Use predict with exploration (returns 3 values)
            unscaled_action, action_values, _ = self.predict(
                self._last_obs, deterministic=False
            )
        
        # Rescale action (not needed for discrete actions)
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            buffer_action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
        
        return unscaled_action, action_values, buffer_action
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: "BaseCallback",
        train_freq: "TrainFreq",
        replay_buffer: ReplayBuffer,
        action_noise: ActionNoise | None = None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        """
        Collect rollouts and store Q-values in info dict.
        """
        self.policy.set_training_mode(False)
        
        num_collected_steps, num_collected_episodes = 0, 0
        
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        
        if self.use_sde:
            self.actor.reset_noise(env.num_envs)
        
        callback.on_rollout_start()
        continue_training = True
        
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            # Sample action (now returns 3 values)
            actions, action_values, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )
            
            # Perform action
            new_obs, rewards, dones, infos = env.step(actions)
            
            # Store Q-values in info
            for i, info in enumerate(infos):
                info["action_values"] = action_values[i]
            
            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            
            callback.update_locals(locals())
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )
            
            self._update_info_buffer(infos, dones)
            
            self._store_transition(
                replay_buffer,
                buffer_actions,
                new_obs,
                rewards,
                dones,
                infos,
            )
            
            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )
            
            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    
                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        
        callback.on_rollout_end()
        
        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )
