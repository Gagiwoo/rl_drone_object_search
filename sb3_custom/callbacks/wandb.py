from __future__ import annotations

from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING

import wandb
from wandb.integration.sb3 import WandbCallback

from gymnasium import make

import shutil
import os

if TYPE_CHECKING:
    from typing import Literal


class CustomWandbCallback(WandbCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str | None = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Literal["gradients", "parameters", "all"] | None = "all",
        log_graph: bool = False,
    ) -> None:
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)

        self.log_graph = log_graph

    def _init_callback(self) -> None:
        """
        Adaptation of default WandB callback to include:
        - uploading of environment config file
        - uploading of rlzoo config file
        - adapting run name based on environment variable
        - adding tags from environment variable (comma seperated)
        """
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__

        for key in self.model.__dict__:
            if key in wandb.config:
                continue

            if isinstance(self.model.__dict__[key], (float, int, str)):
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])

        if self.gradient_save_freq > 0 and self.model is not None:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
                log_graph=self.log_graph,
            )

        wandb.config.setdefaults(d)

        # Hack to upload DroneGridEnv stuff
        self.custom_env_uploads()

        # Upload training yaml file
        if "conf_file" in wandb.config:
            train_config_file: Path = Path.cwd() / wandb.config["conf_file"]
            self.upload_file(train_config_file, "train_config" + train_config_file.suffix)

        assert wandb.run is not None

        # Change run name from env
        if getenv("WANDB_NAME") is not None:
            wandb.run.name = getenv("WANDB_NAME")

        # Add algorithm name as tag
        algo_name = wandb.config.get("algo", None)
        if algo_name is not None:
            wandb.run.tags = wandb.run.tags + tuple((algo_name,))

        # Add tags from env
        wandb_tags = getenv("WANDB_TAGS")
        if wandb_tags is not None and getenv("WANDB_TAGS") != "":
            wandb.run.tags = wandb.run.tags + tuple(wandb_tags.split(","))

    @staticmethod
    def upload_file(file: Path, filename: str, tmp_path: Path = Path.cwd() / "wandb/tmp") -> None:
        """파일을 wandb에 업로드 (symlink 대신 복사 사용)"""
        import shutil
        
        # wandb.run.dir의 files 폴더에 직접 복사
        if wandb.run is not None:
            target_dir = Path(wandb.run.dir) / "files"
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / filename
            
            # 파일 복사
            shutil.copy(str(file), str(target_path))
            
            # wandb.save() 호출하지 않음 - 이미 올바른 위치에 복사했으므로

    @staticmethod
    def custom_env_uploads() -> None:
        # Hack to upload config file for DroneGridEnv
        if all(k in wandb.config for k in ("env", "env_kwargs")) and "DroneGridEnv" in wandb.config["env"]:
            env_kwargs = wandb.config.get("env_kwargs", {})
            if env_kwargs is None:
                env_kwargs = {}
            _env = make(wandb.config["env"], **env_kwargs)
            if hasattr(_env, "config_file_path"):
                CustomWandbCallback.upload_file(_env.config_file_path, "env_config_file.yaml")

            _env.close()  # type: ignore[no-untyped-call]
