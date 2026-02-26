import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch


class ExperimentLogger:
    def __init__(
        self,
        base_dir: str,
        experiment_name: str,
        config: Dict[str, Any],
    ):
        """
        Args:
            base_dir (str): Root directory for experiments.
            experiment_name (str): Name describing this run.
            config (Dict): Hyperparameters and metadata.
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(base_dir) / f"{timestamp}_{experiment_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.start_time = time.time()
        self.global_step = 0

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

    def log(self, metrics: Dict[str, float]):
        """
        Log metrics for current step.

        Automatically logs:
            - global_step
            - wallclock_time
        """
        wallclock_time = time.time() - self.start_time

        log_entry = {
            "step": self.global_step,
            "wallclock_time": wallclock_time,
            **metrics,
        }

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def step(self):
        self.global_step += 1

    def save_checkpoint(self, model, optimizer, iteration: Optional[int] = None):
        """
        Save model + optimizer state.
        """
        ckpt_path = self.checkpoint_dir / f"step_{self.global_step}.pt"

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": self.global_step,
                "iteration": iteration,
            },
            ckpt_path,
        )