"""Experiment logging: CSV + optional Weights & Biases."""

import csv
import json
import os
import time


class ExperimentLogger:
    """
    Logs metrics to a CSV file and optionally to Weights & Biases.

    Each log() call records the metric dict along with the gradient step
    and wall-clock time elapsed since the logger was created.
    """

    def __init__(self, log_dir: str, run_name: str, config: dict, use_wandb: bool = False):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.run_name = run_name
        self.start_time = time.time()
        self.use_wandb = use_wandb

        # Save config
        config_path = os.path.join(log_dir, f"{run_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Open CSV
        self.csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = None  # initialized on first log() so we know all columns
        self._fieldnames = []
        self._rows = []  # buffered rows, replayed if fieldnames grow

        # Optional W&B
        if use_wandb:
            try:
                import wandb
                wandb.init(project="cs336-lm", name=run_name, config=config)
                self._wandb = wandb
            except ImportError:
                print("wandb not installed — skipping W&B logging.")
                self.use_wandb = False

    def log(self, step: int, metrics: dict):
        elapsed = time.time() - self.start_time
        row = {"step": step, "wall_time": round(elapsed, 2), **metrics}
        self._rows.append(row)

        # New columns we haven't seen before: rewrite the file from scratch
        # with the expanded fieldnames so DictWriter doesn't reject the row.
        new_fields = [k for k in row.keys() if k not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(new_fields)
            self._csv_file.close()
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
            for prev_row in self._rows:
                self._csv_writer.writerow(prev_row)
        else:
            self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.use_wandb:
            self._wandb.log(row, step=step)

    def close(self):
        self._csv_file.close()
        if self.use_wandb:
            self._wandb.finish()
