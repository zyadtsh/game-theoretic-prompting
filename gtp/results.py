"""Result storage: JSONL summaries + per-run JSON files with full reasoning traces.

Each experiment run produces:
  results/{name}_{timestamp}/results.jsonl   — one summary line per run
  results/{name}_{timestamp}/runs/{id}.json  — full history + reasoning
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pandas as pd


class ResultStore:
    """Stores experiment results as JSONL summaries + per-run JSON files."""

    def __init__(
        self,
        base_dir: str,
        experiment_name: str,
        resume_path: str | None = None,
    ):
        if resume_path:
            # Resume into an existing results directory. The runner's
            # has_result() check then skips runs whose JSON already exists,
            # so only previously failed/missing runs get re-executed.
            if not os.path.isdir(resume_path):
                raise FileNotFoundError(
                    f"--resume path does not exist: {resume_path}"
                )
            self.path = resume_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        self._runs_dir = os.path.join(self.path, "runs")
        os.makedirs(self._runs_dir, exist_ok=True)

    def save(self, run_id: str, result: dict):
        """Save a completed run: append summary to JSONL + write full JSON."""
        summary = {
            "run_id": run_id,
            "game": result["config"]["game"],
            "condition_id": result["config"].get("condition_id", ""),
            "prompt_a": result["config"]["agent_a"]["system_prompt"][:100],
            "prompt_b": result["config"]["agent_b"]["system_prompt"][:100],
            "prompt_a_key": result["config"].get("prompt_a_key", ""),
            "prompt_b_key": result["config"].get("prompt_b_key", ""),
            "transparency_a": result["config"]["agent_a"]["can_see_opponent_prompt"],
            "transparency_b": result["config"]["agent_b"]["can_see_opponent_prompt"],
            "score_a": result["scores"]["Agent_A"],
            "score_b": result["scores"]["Agent_B"],
            "first_move_name": result.get("first_move_name", ""),
            "first_move_rate_a": result["first_move_rates"]["Agent_A"],
            "first_move_rate_b": result["first_move_rates"]["Agent_B"],
            "num_rounds": len(result["history"]),
            "winner": (
                "Agent_A"
                if result["scores"]["Agent_A"] > result["scores"]["Agent_B"]
                else (
                    "Agent_B"
                    if result["scores"]["Agent_B"] > result["scores"]["Agent_A"]
                    else "Tie"
                )
            ),
        }
        # Resume-safe append: when re-running an errored run via --resume,
        # drop any pre-existing summary line for the same run_id before
        # appending. The per-run JSON below is overwritten atomically, but
        # the JSONL would otherwise accumulate duplicates that confuse
        # downstream group-bys.
        jsonl_path = os.path.join(self.path, "results.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                original = f.readlines()
            kept = [
                line for line in original
                if json.loads(line).get("run_id") != run_id
            ]
            if len(kept) != len(original):
                with open(jsonl_path, "w") as f:
                    f.writelines(kept)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        with open(os.path.join(self._runs_dir, f"{run_id}.json"), "w") as f:
            json.dump(result, f, indent=2)

    def save_error(self, run_id: str, error: str):
        """Save an error entry for a failed run."""
        entry = {"run_id": run_id, "error": error}
        with open(os.path.join(self.path, "errors.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_experiment_config(self, config_path: str):
        """Copy the experiment YAML into the results directory."""
        shutil.copy2(config_path, os.path.join(self.path, "experiment.yaml"))

    def has_result(self, run_id: str) -> bool:
        return os.path.exists(os.path.join(self._runs_dir, f"{run_id}.json"))

    def load_summary(self) -> pd.DataFrame:
        """Load results.jsonl as a DataFrame."""
        jsonl_path = os.path.join(self.path, "results.jsonl")
        if not os.path.exists(jsonl_path):
            return pd.DataFrame()
        return pd.read_json(jsonl_path, lines=True)

    def load_run(self, run_id: str) -> dict:
        """Load full result for a specific run."""
        with open(os.path.join(self._runs_dir, f"{run_id}.json")) as f:
            return json.load(f)
