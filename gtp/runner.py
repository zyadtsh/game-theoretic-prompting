"""Experiment runner that iterates over a config grid and executes game runs.

Supports multiple conditions x repetitions, crash-resilient resumption
(skips completed run_ids), and per-run error isolation. Transient API
errors (502, rate limits) are retried at the LLM call level in builder.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict

from gtp.config import ExperimentConfig, GameRunConfig
from gtp.games import GAME_REGISTRY
from gtp.graph import build_game_graph
from gtp.results import ResultStore


def _derive_seed(run_id: str, agent_name: str) -> int:
    """Stable 32-bit seed from (run_id, agent_name). Deterministic across runs."""
    h = hashlib.sha256(f"{run_id}|{agent_name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


class ExperimentRunner:
    """Runs all configurations x repetitions, saves results, supports resumption."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "results",
        resume_path: str | None = None,
    ):
        self.config = config
        self.store = ResultStore(output_dir, config.name, resume_path=resume_path)

    def run(self, config_path: str | None = None) -> str:
        """Run all experiment configurations. Returns path to results directory."""
        if config_path:
            self.store.save_experiment_config(config_path)

        runs = self.config.runs
        total = len(runs) * self.config.repetitions
        completed = 0
        errors = 0

        print(f"Experiment: {self.config.name}")
        print(f"Conditions: {len(runs)}, Repetitions: {self.config.repetitions}, Total runs: {total}")
        print(f"Output: {self.store.path}")
        print("=" * 60)

        for i, run_config in enumerate(runs):
            for rep in range(self.config.repetitions):
                run_id = self._make_run_id(run_config, rep)

                if self.store.has_result(run_id):
                    print(f"  [{completed + 1}/{total}] Skipping {run_id} (already exists)")
                    completed += 1
                    continue

                print(f"\n[{completed + 1}/{total}] {run_id}")
                try:
                    result = self._run_single(run_config, run_id)
                    self.store.save(run_id, result)
                    completed += 1
                    print(
                        f"  -> A={result['scores']['Agent_A']}, "
                        f"B={result['scores']['Agent_B']}"
                    )
                except Exception as e:
                    errors += 1
                    completed += 1
                    print(f"  ERROR: {e}")
                    self.store.save_error(run_id, str(e))

        print("\n" + "=" * 60)
        print(f"Done. {completed - errors} succeeded, {errors} failed.")
        print(f"Results: {self.store.path}")
        return self.store.path

    def _run_single(self, run_config: GameRunConfig, run_id: str) -> dict:
        """Execute one game run."""
        game_cls = GAME_REGISTRY[run_config.game]
        game = game_cls()

        # Record original prompt keys (e.g. "cooperative") for results metadata
        prompt_a_key = run_config.agent_a.system_prompt
        prompt_b_key = run_config.agent_b.system_prompt

        graph, game = build_game_graph(
            run_config,
            game,
            seed_a=_derive_seed(run_id, "Agent_A"),
            seed_b=_derive_seed(run_id, "Agent_B"),
        )

        result = graph.invoke(
            {
                "phase": "controller",
                "current_round": 0,
                "move_a": None,
                "move_b": None,
                "round_results": [],
                "game_over": False,
            },
            config={"recursion_limit": 4 * run_config.num_rounds + 20},
        )

        moves = game.get_moves()
        first_move = moves[0]  # first move for rate calculation (e.g. COOPERATE, SWERVE, STAG)

        config_dict = asdict(run_config)
        config_dict["prompt_a_key"] = prompt_a_key
        config_dict["prompt_b_key"] = prompt_b_key

        return {
            "run_id": run_id,
            "config": config_dict,
            "history": game.history,
            "scores": game.scores,
            "reasoning_log": game.reasoning_log,
            "first_move_name": first_move,
            "first_move_rates": {
                "Agent_A": (
                    sum(1 for r in game.history if r["move_a"] == first_move)
                    / len(game.history)
                    if game.history
                    else 0.0
                ),
                "Agent_B": (
                    sum(1 for r in game.history if r["move_b"] == first_move)
                    / len(game.history)
                    if game.history
                    else 0.0
                ),
            },
        }

    def _make_run_id(self, run_config: GameRunConfig, rep: int) -> str:
        if run_config.condition_id:
            return f"{run_config.condition_id}_rep{rep}"
        # Fallback: generate from config
        pa = run_config.agent_a.system_prompt[:20].replace(" ", "_")
        pb = run_config.agent_b.system_prompt[:20].replace(" ", "_")
        return f"{run_config.game}_{pa}_{pb}_rep{rep}"

    def dry_run(self):
        """Print the expanded grid without executing."""
        runs = self.config.runs
        total = len(runs) * self.config.repetitions
        print(f"Experiment: {self.config.name}")
        print(f"Conditions: {len(runs)}, Repetitions: {self.config.repetitions}, Total runs: {total}")
        print()
        for i, r in enumerate(runs):
            cid = r.condition_id or f"run_{i}"
            print(f"  {cid}")
            print(f"    game={r.game}, rounds={r.num_rounds}")
            print(f"    A: prompt={r.agent_a.system_prompt[:40]}..., sees_B={r.agent_a.can_see_opponent_prompt}")
            print(f"    B: prompt={r.agent_b.system_prompt[:40]}..., sees_A={r.agent_b.can_see_opponent_prompt}")
