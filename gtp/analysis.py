"""Analysis utilities for loading experiment results, aggregating, and plotting.

Three classes of analysis live here:

1. Light aggregations over `results.jsonl` summaries (standings, diagnostic,
   cooperation plots).
2. A replay harness (`replay_llm_strategy`) that re-runs an LLM strategy
   agent against a recorded opponent trajectory from a fixed-twin match, to
   measure strategy-execution fidelity directly. See `fidelity_analysis`.
3. Round-level and reasoning-level analyses built on top of two flat
   DataFrames (`load_per_round_df`, `load_per_reasoning_df`) — cooperation
   trajectories, move heatmaps, intent/action mismatch detection, endgame
   reference counts, and a markdown experiment summary.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from gtp.config import AgentConfig, GameRunConfig, PROMPT_REGISTRY, format_prompt
from gtp.agents.fixed import (
    FIXED_STRATEGY_REGISTRY,
    DETERMINISTIC_FIXED_KEYS,
    STOCHASTIC_FIXED_KEYS,
    is_fixed_strategy,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """Load results.jsonl from a results directory as a DataFrame."""
    return pd.read_json(os.path.join(results_dir, "results.jsonl"), lines=True)


def load_run(results_dir: str, run_id: str) -> dict:
    """Load full result for a specific run."""
    path = os.path.join(results_dir, "runs", f"{run_id}.json")
    with open(path) as f:
        return json.load(f)


def list_runs(results_dir: str) -> list[str]:
    """Return all run_ids with a saved per-run JSON in `results_dir`."""
    runs_dir = os.path.join(results_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []
    return sorted(
        f[:-5] for f in os.listdir(runs_dir) if f.endswith(".json")
    )


# ---------------------------------------------------------------------------
# Strategy family / bucket helpers
# ---------------------------------------------------------------------------

# The Axelrod 1980 family is the 14 strategies with matching *_fixed / *_llm
# twins used in the main experiment. Legacy fixed keys aren't part of the
# fidelity comparison because they have no named LLM twin.
_AXELROD_FAMILIES: tuple[str, ...] = (
    "tft",
    "tideman_chieruzzi",
    "nydegger",
    "shubik",
    "grudger",
    "davis",
    "downing",
    "random",
    "grofman",
    "joss",
    "feld",
    "tullock",
    "graaskamp",
    "stein_rapoport",
)


def _bucket(key: str) -> str:
    if key in FIXED_STRATEGY_REGISTRY:
        return "fixed"
    if key == "minimal":
        return "minimal"
    return "llm"


def _family(key: str) -> str:
    for fam in _AXELROD_FAMILIES:
        if key == f"{fam}_fixed" or key == f"{fam}_llm":
            return fam
    return key  # minimal, legacy fixed, etc.


def _is_deterministic(key: str) -> bool:
    """Deterministic for rep-count purposes: fixed-deterministic or an LLM
    whose paired fixed twin is deterministic."""
    if key in DETERMINISTIC_FIXED_KEYS:
        return True
    fam = _family(key)
    if fam in _AXELROD_FAMILIES:
        return f"{fam}_fixed" in DETERMINISTIC_FIXED_KEYS
    return True  # unknown -> assume deterministic


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------

def standings_table(df: pd.DataFrame) -> pd.DataFrame:
    """Score + cooperation rates per agent vs `minimal`.

    Works on the output of axelrod_deterministic / axelrod_stochastic where
    every match is one of the 28 strategy-prompted agents on side A and
    `minimal` on side B. Returns one row per `prompt_a_key`.
    """
    if "prompt_a_key" not in df.columns:
        raise ValueError(
            "standings_table expects results.jsonl with prompt_a_key column; "
            "the runner writes this field automatically."
        )

    # First-move rate for A is the cooperation rate in PD because
    # first_move is COOPERATE. Ditto for B.
    grouped = df.groupby("prompt_a_key").agg(
        score=("score_a", "mean"),
        score_minimal=("score_b", "mean"),
        coop_rate_vs_minimal=("first_move_rate_a", "mean"),
        minimal_coop_rate=("first_move_rate_b", "mean"),
        n_reps=("run_id", "count"),
    )

    grouped["bucket"] = grouped.index.map(_bucket)
    grouped["strategy_family"] = grouped.index.map(_family)
    grouped["deterministic"] = grouped.index.map(_is_deterministic)

    grouped = grouped.round(3).sort_values("score", ascending=False)
    return grouped.reset_index().rename(columns={"prompt_a_key": "agent"})


# ---------------------------------------------------------------------------
# Diagnostic report (stage 0)
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticVerdict:
    model_label: str
    coop_vs_ac: float
    defect_rate_vs_ad_by_round_20: float
    coop_vs_stft: float
    coop_vs_tft: float
    passes: bool
    bucket: str  # "reactive" | "context-insensitive" | "degenerate"
    notes: list[str]

    def render(self) -> str:
        lines = [
            f"Diagnostic verdict for {self.model_label}:",
            f"  coop rate vs always_cooperate_fixed:     {self.coop_vs_ac:.0%} "
            f"(threshold >70%)",
            f"  defect rate vs always_defect_fixed@r20:  {self.defect_rate_vs_ad_by_round_20:.0%} "
            f"(threshold >50%)",
            f"  coop rate vs suspicious_tit_for_tat:     {self.coop_vs_stft:.0%}",
            f"  coop rate vs tft:                        {self.coop_vs_tft:.0%}",
            f"  bucket: {self.bucket}",
            f"  verdict: {'PASS' if self.passes else 'FAIL'}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def diagnostic_report(
    results_dir: str, model_label: str = "(unspecified model)"
) -> DiagnosticVerdict:
    """Evaluate stage 0 (axelrod_diagnostic.yaml) output against the gating
    criteria described in the experiment plan. Reads per-run JSONs for
    round-level granularity (needed for "by round 20" threshold)."""

    def _coop_rate(run_id: str, agent_key: str) -> Optional[float]:
        try:
            run = load_run(results_dir, run_id)
        except FileNotFoundError:
            return None
        history = run.get("history", [])
        if not history:
            return None
        # Agent_A in every diagnostic run is the `minimal` LLM under test.
        return sum(1 for r in history if r[agent_key] == "COOPERATE") / len(history)

    def _defect_rate_first_n(run_id: str, n: int) -> Optional[float]:
        try:
            run = load_run(results_dir, run_id)
        except FileNotFoundError:
            return None
        history = run.get("history", [])[:n]
        if not history:
            return None
        return sum(1 for r in history if r["move_a"] == "DEFECT") / len(history)

    coop_vs_ac = _coop_rate("minimal_vs_always_cooperate_rep0", "move_a") or 0.0
    defect20_vs_ad = _defect_rate_first_n("minimal_vs_always_defect_rep0", 20) or 0.0
    coop_vs_stft = _coop_rate("minimal_vs_suspicious_tft_rep0", "move_a") or 0.0
    coop_vs_tft = _coop_rate("minimal_vs_tft_rep0", "move_a") or 0.0

    notes: list[str] = []
    passes_ac = coop_vs_ac > 0.70
    passes_ad = defect20_vs_ad > 0.50
    if not passes_ac:
        notes.append(
            "Failed always-cooperate test: model is not rewarding a safe opponent."
        )
    if not passes_ad:
        notes.append(
            "Failed always-defect test: model does not punish a hostile opponent "
            "within 20 rounds."
        )

    if passes_ac and passes_ad:
        bucket = "reactive"
    elif not passes_ac and not passes_ad:
        bucket = "degenerate"
    else:
        bucket = "context-insensitive"

    return DiagnosticVerdict(
        model_label=model_label,
        coop_vs_ac=coop_vs_ac,
        defect_rate_vs_ad_by_round_20=defect20_vs_ad,
        coop_vs_stft=coop_vs_stft,
        coop_vs_tft=coop_vs_tft,
        passes=passes_ac and passes_ad,
        bucket=bucket,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Replay harness — the heart of fidelity analysis
# ---------------------------------------------------------------------------

class _ReplayGame:
    """Minimal BaseGame-compatible shim used by the replay harness.

    Exposes exactly the attributes and methods the LLM agent harness touches:
    `name`, `num_rounds`, `current_round`, `scores`, `history`,
    `reasoning_log`, `get_moves`, `get_observation`, `parse_move`,
    `get_payoff_description`, and `log_reasoning`. Crucially, `step` is NOT
    used — we replay move-by-move without the opponent node participating.
    """

    # Only PD is supported by the fidelity matrix today (consistent with the
    # Axelrod experiment). Adding other games is a small extension.
    name = "prisoners_dilemma"
    verbose_history = False

    def __init__(self, num_rounds: int):
        self.num_rounds = num_rounds
        self.current_round = 0
        self.scores = {"Agent_A": 0, "Agent_B": 0}
        self.history: list[dict] = []
        self.reasoning_log: list[dict] = []

    def get_moves(self) -> list[str]:
        return ["COOPERATE", "DEFECT"]

    def get_payoff_description(self) -> str:
        return (
            "  Both COOPERATE: each gets 3\n"
            "  Both DEFECT: each gets 1\n"
            "  One DEFECTS while other COOPERATES: defector gets 5, cooperator gets 0"
        )

    def parse_move(self, text: str) -> str:
        from gtp.games.base import BaseGame
        return BaseGame.parse_move(self, text)

    def _compact_history_lines(self, agent_name: str, last_k: int = 10):
        from gtp.games.base import BaseGame
        return BaseGame._compact_history_lines(self, agent_name, last_k)

    def get_observation(self, agent_name: str) -> str:
        from gtp.games.base import BaseGame
        return BaseGame.get_observation(self, agent_name)

    def log_reasoning(self, round_num: int, agent: str, messages: list):
        from gtp.games.base import BaseGame
        return BaseGame.log_reasoning(self, round_num, agent, messages)


def replay_llm_strategy(
    prompt_key: str,
    recorded_run: dict,
    model_config_dict: dict,
    *,
    agent_name: str = "Agent_A",
    max_tool_calls: int = 3,
) -> list[str]:
    """Replay an LLM strategy agent against a recorded opponent trajectory.

    Given a fixed-vs-minimal run, we know exactly what minimal played in each
    round. This function constructs the LLM twin (`prompt_key`), feeds it the
    same round-by-round opponent moves (and whatever my-side history the LLM
    would have produced if it had been playing those rounds — see caveat
    below), and returns the LLM's move sequence.

    Caveat: at round N, the LLM sees ITS OWN past moves, which in a true
    online match would be the LLM's own outputs (possibly different from the
    fixed twin's). In the replay we use the LLM's rolling outputs, not the
    fixed twin's — so this measures "what would the LLM do if it had been
    playing against minimal's recorded trajectory". That's the right
    counterfactual for fidelity: if LLM-TFT would have produced the same
    moves as fixed-TFT given the same opponent sequence, the LLM executes
    TFT faithfully.

    `recorded_run` should be a dict from `load_run` where side B played
    `minimal` and side A played the fixed twin. Only `history` is consumed.
    """
    from gtp.agents.builder import build_agent_subgraph
    from gtp.agents.tools import make_tools
    from langchain_core.messages import HumanMessage

    history = recorded_run["history"]
    num_rounds = len(history)

    # The opponent key in the recorded history depends on which side the
    # fixed twin occupied. In our experiments it's Agent_A = fixed/llm and
    # Agent_B = minimal, so opponent_key = move_b. Keep explicit for safety.
    is_a = agent_name == "Agent_A"
    opp_round_key = "move_b" if is_a else "move_a"
    my_round_key = "move_a" if is_a else "move_b"

    game = _ReplayGame(num_rounds=num_rounds)

    agent_cfg = AgentConfig(name=agent_name)
    # Resolve prompt template (mirrors what build_game_graph does online).
    if prompt_key in PROMPT_REGISTRY:
        agent_cfg.system_prompt = format_prompt(
            prompt_key, game.name, game.get_moves(), num_rounds=num_rounds
        )
    else:
        raise KeyError(f"{prompt_key!r} is not in PROMPT_REGISTRY")

    opp_cfg = AgentConfig(name="Agent_B" if is_a else "Agent_A")

    from gtp.config import ModelConfig
    agent_cfg.model_config = ModelConfig(**model_config_dict)
    agent_cfg.max_tool_calls = max_tool_calls

    tools = make_tools(game, agent_name, agent_cfg, opp_cfg)
    agent_graph = build_agent_subgraph(agent_cfg, tools)

    llm_moves: list[str] = []
    for round_idx, recorded in enumerate(history):
        opp_move = recorded[opp_round_key]
        # At this point game.history has round_idx entries; the LLM sees them.
        observation = game.get_observation(agent_name)
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=observation)]},
            config={"recursion_limit": 2 * max_tool_calls + 10},
        )
        # Parse move from the final AI message; match builder.py semantics.
        move = None
        for msg in reversed(result["messages"]):
            content = getattr(msg, "content", "")
            if content:
                try:
                    move = game.parse_move(content)
                    break
                except ValueError:
                    continue
        if move is None:
            move = "COOPERATE"  # same fallback as builder.py
        llm_moves.append(move)

        # Advance the replay game state as if the match had happened.
        # Source the payoff matrix from the live game class so the replay
        # cannot drift if PD's payoffs are ever retuned.
        from gtp.games.prisoners_dilemma import PrisonersDilemmaGame
        game.current_round += 1
        move_a = move if is_a else opp_move
        move_b = opp_move if is_a else move
        p_a, p_b = PrisonersDilemmaGame().resolve(move_a, move_b)
        game.scores["Agent_A"] += p_a
        game.scores["Agent_B"] += p_b
        game.history.append(
            {"round": game.current_round, "move_a": move_a, "move_b": move_b,
             "payoff_a": p_a, "payoff_b": p_b}
        )

    return llm_moves


# ---------------------------------------------------------------------------
# Fidelity analysis
# ---------------------------------------------------------------------------

def _replay_agreement_for_run(
    results_dir: str, fixed_run_id: str, llm_key: str, model_config_dict: dict
) -> Optional[float]:
    """Replay `llm_key` against the recorded opponent trajectory in
    `fixed_run_id` and return the round-by-round move-agreement rate."""
    fixed_run = load_run(results_dir, fixed_run_id)
    fixed_moves = [r["move_a"] for r in fixed_run["history"]]
    if not fixed_moves:
        return None
    llm_moves = replay_llm_strategy(llm_key, fixed_run, model_config_dict)
    agree = sum(1 for x, y in zip(fixed_moves, llm_moves) if x == y)
    return agree / len(fixed_moves)


def fidelity_analysis(
    results_dir: str,
    model_config_dict: dict,
    *,
    families: Optional[list[str]] = None,
    do_replay: bool = True,
) -> pd.DataFrame:
    """Compute strategy fidelity per Axelrod family present in results_dir.

    Three metrics per family (per the plan):
      - replay_agreement: round-by-round move-agreement rate when LLM-X is
        re-run against the opponent trajectory recorded from fixed-X vs
        minimal. Only computed when do_replay=True. This is the headline
        fidelity number. For stochastic families with multiple reps, the
        replay is run once per recorded fixed-twin trajectory and the
        agreement rates are averaged — rep 0 alone would throw away 4/5 of
        the trajectory information when reps=5.
      - first_divergence_online: earliest round (1-indexed) where LLM-X vs
        minimal and fixed-X vs minimal disagreed in the actual online
        matches. NaN if they never disagreed within the match length.
      - aggregate_coop_gap: absolute difference in overall cooperation rate
        between fixed-X-vs-minimal and LLM-X-vs-minimal (online).

    Returns a DataFrame indexed by family.
    """
    df = load_experiment_results(results_dir)
    if families is None:
        families_present = sorted(
            {_family(k) for k in df["prompt_a_key"].unique()} & set(_AXELROD_FAMILIES)
        )
    else:
        families_present = list(families)

    rows = []
    for fam in families_present:
        fixed_key = f"{fam}_fixed"
        llm_key = f"{fam}_llm"

        fixed_subset = df[df["prompt_a_key"] == fixed_key]
        llm_subset = df[df["prompt_a_key"] == llm_key]
        if fixed_subset.empty or llm_subset.empty:
            continue

        fixed_coop = fixed_subset["first_move_rate_a"].mean()
        llm_coop = llm_subset["first_move_rate_a"].mean()
        coop_gap = abs(float(fixed_coop) - float(llm_coop))

        first_divergence = _online_first_divergence(
            results_dir, fixed_subset.iloc[0]["run_id"], llm_subset.iloc[0]["run_id"]
        )

        replay_agreement = None
        replay_n_reps = 0
        if do_replay:
            # Average across every recorded fixed-twin trajectory in this
            # family. Deterministic families collapse to 1 rep automatically.
            agreements: list[float] = []
            for fixed_run_id in fixed_subset["run_id"].tolist():
                ag = _replay_agreement_for_run(
                    results_dir, fixed_run_id, llm_key, model_config_dict
                )
                if ag is not None:
                    agreements.append(ag)
            if agreements:
                replay_agreement = sum(agreements) / len(agreements)
                replay_n_reps = len(agreements)

        rows.append({
            "family": fam,
            "fixed_coop_rate": round(float(fixed_coop), 3),
            "llm_coop_rate": round(float(llm_coop), 3),
            "aggregate_coop_gap": round(coop_gap, 3),
            "first_divergence_online": first_divergence,
            "replay_agreement": (
                round(replay_agreement, 3) if replay_agreement is not None else None
            ),
            "replay_n_reps": replay_n_reps,
        })

    result = pd.DataFrame(rows).set_index("family")
    return result.sort_values("replay_agreement", ascending=False, na_position="last")


def _online_first_divergence(
    results_dir: str, fixed_run_id: str, llm_run_id: str
) -> Optional[int]:
    fixed = load_run(results_dir, fixed_run_id)["history"]
    llm = load_run(results_dir, llm_run_id)["history"]
    for i, (f, l) in enumerate(zip(fixed, llm), start=1):
        if f["move_a"] != l["move_a"]:
            return i
    return None


# ---------------------------------------------------------------------------
# Cooperation-vs-minimal plot (14 x 2 grid per Axelrod family)
# ---------------------------------------------------------------------------

def cooperation_vs_minimal_plot(df: pd.DataFrame, title: Optional[str] = None):
    """Bar chart: cooperation rate against `minimal` per strategy family,
    fixed twin vs LLM twin, with a delta row visually showing the gap.

    Returns a matplotlib Figure. `df` must be the results.jsonl DataFrame
    from one of the Axelrod experiments (or both concatenated).
    """
    rows = []
    for fam in _AXELROD_FAMILIES:
        fixed_key = f"{fam}_fixed"
        llm_key = f"{fam}_llm"
        fixed_row = df[df["prompt_a_key"] == fixed_key]
        llm_row = df[df["prompt_a_key"] == llm_key]
        if fixed_row.empty and llm_row.empty:
            continue
        fixed_coop = fixed_row["first_move_rate_a"].mean() if not fixed_row.empty else None
        llm_coop = llm_row["first_move_rate_a"].mean() if not llm_row.empty else None
        rows.append({"family": fam, "fixed": fixed_coop, "llm": llm_coop})

    plot_df = pd.DataFrame(rows).set_index("family")

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(plot_df) + 3), 5))
    x = range(len(plot_df))
    width = 0.38

    ax.bar([i - width / 2 for i in x], plot_df["fixed"], width=width, label="fixed twin")
    ax.bar([i + width / 2 for i in x], plot_df["llm"], width=width, label="LLM twin")

    # Draw connecting gap whiskers so the fixed/LLM delta is visible.
    for i, (f, l) in enumerate(zip(plot_df["fixed"], plot_df["llm"])):
        if pd.notna(f) and pd.notna(l):
            ax.plot([i - width / 2, i + width / 2], [f, l], "k-", alpha=0.4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cooperation rate vs minimal")
    ax.set_title(title or "Strategy-fidelity cooperation gap (fixed vs LLM)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-round & reasoning-trace analyses
# ---------------------------------------------------------------------------

def _iter_run_files(results_dir: str):
    runs_dir = os.path.join(results_dir, "runs")
    for path in sorted(glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            yield json.load(f)


_GAME_FIRST_MOVES = {
    "prisoners_dilemma": "COOPERATE",
    "chicken": "SWERVE",
    "stag_hunt": "STAG",
}


def _infer_first_move(cfg: dict) -> str:
    """Infer the cooperative move from the game name in the run config."""
    game = cfg.get("game", "prisoners_dilemma")
    return _GAME_FIRST_MOVES.get(game, "COOPERATE")


def load_per_round_df(results_dir: str) -> pd.DataFrame:
    """Flatten every run's history into one row per (run_id, round).

    Columns: run_id, condition_id, prompt_a_key, prompt_b_key, round,
    move_a, move_b, payoff_a, payoff_b, coop_a, coop_b,
    num_rounds, rounds_remaining, is_final_round, first_move.
    """
    rows = []
    for data in _iter_run_files(results_dir):
        cfg = data.get("config", {})
        run_id = data["run_id"]
        condition_id = cfg.get("condition_id", "")
        prompt_a_key = cfg.get("prompt_a_key", "")
        prompt_b_key = cfg.get("prompt_b_key", "")
        first_move = data.get("first_move_name") or _infer_first_move(cfg)
        history = data.get("history", [])
        num_rounds = len(history)
        for r in history:
            rnd = r["round"]
            rows.append({
                "run_id": run_id,
                "condition_id": condition_id,
                "prompt_a_key": prompt_a_key,
                "prompt_b_key": prompt_b_key,
                "round": rnd,
                "move_a": r["move_a"],
                "move_b": r["move_b"],
                "payoff_a": r["payoff_a"],
                "payoff_b": r["payoff_b"],
                "coop_a": r["move_a"] == first_move,
                "coop_b": r["move_b"] == first_move,
                "num_rounds": num_rounds,
                "rounds_remaining": num_rounds - rnd,
                "is_final_round": rnd == num_rounds,
                "first_move": first_move,
            })
    return pd.DataFrame(rows)


def _messages_to_text(messages: list, roles: set[str] | None = None) -> str:
    """Concatenate message contents. If `roles` is given, include only those roles.

    The reasoning_log stores a full conversation (HumanMessage observation +
    AI response + optional tool calls). For reasoning analysis we usually
    want just the AI's own text, not the observation that was fed in.
    """
    parts = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            role = getattr(m, "type", "")
            content = getattr(m, "content", "")
        if roles is not None and role not in roles:
            continue
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def load_per_reasoning_df(results_dir: str) -> pd.DataFrame:
    """Flatten reasoning_log into one row per (run_id, round, agent).

    Columns: run_id, condition_id, prompt_a_key, prompt_b_key,
    round, agent, content.
    """
    rows = []
    for data in _iter_run_files(results_dir):
        cfg = data.get("config", {})
        run_id = data["run_id"]
        condition_id = cfg.get("condition_id", "")
        prompt_a_key = cfg.get("prompt_a_key", "")
        prompt_b_key = cfg.get("prompt_b_key", "")
        for entry in data.get("reasoning_log", []):
            rows.append({
                "run_id": run_id,
                "condition_id": condition_id,
                "prompt_a_key": prompt_a_key,
                "prompt_b_key": prompt_b_key,
                "round": entry.get("round"),
                "agent": entry.get("agent"),
                "content": _messages_to_text(entry.get("messages", []), roles={"ai", "AIMessage"}),
            })
    return pd.DataFrame(rows)


def plot_cooperation_by_round(
    df_rounds: pd.DataFrame,
    group_by: str = "condition_id",
    agent: str = "both",
):
    """Line plot of cooperation rate by round, one line per group.

    `agent` is "a", "b", or "both" (average of A and B).
    """
    if df_rounds.empty:
        raise ValueError("df_rounds is empty")

    df = df_rounds.copy()
    if agent == "a":
        df["coop"] = df["coop_a"].astype(float)
    elif agent == "b":
        df["coop"] = df["coop_b"].astype(float)
    else:
        df["coop"] = (df["coop_a"].astype(float) + df["coop_b"].astype(float)) / 2.0

    grouped = df.groupby([group_by, "round"])["coop"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, sub in grouped.groupby(group_by):
        ax.plot(sub["round"], sub["coop"], marker="o", markersize=4, label=str(key))
    first_move = df_rounds["first_move"].iloc[0] if "first_move" in df_rounds.columns else "First move"
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{first_move} rate")
    ax.set_title(f"{first_move} rate by round (agent={agent}, grouped by {group_by})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    return fig


def plot_move_heatmap(df_rounds: pd.DataFrame):
    """Heatmap: one row per (run, agent), cells colored by move.

    Green = cooperation (first move), red = defection.
    """
    if df_rounds.empty:
        raise ValueError("df_rounds is empty")

    run_ids = sorted(df_rounds["run_id"].unique())
    max_round = int(df_rounds["round"].max())

    labels = []
    matrix = []
    for run_id in run_ids:
        sub = df_rounds[df_rounds["run_id"] == run_id].sort_values("round")
        row_a = [1 if v else 0 for v in sub["coop_a"].tolist()]
        row_b = [1 if v else 0 for v in sub["coop_b"].tolist()]
        row_a += [np.nan] * (max_round - len(row_a))
        row_b += [np.nan] * (max_round - len(row_b))
        matrix.append(row_a)
        matrix.append(row_b)
        labels.append(f"{run_id} [A]")
        labels.append(f"{run_id} [B]")

    arr = np.array(matrix, dtype=float)
    fig_h = max(4, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(max(8, max_round * 0.4), fig_h))
    cmap = ListedColormap(["#d62728", "#2ca02c"])  # red=defect, green=coop
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(max_round))
    ax.set_xticklabels([str(i + 1) for i in range(max_round)], fontsize=7)
    ax.set_xlabel("Round")
    first_move = df_rounds["first_move"].iloc[0] if "first_move" in df_rounds.columns else "coop"
    ax.set_title(f"Move heatmap (green = {first_move}, red = other)")
    plt.tight_layout()
    return fig


def last_n_rounds_table(df_rounds: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Per-run table of the final N rounds' moves.

    Columns: run_id, condition_id, round_{k}_a, round_{k}_b for k in last N rounds.
    """
    if df_rounds.empty:
        return pd.DataFrame()
    rows = []
    for run_id, sub in df_rounds.groupby("run_id"):
        sub = sub.sort_values("round")
        tail = sub.tail(n)
        row = {
            "run_id": run_id,
            "condition_id": sub["condition_id"].iloc[0],
        }
        for _, r in tail.iterrows():
            rnd = int(r["round"])
            row[f"r{rnd}_a"] = r["move_a"]
            row[f"r{rnd}_b"] = r["move_b"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_score_trajectories_from_df(df_rounds: pd.DataFrame, run_ids: list[str] | None = None):
    """Like plot_score_trajectories but operates on the flat per-round DataFrame.

    If run_ids is None, plots all runs.
    """
    if df_rounds.empty:
        raise ValueError("df_rounds is empty")
    if run_ids is None:
        run_ids = sorted(df_rounds["run_id"].unique())

    n = len(run_ids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, run_id in zip(axes[0], run_ids):
        sub = df_rounds[df_rounds["run_id"] == run_id].sort_values("round").copy()
        sub["cum_a"] = sub["payoff_a"].cumsum()
        sub["cum_b"] = sub["payoff_b"].cumsum()
        ax.plot(sub["round"], sub["cum_a"], "b-o", label="Agent A", markersize=4)
        ax.plot(sub["round"], sub["cum_b"], "r-o", label="Agent B", markersize=4)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative score")
        ax.set_title(run_id, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# Regex patterns signaling stated intent. Match case-insensitive.
_INTENT_PATTERNS = [
    (re.compile(r"\b(i will|i'll|i should|i am going to|i plan to|my (?:final )?(?:choice|decision|move) (?:is|will be))\s+(cooperate|defect)\b", re.IGNORECASE), 2),
    (re.compile(r"\bthe (?:rational|best|optimal) (?:choice|move|action|decision)\s+(?:is|would be)\s+(?:to\s+)?(cooperate|defect)\b", re.IGNORECASE), 1),
    (re.compile(r"\b(?:continue|keep)\s+(cooperating|defecting)\b", re.IGNORECASE), 1),
    (re.compile(r"\b(?:switch|start)\s+(?:to\s+)?(cooperating|defecting)\b", re.IGNORECASE), 1),
]


def _extract_intent(text: str) -> str | None:
    """Return 'COOPERATE', 'DEFECT', or None based on regex matches.

    If multiple patterns disagree, returns None (ambiguous).
    """
    intents = set()
    for pattern, group in _INTENT_PATTERNS:
        for m in pattern.finditer(text or ""):
            verb = m.group(group).lower()
            if verb.startswith("cooperat"):
                intents.add("COOPERATE")
            elif verb.startswith("defect"):
                intents.add("DEFECT")
    if len(intents) == 1:
        return intents.pop()
    return None


def find_reasoning_action_mismatches(
    df_reasoning: pd.DataFrame,
    df_rounds: pd.DataFrame,
) -> pd.DataFrame:
    """Return rows where the agent's stated intent differs from the action played.

    Columns: run_id, condition_id, round, agent, intent, action, snippet.
    """
    if df_reasoning.empty or df_rounds.empty:
        return pd.DataFrame(columns=["run_id", "condition_id", "round", "agent", "intent", "action", "snippet"])

    rows = []
    for _, r in df_reasoning.iterrows():
        intent = _extract_intent(r["content"])
        if intent is None:
            continue

        move_col = "move_a" if r["agent"] == "Agent_A" else "move_b"
        match = df_rounds[
            (df_rounds["run_id"] == r["run_id"]) & (df_rounds["round"] == r["round"])
        ]
        if match.empty:
            continue
        action = match.iloc[0][move_col]

        if intent != action:
            snippet = (r["content"] or "")[:300].replace("\n", " ")
            rows.append({
                "run_id": r["run_id"],
                "condition_id": r["condition_id"],
                "round": r["round"],
                "agent": r["agent"],
                "intent": intent,
                "action": action,
                "snippet": snippet,
            })
    return pd.DataFrame(rows)


_ENDGAME_PATTERNS = [
    re.compile(r"\blast round\b", re.IGNORECASE),
    re.compile(r"\bfinal round\b", re.IGNORECASE),
    re.compile(r"\bendgame\b", re.IGNORECASE),
    re.compile(r"\bend of (?:the )?game\b", re.IGNORECASE),
    re.compile(r"\bbackward induct", re.IGNORECASE),
    re.compile(r"\blast (?:few )?rounds?\b", re.IGNORECASE),
    re.compile(r"\bgame ends?\b", re.IGNORECASE),
]


def count_endgame_references(df_reasoning: pd.DataFrame) -> pd.DataFrame:
    """Count endgame-related phrases per (run_id, round, agent).

    Returns a DataFrame filtered to only rows with at least one match.
    Columns: run_id, condition_id, round, agent, endgame_count, matches.
    """
    if df_reasoning.empty:
        return pd.DataFrame(columns=["run_id", "condition_id", "round", "agent", "endgame_count", "matches"])

    rows = []
    for _, r in df_reasoning.iterrows():
        text = r["content"] or ""
        found = []
        for pat in _ENDGAME_PATTERNS:
            found.extend(pat.findall(text))
        if found:
            rows.append({
                "run_id": r["run_id"],
                "condition_id": r["condition_id"],
                "round": r["round"],
                "agent": r["agent"],
                "endgame_count": len(found),
                "matches": "; ".join(found),
            })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without requiring tabulate."""
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(
            ("" if pd.isna(v) else str(v)) for v in row.tolist()
        ) + " |")
    return "\n".join([header, sep] + body)


def summarize_experiment(results_dir: str) -> str:
    """Return a markdown summary of the experiment."""
    summary = load_experiment_results(results_dir)
    df_rounds = load_per_round_df(results_dir)

    lines = []
    name = os.path.basename(os.path.normpath(results_dir))
    lines.append(f"# Experiment Report: {name}\n")
    lines.append(f"**Runs:** {len(summary)}  ")
    if not df_rounds.empty:
        lines.append(f"**Rounds per run:** {int(df_rounds['num_rounds'].iloc[0])}  ")
        first_move = df_rounds["first_move"].iloc[0]
        lines.append(f"**First move (= cooperation):** {first_move}\n")

    if not summary.empty:
        lines.append("## Conditions\n")
        cols = ["condition_id", "prompt_a_key", "prompt_b_key",
                "score_a", "score_b", "first_move_rate_a", "first_move_rate_b", "winner"]
        available = [c for c in cols if c in summary.columns]
        lines.append(_df_to_markdown(summary[available]))
        lines.append("")

    if not df_rounds.empty:
        lines.append("## Last 3 rounds per run\n")
        tail = last_n_rounds_table(df_rounds, n=3)
        lines.append(_df_to_markdown(tail))
        lines.append("")

        final = df_rounds[df_rounds["is_final_round"]]
        lines.append("## Final-round moves\n")
        final_tbl = final[["run_id", "condition_id", "move_a", "move_b"]].copy()
        lines.append(_df_to_markdown(final_tbl))
        lines.append("")

        lines.append("## Aggregate cooperation rates\n")
        agg = df_rounds.groupby("condition_id").agg(
            coop_a=("coop_a", "mean"),
            coop_b=("coop_b", "mean"),
        ).round(3).reset_index()
        lines.append(_df_to_markdown(agg))
        lines.append("")

    lines.append("## Plots\n")
    lines.append("- `plots/cooperation_by_round.png`")
    lines.append("- `plots/move_heatmap.png`")
    lines.append("- `plots/score_trajectories.png`")
    lines.append("")
    lines.append("## Companion files\n")
    lines.append("- `reasoning_mismatches.csv` — candidates where stated intent differed from the move played")
    lines.append("- `endgame_refs.csv` — reasoning entries mentioning game ending")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy analyses (kept for backward compatibility with pilot/transparency)
# ---------------------------------------------------------------------------

def first_move_rate_table(
    df: pd.DataFrame, group_by: list[str]
) -> pd.DataFrame:
    """Pivot table of mean first-move rates grouped by conditions."""
    return (
        df.groupby(group_by)
        .agg(
            first_move_rate_a_mean=("first_move_rate_a", "mean"),
            first_move_rate_a_std=("first_move_rate_a", "std"),
            first_move_rate_b_mean=("first_move_rate_b", "mean"),
            first_move_rate_b_std=("first_move_rate_b", "std"),
            score_a_mean=("score_a", "mean"),
            score_b_mean=("score_b", "mean"),
            n=("run_id", "count"),
        )
        .round(3)
    )


def plot_score_trajectories(results_dir: str, run_ids: list[str]):
    """Plot cumulative score trajectories for selected runs."""
    fig, axes = plt.subplots(1, len(run_ids), figsize=(6 * len(run_ids), 5), squeeze=False)
    for ax, run_id in zip(axes[0], run_ids):
        data = load_run(results_dir, run_id)
        df = pd.DataFrame(data["history"])
        df["cumulative_a"] = df["payoff_a"].cumsum()
        df["cumulative_b"] = df["payoff_b"].cumsum()
        ax.plot(df["round"], df["cumulative_a"], "b-o", label="Agent A", markersize=4)
        ax.plot(df["round"], df["cumulative_b"], "r-o", label="Agent B", markersize=4)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Score")
        ax.set_title(run_id, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_transparency_effect(df: pd.DataFrame):
    """Bar chart comparing first-move rates across transparency conditions."""
    from gtp.config import transparency_label

    df = df.copy()
    df["transparency"] = df.apply(
        lambda r: transparency_label(r["transparency_a"], r["transparency_b"]),
        axis=1,
    )
    grouped = df.groupby("transparency").agg(
        rate_a=("first_move_rate_a", "mean"),
        rate_b=("first_move_rate_b", "mean"),
    )
    move_name = df["first_move_name"].iloc[0] if "first_move_name" in df.columns and len(df) > 0 else "First Move"
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel(f"Mean {move_name} Rate")
    ax.set_title(f"{move_name} Rate by Transparency Condition")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig
