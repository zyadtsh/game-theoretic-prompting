"""Configuration dataclasses, prompt registry, and YAML experiment loader.

Defines ModelConfig, AgentConfig, GameRunConfig, and ExperimentConfig.
PROMPT_REGISTRY maps strategy names (e.g. "cooperative") to prompt templates.
load_experiment() parses YAML files with optional grid expansion.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import yaml

# Default config for LM Studio local inference
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "meta-llama-3.1-8b-instruct"


@dataclass
class ModelConfig:
    model: str = LM_STUDIO_MODEL
    base_url: Optional[str] = LM_STUDIO_BASE_URL
    api_key: str = "lm-studio"
    temperature: float = 0.7


@dataclass
class AgentConfig:
    name: str
    system_prompt: str = "You are playing an iterated game."
    can_see_opponent_prompt: bool = False
    max_tool_calls: int = 3
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class GameRunConfig:
    game: str = "prisoners_dilemma"
    agent_a: AgentConfig = field(default_factory=lambda: AgentConfig(name="Agent_A"))
    agent_b: AgentConfig = field(default_factory=lambda: AgentConfig(name="Agent_B"))
    num_rounds: int = 10
    condition_id: str = ""


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    repetitions: int = 3
    runs: list[GameRunConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, str] = {
    "cooperative": (
        "You are playing an iterated {game_name} game against another AI agent. "
        "You want to maximize your total score over all rounds. "
        "Consider the benefits of mutual cooperation and building trust over time. "
        "Use your tools to review the game history and scores before deciding. "
        "After reasoning, state your final decision clearly as either {moves}."
    ),
    "competitive": (
        "You are playing an iterated {game_name} game against another AI agent. "
        "Your primary goal is to WIN by scoring higher than your opponent. "
        "Be strategic and exploit any predictable patterns you observe. "
        "Use your tools to analyze game history, scores, and your opponent's instructions. "
        "After reasoning, state your final decision clearly as either {moves}."
    ),
    "neutral": (
        "You are playing an iterated {game_name} game against another AI agent. "
        "Analyze the situation carefully and choose the best action each round. "
        "Use your tools to review the game history and scores before deciding. "
        "After reasoning, state your final decision clearly as either {moves}."
    ),
    "tit_for_tat": (
        "You are playing an iterated {game_name} game against another AI agent. "
        "Start friendly, then mirror your opponent's previous move. Adapt if you notice patterns. "
        "Use your tools to review the game history before deciding. "
        "After reasoning, state your final decision clearly as either {moves}."
    ),
    "deceptive": (
        "You are playing an iterated {game_name} game against another AI agent. "
        "Appear cooperative to build trust, then exploit your opponent at key moments "
        "to maximize your advantage. "
        "Use your tools to analyze game history, scores, and your opponent's instructions. "
        "After reasoning, state your final decision clearly as either {moves}."
    ),
}


def format_prompt(prompt_key: str, game_name: str, moves: list[str]) -> str:
    """Resolve a prompt template from the registry."""
    template = PROMPT_REGISTRY[prompt_key]
    moves_str = " or ".join(moves)
    return template.format(game_name=game_name, moves=moves_str)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _build_model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        model=raw.get("model", LM_STUDIO_MODEL),
        base_url=raw.get("base_url", LM_STUDIO_BASE_URL),
        api_key=raw.get("api_key", "lm-studio"),
        temperature=raw.get("temperature", 0.7),
    )


def load_experiment(path: str) -> ExperimentConfig:
    """Load a YAML experiment file. Supports both explicit `runs` and `grid` modes."""
    with open(path) as f:
        data = yaml.safe_load(f)

    defaults = data.get("defaults", {})
    default_game = defaults.get("game", "prisoners_dilemma")
    default_num_rounds = defaults.get("num_rounds", 10)
    default_model = _build_model_config(defaults.get("model", {}))

    config = ExperimentConfig(
        name=data["name"],
        description=data.get("description", ""),
        repetitions=data.get("repetitions", 3),
    )

    # Explicit runs mode
    if "runs" in data:
        for r in data["runs"]:
            mc = _build_model_config(r.get("model", {})) if "model" in r else default_model
            agent_a_raw = r.get("agent_a", {})
            agent_b_raw = r.get("agent_b", {})
            run = GameRunConfig(
                game=r.get("game", default_game),
                num_rounds=r.get("num_rounds", default_num_rounds),
                agent_a=AgentConfig(
                    name="Agent_A",
                    system_prompt=agent_a_raw.get("prompt", "cooperative"),
                    can_see_opponent_prompt=agent_a_raw.get("can_see_opponent", False),
                    model_config=mc,
                ),
                agent_b=AgentConfig(
                    name="Agent_B",
                    system_prompt=agent_b_raw.get("prompt", "cooperative"),
                    can_see_opponent_prompt=agent_b_raw.get("can_see_opponent", False),
                    model_config=mc,
                ),
                condition_id=r.get("id", ""),
            )
            config.runs.append(run)
        return config

    # Grid mode — Cartesian product
    grid = data.get("grid", {})
    prompts_a = grid.get("prompt_a", ["cooperative"])
    prompts_b = grid.get("prompt_b", ["cooperative"])
    transparencies = grid.get("transparency", [{"a_sees_b": False, "b_sees_a": False}])
    games = grid.get("game", [default_game])
    if isinstance(games, str):
        games = [games]

    for game, pa, pb, tr in itertools.product(games, prompts_a, prompts_b, transparencies):
        a_sees_b = tr.get("a_sees_b", False)
        b_sees_a = tr.get("b_sees_a", False)

        # Build transparency label
        if a_sees_b and b_sees_a:
            tr_label = "both_see"
        elif not a_sees_b and not b_sees_a:
            tr_label = "both_blind"
        elif a_sees_b:
            tr_label = "a_sees_b"
        else:
            tr_label = "b_sees_a"

        condition_id = f"{game}_{pa}_{pb}_{tr_label}"

        run = GameRunConfig(
            game=game,
            num_rounds=default_num_rounds,
            agent_a=AgentConfig(
                name="Agent_A",
                system_prompt=pa,  # resolved to full text at run time
                can_see_opponent_prompt=a_sees_b,
                model_config=default_model,
            ),
            agent_b=AgentConfig(
                name="Agent_B",
                system_prompt=pb,
                can_see_opponent_prompt=b_sees_a,
                model_config=default_model,
            ),
            condition_id=condition_id,
        )
        config.runs.append(run)

    return config
