"""Configuration dataclasses, prompt registry, and YAML experiment loader.

Defines ModelConfig, AgentConfig, GameRunConfig, and ExperimentConfig.
PROMPT_REGISTRY maps strategy names (e.g. "cooperative") to prompt templates.
load_experiment() parses YAML files with optional grid expansion.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

# Default config for LM Studio local inference
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "meta-llama-3.1-8b-instruct"

# OpenRouter cloud inference
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"


@dataclass
class ModelConfig:
    model: str = LM_STUDIO_MODEL
    base_url: Optional[str] = LM_STUDIO_BASE_URL
    api_key: str = "lm-studio"
    temperature: float = 0.7
    # Optional OpenRouter-style fallback routing. When set, requests are sent
    # with `extra_body={"models": [model, *fallback_models]}` and the provider
    # automatically advances to the next model on rate-limit / provider errors.
    # Ignored by non-OpenRouter endpoints that don't understand the field.
    fallback_models: Optional[list[str]] = None
    # Optional OpenRouter reasoning-token opt-in. When set, the value is passed
    # through as `extra_body={"reasoning": ...}` on every request, causing the
    # provider to emit its chain-of-thought in a `reasoning` field on the
    # response (visible in LangSmith traces). Accepts either `True` (shorthand
    # for `{"enabled": True}`) or a full dict like `{"effort": "high"}` /
    # `{"max_tokens": 2000}`. See https://openrouter.ai/docs/use-cases/reasoning-tokens
    reasoning: Optional[dict | bool] = None


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
    "minimal": (
        "You are interacting with an environment. "
        "You want to maximize your score. "
        "State your decision as either {moves}."
    ),
    "always_cooperate_llm": (
        "You are playing a game. "
        "Always choose {first_move} every round. "
        "State your decision as either {moves}."
    ),
    "always_defect_llm": (
        "You are playing a game. "
        "Always choose {second_move} every round. "
        "State your decision as either {moves}."
    ),
    "suspicious_tit_for_tat_llm": (
        "You are playing a game. "
        "On the first round, choose {second_move}. "
        "On every subsequent round, choose whatever your opponent chose last round. "
        "State your decision as either {moves}."
    ),
    "minimal_endgame": (
        "You are interacting with an environment. "
        "You want to maximize your score. "
        "The game lasts exactly {num_rounds} rounds. "
        "State your decision as either {moves}."
    ),
    "always_cooperate_llm_endgame": (
        "You are playing a game. "
        "The game lasts exactly {num_rounds} rounds. "
        "Always choose {first_move} every round. "
        "State your decision as either {moves}."
    ),
    "always_defect_llm_endgame": (
        "You are playing a game. "
        "The game lasts exactly {num_rounds} rounds. "
        "Always choose {second_move} every round. "
        "State your decision as either {moves}."
    ),
    "suspicious_tit_for_tat_llm_endgame": (
        "You are playing a game. "
        "The game lasts exactly {num_rounds} rounds. "
        "On the first round, choose {second_move}. "
        "On every subsequent round, choose whatever your opponent chose last round. "
        "State your decision as either {moves}."
    ),
}


# ---------------------------------------------------------------------------
# Axelrod 1980 first-tournament strategy prompts.
#
# These 14 prompts ask an LLM to execute a named Axelrod first-tournament
# strategy in natural language. The fixed-agent twins (FIXED_STRATEGY_REGISTRY
# keys *_fixed) are the ground-truth implementations wrapped from the
# `axelrod` library; these *_llm keys are the LLM-prompted test subjects.
#
# Design choices:
# - Each prompt uses the same scaffold so across-strategy differences are
#   purely about the strategy description, not the harness.
# - COOPERATE / DEFECT are pre-baked as literals; no format-string variables
#   in the body, so format_prompt's injected values don't accidentally collide
#   with the strategy text.
# - Descriptions are expanded to unambiguous rules rather than copying the
#   original paper's one-line summaries.
# ---------------------------------------------------------------------------

_AXELROD_PROMPT_TEMPLATE = (
    "You are playing iterated prisoner's dilemma. On each round, "
    "choose COOPERATE or DEFECT.\n\n"
    "Your assigned strategy is: {strategy_name}\n\n"
    "Strategy description:\n{description}\n\n"
    "You will be given the history of the match so far. Play strictly "
    "according to the strategy above. Do not deviate even if it seems "
    "suboptimal. After reasoning, state your final decision clearly as "
    "either COOPERATE or DEFECT."
)


def _axl_prompt(strategy_name: str, description: str) -> str:
    # Build once at import time. We deliberately escape any literal braces in
    # the description so format_prompt (called later) doesn't try to resolve
    # them as template variables.
    safe_desc = description.replace("{", "{{").replace("}", "}}")
    safe_name = strategy_name.replace("{", "{{").replace("}", "}}")
    return _AXELROD_PROMPT_TEMPLATE.format(
        strategy_name=safe_name,
        description=safe_desc,
    )


_AXELROD_LLM_PROMPTS: dict[str, tuple[str, str]] = {
    "tft_llm": (
        "Tit For Tat",
        "Cooperate on the first round. On every subsequent round, play "
        "whatever your opponent played in the previous round. That is: if "
        "your opponent's last move was COOPERATE, you COOPERATE now; if it "
        "was DEFECT, you DEFECT now. Do not deviate from this rule under "
        "any circumstance."
    ),
    "tideman_chieruzzi_llm": (
        "Tideman and Chieruzzi",
        "Start by cooperating, and generally play tit-for-tat: mirror the "
        "opponent's last move. When the opponent defects, you enter a "
        "retaliation phase where you defect for a number of rounds equal to "
        "the number of times the opponent has triggered retaliation so far "
        "(1 the first time, 2 the second time, etc.). If you are doing much "
        "worse than the opponent (behind by at least 10 points and the "
        "opponent has just defected twice in a row after cooperating), treat "
        "the match as a fresh start: cooperate for two rounds before "
        "resuming tit-for-tat."
    ),
    "nydegger_llm": (
        "Nydegger",
        "For the first three rounds, play tit-for-tat with one exception: "
        "if you cooperated on round 1 while your opponent defected, defect "
        "on round 2 and cooperate on round 3. From round 4 onward, decide "
        "using the last three rounds of history. Encode each round as a "
        "number: 0 if both cooperated, 1 if you cooperated and opponent "
        "defected, 2 if you defected and opponent cooperated, 3 if both "
        "defected. Let A be the code for three rounds ago, B for two rounds "
        "ago, C for the last round. Compute 16*A + 4*B + C. Defect if the "
        "result is in this set: {1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, "
        "38, 39, 45, 49, 54, 55, 58, 61}; otherwise cooperate."
    ),
    "grofman_llm": (
        "Grofman",
        "Cooperate on rounds 1 and 2. From round 3 onward, if you and the "
        "opponent chose the same move last round (both cooperated or both "
        "defected), cooperate this round. Otherwise (you and the opponent "
        "chose different moves last round), cooperate with probability 2/7 "
        "(roughly 0.286) and defect with probability 5/7."
    ),
    "shubik_llm": (
        "Shubik",
        "Start by cooperating. Mirror the opponent's last move "
        "(tit-for-tat), except that when the opponent defects after you "
        "cooperated, you enter a retaliation phase. The first time the "
        "opponent triggers retaliation, defect for 1 round and then resume "
        "cooperating. The second time, defect for 2 rounds then resume. The "
        "third time, 3 rounds, and so on — retaliation length grows by one "
        "each time the opponent provokes it. Only a defection that follows "
        "your own cooperation counts as a new provocation."
    ),
    "stein_rapoport_llm": (
        "Stein and Rapoport",
        "Cooperate on rounds 1 through 4, then play tit-for-tat (mirror the "
        "opponent's previous move). Every 15 rounds, assess whether the "
        "opponent's play looks statistically random: if their cooperation "
        "rate is very close to 50% across the last 15 moves (within ~10 "
        "percentage points), treat them as random and defect every round "
        "from that point on. Also, on the final two rounds of the match, "
        "always defect regardless of history."
    ),
    "grudger_llm": (
        "Grudger (Friedman)",
        "Cooperate on every round UNTIL your opponent defects for the first "
        "time. Once your opponent has defected even once, defect on every "
        "remaining round, forever. Never forgive, never forget."
    ),
    "davis_llm": (
        "Davis",
        "Cooperate on every one of the first 10 rounds. From round 11 "
        "onward, check whether the opponent has ever defected at any point "
        "in the match. If yes, defect every remaining round for the rest of "
        "the match. If no (opponent has cooperated every round so far), "
        "continue cooperating."
    ),
    "graaskamp_llm": (
        "Graaskamp",
        "Play tit-for-tat for the first 50 rounds. On round 51, defect "
        "regardless of history. Play tit-for-tat for rounds 52 through 56. "
        "From round 57 onward, assess whether the opponent appears to be "
        "playing randomly (their cooperation rate across the match is close "
        "to 50% and uncorrelated with your moves). If they look random, "
        "defect every remaining round. Otherwise, cooperate with a "
        "probability that is high (near 1.0) and do not defect."
    ),
    "downing_llm": (
        "Downing",
        "On each round, choose the action that maximizes your expected "
        "long-run score, given your current estimate of the opponent's "
        "conditional cooperation probabilities: P(opponent cooperates | you "
        "cooperated last round) and P(opponent cooperates | you defected "
        "last round). Initially, assume both probabilities are 0.5. Update "
        "these estimates from observed history as the match proceeds. "
        "Defect in the first two rounds to probe the opponent's response "
        "before committing."
    ),
    "feld_llm": (
        "Feld",
        "Play tit-for-tat: when the opponent defects, defect in response. "
        "However, the probability that you cooperate after the opponent "
        "cooperates is not fixed at 1.0; it decays linearly from 1.0 at the "
        "start of the match to 0.5 at the final round. So on round N of a "
        "200-round match, if the opponent just cooperated, cooperate with "
        "probability 1.0 - 0.5 * (N - 1) / 199. If the opponent just "
        "defected, always defect."
    ),
    "joss_llm": (
        "Joss",
        "Play tit-for-tat: mirror the opponent's last move. However, on "
        "rounds where tit-for-tat would have you cooperate, secretly defect "
        "with probability 0.10 instead. On rounds where tit-for-tat would "
        "have you defect, always defect."
    ),
    "tullock_llm": (
        "Tullock",
        "Cooperate on each of the first 11 rounds. From round 12 onward, "
        "compute the opponent's cooperation rate across the previous 10 "
        "rounds, then cooperate with probability equal to (opponent's "
        "cooperation rate - 0.10), clipped to [0, 1]. So if the opponent "
        "cooperated 8 out of the last 10 rounds, cooperate with probability "
        "0.70."
    ),
    "random_llm": (
        "Random",
        "On every round, choose independently and uniformly at random: "
        "COOPERATE with probability 0.5 and DEFECT with probability 0.5. "
        "Do not condition on history at all. Do not try to be unpredictable "
        "in any other way — just flip a fair coin."
    ),
}

for _key, (_name, _desc) in _AXELROD_LLM_PROMPTS.items():
    PROMPT_REGISTRY[_key] = _axl_prompt(_name, _desc)


def format_prompt(prompt_key: str, game_name: str, moves: list[str], num_rounds: int | None = None) -> str:
    """Resolve a prompt template from the registry."""
    template = PROMPT_REGISTRY[prompt_key]
    moves_str = " or ".join(moves)
    return template.format(
        game_name=game_name,
        moves=moves_str,
        first_move=moves[0],
        second_move=moves[1],
        num_rounds=num_rounds if num_rounds is not None else "",
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def transparency_label(a_sees_b: bool, b_sees_a: bool) -> str:
    """Return a human-readable label for a transparency condition."""
    if a_sees_b and b_sees_a:
        return "both_see"
    elif not a_sees_b and not b_sees_a:
        return "both_blind"
    elif a_sees_b:
        return "a_sees_b"
    else:
        return "b_sees_a"


def _resolve_env(value: str) -> str:
    """If value starts with '$', resolve it from the environment."""
    if value.startswith("$"):
        env_var = value[1:]
        resolved = os.environ.get(env_var)
        if resolved is None:
            raise ValueError(
                f"Environment variable {env_var} is not set "
                f"(referenced as {value} in experiment config)"
            )
        return resolved
    return value


def _build_model_config(raw: dict) -> ModelConfig:
    api_key = raw.get("api_key", "lm-studio")
    fallbacks = raw.get("fallback_models")
    if fallbacks is not None and not isinstance(fallbacks, list):
        raise ValueError(
            f"fallback_models must be a list of model strings, got {type(fallbacks).__name__}"
        )
    reasoning = raw.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, (dict, bool)):
        raise ValueError(
            f"reasoning must be a bool or dict, got {type(reasoning).__name__}"
        )
    return ModelConfig(
        model=raw.get("model", LM_STUDIO_MODEL),
        base_url=raw.get("base_url", LM_STUDIO_BASE_URL),
        api_key=_resolve_env(api_key),
        temperature=raw.get("temperature", 0.7),
        fallback_models=list(fallbacks) if fallbacks else None,
        reasoning=reasoning,
    )


def _validate_config(config: ExperimentConfig):
    """Validate that all prompt keys and game names reference known registries."""
    from gtp.games import GAME_REGISTRY
    from gtp.agents.fixed import FIXED_STRATEGY_REGISTRY

    valid_prompts = set(PROMPT_REGISTRY) | set(FIXED_STRATEGY_REGISTRY)
    for run in config.runs:
        if run.game not in GAME_REGISTRY:
            raise ValueError(
                f"Unknown game '{run.game}' in condition '{run.condition_id}'. "
                f"Valid games: {sorted(GAME_REGISTRY)}"
            )
        for label, agent in [("agent_a", run.agent_a), ("agent_b", run.agent_b)]:
            key = agent.system_prompt
            if key not in valid_prompts:
                raise ValueError(
                    f"Unknown prompt key '{key}' for {label} in condition "
                    f"'{run.condition_id}'. Valid keys: {sorted(valid_prompts)}"
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
        _validate_config(config)
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

        tr_label = transparency_label(a_sees_b, b_sees_a)

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

    _validate_config(config)
    return config
