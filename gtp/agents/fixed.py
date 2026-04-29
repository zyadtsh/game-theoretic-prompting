"""Fixed (non-LLM) deterministic strategy agents.

Two layers live here:

1. Legacy hand-rolled strategies (always_cooperate_fixed, always_defect_fixed,
   tit_for_tat_fixed, suspicious_tit_for_tat_fixed) kept for backward
   compatibility with existing experiments.

2. Wrappers around the `axelrod` library's canonical implementations of the
   first-tournament (Axelrod 1980) entries. These back the Axelrod-vs-minimal
   experiment described in README.

Each strategy is a callable: (agent_name, game, seed=None) -> node_function
where node_function has signature (state: dict) -> dict, matching the
interface expected by the LangGraph game state machine.
"""
from __future__ import annotations

from typing import Callable, Optional

from gtp.games.base import BaseGame


# ---------------------------------------------------------------------------
# Legacy hand-rolled strategies
# ---------------------------------------------------------------------------

def _opponent_move_key(agent_name: str) -> str:
    return "move_b" if agent_name == "Agent_A" else "move_a"


def _state_key(agent_name: str) -> str:
    return "move_a" if agent_name == "Agent_A" else "move_b"


def _next_phase(agent_name: str) -> str:
    return "agent_b_turn" if agent_name == "Agent_A" else "resolve"


def make_always_first(agent_name: str, game: BaseGame, seed: Optional[int] = None):
    """Always play the first move (e.g. COOPERATE, SWERVE, STAG)."""
    move = game.get_moves()[0]
    key = _state_key(agent_name)
    phase = _next_phase(agent_name)

    def node(state: dict) -> dict:
        game.log_reasoning(
            round_num=game.current_round + 1,
            agent=agent_name,
            messages=[],
        )
        return {key: move, "phase": phase}

    return node


def make_always_second(agent_name: str, game: BaseGame, seed: Optional[int] = None):
    """Always play the second move (e.g. DEFECT, STRAIGHT, HARE)."""
    move = game.get_moves()[1]
    key = _state_key(agent_name)
    phase = _next_phase(agent_name)

    def node(state: dict) -> dict:
        game.log_reasoning(
            round_num=game.current_round + 1,
            agent=agent_name,
            messages=[],
        )
        return {key: move, "phase": phase}

    return node


def make_tit_for_tat(agent_name: str, game: BaseGame, seed: Optional[int] = None):
    """Cooperate first, then mirror opponent's last move."""
    first_move = game.get_moves()[0]
    key = _state_key(agent_name)
    phase = _next_phase(agent_name)
    opp_key = _opponent_move_key(agent_name)

    def node(state: dict) -> dict:
        if not game.history:
            move = first_move
        else:
            move = game.history[-1][opp_key]
        game.log_reasoning(
            round_num=game.current_round + 1,
            agent=agent_name,
            messages=[],
        )
        return {key: move, "phase": phase}

    return node


def make_suspicious_tit_for_tat(agent_name: str, game: BaseGame, seed: Optional[int] = None):
    """Defect first, then mirror opponent's last move."""
    second_move = game.get_moves()[1]
    key = _state_key(agent_name)
    phase = _next_phase(agent_name)
    opp_key = _opponent_move_key(agent_name)

    def node(state: dict) -> dict:
        if not game.history:
            move = second_move
        else:
            move = game.history[-1][opp_key]
        game.log_reasoning(
            round_num=game.current_round + 1,
            agent=agent_name,
            messages=[],
        )
        return {key: move, "phase": phase}

    return node


# ---------------------------------------------------------------------------
# Axelrod-library-backed strategies (first-tournament entries)
# ---------------------------------------------------------------------------
#
# Each wrapper creates a persistent axelrod.Player at graph-build time, plus
# a lightweight opponent stub whose .history mirrors the real opponent's
# moves. Before each decision we sync both histories from game.history so the
# axelrod strategy sees the correct state. This relies on game.history being
# append-only within a run (true — BaseGame.step() appends and never rewrites).
#
# Only the 7 stochastic wrappers consume the seed argument; the 7
# deterministic ones ignore it by design.


# Maps our move strings to axelrod Action enum and back. Lazily imported so
# users who never touch axelrod-backed strategies aren't forced to install the
# dependency at runtime (it's declared in requirements.txt but this keeps the
# legacy strategies independent of the import).

def _axl_action(move: str):
    import axelrod as axl
    return axl.Action.C if move == "COOPERATE" else axl.Action.D


def _from_axl(action) -> str:
    import axelrod as axl
    return "COOPERATE" if action == axl.Action.C else "DEFECT"


def _make_axelrod_wrapper(axl_cls_name: str, *, stochastic: bool):
    """Return a factory (agent_name, game, seed=None) -> node.

    axl_cls_name is looked up lazily on the axelrod module so this file is
    importable without the dependency.
    """

    def factory(agent_name: str, game: BaseGame, seed: Optional[int] = None):
        import axelrod as axl

        axl_cls = getattr(axl, axl_cls_name)
        player = axl_cls()
        # An axelrod.Cooperator instance is used purely as a history container
        # so the real player can call opponent.history / opponent.cooperations
        # / opponent.defections as any FirstBy* strategy expects.
        opponent = axl.Cooperator()

        # Inform strategies of the match length and payoff matrix so
        # length-aware ones (Stein-Rapoport, Graaskamp, Feld, Downing,
        # Nydegger) behave as designed.
        player.set_match_attributes(length=game.num_rounds, game=axl.Game())
        opponent.set_match_attributes(length=game.num_rounds, game=axl.Game())

        if stochastic and seed is not None:
            player.set_seed(seed)

        key = _state_key(agent_name)
        phase = _next_phase(agent_name)
        opp_key = _opponent_move_key(agent_name)
        my_key = "move_a" if agent_name == "Agent_A" else "move_b"

        def node(state: dict) -> dict:
            # Sync axelrod history from game.history. synced == rounds already
            # reflected in player.history; sync only the new ones.
            synced = len(player.history)
            for r in game.history[synced:]:
                my_move = _axl_action(r[my_key])
                opp_move = _axl_action(r[opp_key])
                player.update_history(my_move, opp_move)
                opponent.update_history(opp_move, my_move)

            action = player.strategy(opponent)
            move = _from_axl(action)

            game.log_reasoning(
                round_num=game.current_round + 1,
                agent=agent_name,
                messages=[],
            )
            return {key: move, "phase": phase}

        return node

    return factory


# 7 deterministic first-tournament strategies at T=0.
# "Deterministic" here means: given the same opponent move sequence, always
# returns the same moves. Relevant for the experiment because the opponent
# (minimal LLM at temperature 0) is itself expected to be near-deterministic.
_DETERMINISTIC_STRATEGIES: dict[str, str] = {
    "tft_fixed": "TitForTat",
    "tideman_chieruzzi_fixed": "FirstByTidemanAndChieruzzi",
    "nydegger_fixed": "FirstByNydegger",
    "shubik_fixed": "FirstByShubik",
    "grudger_fixed": "Grudger",
    "davis_fixed": "FirstByDavis",
    "downing_fixed": "FirstByDowning",
}

# 7 stochastic first-tournament strategies. Each uses its own RNG seed,
# derived per-run from (run_id, agent_name), so `repetitions: N` actually
# produces distinct trajectories.
_STOCHASTIC_STRATEGIES: dict[str, str] = {
    "random_fixed": "Random",
    "grofman_fixed": "FirstByGrofman",
    "joss_fixed": "FirstByJoss",
    "feld_fixed": "FirstByFeld",
    "tullock_fixed": "FirstByTullock",
    "graaskamp_fixed": "FirstByGraaskamp",
    "stein_rapoport_fixed": "FirstBySteinAndRapoport",
}


# Registry: maps YAML prompt key -> factory function.
# Each factory has signature (agent_name: str, game: BaseGame, seed: int | None) -> node_function
FIXED_STRATEGY_REGISTRY: dict[str, Callable] = {
    # Legacy hand-rolled
    "always_cooperate_fixed": make_always_first,
    "always_defect_fixed": make_always_second,
    "tit_for_tat_fixed": make_tit_for_tat,
    "suspicious_tit_for_tat_fixed": make_suspicious_tit_for_tat,
}

for _key, _cls in _DETERMINISTIC_STRATEGIES.items():
    FIXED_STRATEGY_REGISTRY[_key] = _make_axelrod_wrapper(_cls, stochastic=False)

for _key, _cls in _STOCHASTIC_STRATEGIES.items():
    FIXED_STRATEGY_REGISTRY[_key] = _make_axelrod_wrapper(_cls, stochastic=True)


# Strategy classification exposed for analysis + YAML splits.
# The legacy hand-rolled entries are all deterministic; include them for
# completeness so callers don't special-case.
DETERMINISTIC_FIXED_KEYS: frozenset[str] = frozenset(
    {
        "always_cooperate_fixed",
        "always_defect_fixed",
        "tit_for_tat_fixed",
        "suspicious_tit_for_tat_fixed",
    }
    | set(_DETERMINISTIC_STRATEGIES)
)

STOCHASTIC_FIXED_KEYS: frozenset[str] = frozenset(_STOCHASTIC_STRATEGIES)


def is_fixed_strategy(prompt_key: str) -> bool:
    """Check whether a prompt key refers to a fixed (non-LLM) strategy."""
    return prompt_key in FIXED_STRATEGY_REGISTRY


def is_stochastic_fixed(prompt_key: str) -> bool:
    """Check whether a fixed strategy needs per-run seeding."""
    return prompt_key in STOCHASTIC_FIXED_KEYS


def make_fixed_node(
    strategy_key: str,
    agent_name: str,
    game: BaseGame,
    seed: Optional[int] = None,
):
    """Build a fixed-strategy node function from a registry key."""
    factory = FIXED_STRATEGY_REGISTRY[strategy_key]
    return factory(agent_name, game, seed=seed)
