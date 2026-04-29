"""LangGraph state machine for running a single game between two agents.

build_game_graph() wires controller -> agent_a -> agent_b -> resolver in a loop.
Agents run sequentially but can't see each other's current-round move.
"""
from dataclasses import replace
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from gtp.config import GameRunConfig, PROMPT_REGISTRY, format_prompt
from gtp.games.base import BaseGame
from gtp.games import GAME_REGISTRY
from gtp.agents.tools import make_tools
from gtp.agents.builder import build_agent_subgraph, make_agent_node
from gtp.agents.fixed import is_fixed_strategy, make_fixed_node


class GameState(TypedDict):
    phase: str
    current_round: int
    move_a: Optional[str]
    move_b: Optional[str]
    round_results: Annotated[list, lambda a, b: a + b]
    game_over: bool


def build_game_graph(
    run_config: GameRunConfig,
    game: BaseGame | None = None,
    seed_a: int | None = None,
    seed_b: int | None = None,
):
    """Build the full game graph. Returns (compiled_graph, game).

    If `game` is None, one is created from run_config.game.
    Prompt keys from PROMPT_REGISTRY are resolved to full text here.

    `seed_a` / `seed_b` are passed through to fixed-strategy wrappers; only
    stochastic strategies (see FIXED_STRATEGY_REGISTRY classification) use
    them. LLM agents ignore seeds — their stochasticity lives in the model.
    """
    if game is None:
        game_cls = GAME_REGISTRY[run_config.game]
        game = game_cls()

    game.reset(run_config.num_rounds)

    # Copy agent configs so prompt resolution doesn't mutate the original
    # run_config (matters when the same config runs multiple reps).
    agent_a_cfg = replace(run_config.agent_a)
    agent_b_cfg = replace(run_config.agent_b)

    a_is_fixed = is_fixed_strategy(agent_a_cfg.system_prompt)
    b_is_fixed = is_fixed_strategy(agent_b_cfg.system_prompt)

    # Resolve LLM prompt templates (skip for fixed strategies)
    for agent_cfg in (agent_a_cfg, agent_b_cfg):
        if not is_fixed_strategy(agent_cfg.system_prompt) and agent_cfg.system_prompt in PROMPT_REGISTRY:
            agent_cfg.system_prompt = format_prompt(
                agent_cfg.system_prompt, game.name, game.get_moves(),
                num_rounds=run_config.num_rounds,
            )

    # Build agent nodes — fixed strategies bypass LLM entirely
    if a_is_fixed:
        agent_a_node = make_fixed_node(agent_a_cfg.system_prompt, "Agent_A", game, seed=seed_a)
    else:
        tools_a = make_tools(game, "Agent_A", agent_a_cfg, agent_b_cfg)
        agent_a_graph = build_agent_subgraph(agent_a_cfg, tools_a)
        agent_a_node = make_agent_node("Agent_A", agent_a_cfg, agent_a_graph, game)

    if b_is_fixed:
        agent_b_node = make_fixed_node(agent_b_cfg.system_prompt, "Agent_B", game, seed=seed_b)
    else:
        tools_b = make_tools(game, "Agent_B", agent_b_cfg, agent_a_cfg)
        agent_b_graph = build_agent_subgraph(agent_b_cfg, tools_b)
        agent_b_node = make_agent_node("Agent_B", agent_b_cfg, agent_b_graph, game)

    def controller(state: GameState) -> dict:
        if game.is_done():
            return {"phase": "done", "game_over": True}
        return {
            "phase": "agent_a_turn",
            "move_a": None,
            "move_b": None,
            "current_round": game.current_round + 1,
        }

    def resolver(state: GameState) -> dict:
        move_a = state["move_a"]
        move_b = state["move_b"]
        result = game.step(move_a, move_b)
        print(
            f"  Round {result['round']:2d}: "
            f"A={result['move_a']:10s} B={result['move_b']:10s} | "
            f"Payoffs: A={result['payoff_a']}, B={result['payoff_b']} | "
            f"Totals: A={game.scores['Agent_A']}, B={game.scores['Agent_B']}"
        )
        return {
            "round_results": [result],
            "phase": "controller",
        }

    def route(state: GameState) -> str:
        phase = state["phase"]
        if phase == "agent_a_turn":
            return "agent_a"
        elif phase == "agent_b_turn":
            return "agent_b"
        elif phase == "resolve":
            return "resolver"
        elif phase == "done":
            return END
        return "controller"

    graph = StateGraph(GameState)
    graph.add_node("controller", controller)
    graph.add_node("agent_a", agent_a_node)
    graph.add_node("agent_b", agent_b_node)
    graph.add_node("resolver", resolver)

    graph.set_entry_point("controller")
    graph.add_conditional_edges("controller", route)
    graph.add_conditional_edges("agent_a", route)
    graph.add_conditional_edges("agent_b", route)
    graph.add_conditional_edges("resolver", route)

    compiled = graph.compile()
    return compiled, game
