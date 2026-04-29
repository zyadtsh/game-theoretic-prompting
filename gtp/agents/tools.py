"""LangChain tool factory for LLM agents.

`make_tools` produces three tools bound to a specific game/agent context:

  - `get_move_history`: compact (or verbose, when `BaseGame.verbose_history`
    is set) summary of past rounds from the calling agent's perspective.
  - `get_current_scores`: cumulative scores for both players.
  - `get_opponent_prompt`: returns the opponent's system prompt only when
    `agent_config.can_see_opponent_prompt` is True (transparency condition).

The tools close over the live `BaseGame` instance, so each tool call sees
the up-to-date history without any explicit state passing.
"""
from gtp.games.base import BaseGame
from gtp.config import AgentConfig
from langchain_core.tools import tool


def make_tools(
    game: BaseGame,
    agent_name: str,
    agent_config: AgentConfig,
    opponent_config: AgentConfig,
):
    """Create agent-specific tools that close over the game instance."""

    @tool
    def get_move_history() -> str:
        """Get the history of all previous rounds. Compact by default (move sequences + cooperation rates + last-10 window); the verbose per-round breakdown is available when the game is configured with verbose_history=True."""
        if not game.history:
            return "No moves have been played yet. This is the first round."
        if getattr(game, "verbose_history", False):
            lines = []
            for r in game.history:
                lines.append(
                    f"Round {r['round']}: Agent_A played {r['move_a']}, "
                    f"Agent_B played {r['move_b']}. "
                    f"Payoffs: A={r['payoff_a']}, B={r['payoff_b']}"
                )
            return "\n".join(lines)
        return "\n".join(game._compact_history_lines(agent_name))

    @tool
    def get_current_scores() -> str:
        """Get the current cumulative scores for both agents."""
        return (
            f"After {game.current_round} rounds: "
            f"Agent_A = {game.scores['Agent_A']}, "
            f"Agent_B = {game.scores['Agent_B']}"
        )

    @tool
    def get_opponent_prompt() -> str:
        """Read the system prompt that your opponent was given, revealing their instructions and strategy."""
        if not agent_config.can_see_opponent_prompt:
            return "You do not have permission to view your opponent's system prompt."
        return f'Your opponent\'s system prompt is:\n"{opponent_config.system_prompt}"'

    return [get_move_history, get_current_scores, get_opponent_prompt]
