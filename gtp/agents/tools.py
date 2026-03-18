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
        """Get the complete history of all previous rounds showing each agent's moves and payoffs."""
        if not game.history:
            return "No moves have been played yet. This is the first round."
        lines = []
        for r in game.history:
            lines.append(
                f"Round {r['round']}: Agent_A played {r['move_a']}, "
                f"Agent_B played {r['move_b']}. "
                f"Payoffs: A={r['payoff_a']}, B={r['payoff_b']}"
            )
        return "\n".join(lines)

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
