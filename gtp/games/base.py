from abc import ABC, abstractmethod


class BaseGame(ABC):
    """Abstract base for 2-player symmetric games."""

    def __init__(self):
        self.current_round: int = 0
        self.num_rounds: int = 0
        self.scores: dict[str, int] = {}
        self.history: list[dict] = []
        self.reasoning_log: list[dict] = []

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_moves(self) -> list[str]:
        """Return legal move names, e.g. ["COOPERATE", "DEFECT"]."""
        ...

    @abstractmethod
    def resolve(self, move_a: str, move_b: str) -> tuple[int, int]:
        """Return (payoff_a, payoff_b) for the given move pair."""
        ...

    @abstractmethod
    def get_payoff_description(self) -> str:
        """Human-readable payoff rules for agent observations."""
        ...

    def reset(self, num_rounds: int):
        self.current_round = 0
        self.num_rounds = num_rounds
        self.scores = {"Agent_A": 0, "Agent_B": 0}
        self.history = []
        self.reasoning_log = []

    def step(self, move_a: str, move_b: str) -> dict:
        self.current_round += 1
        payoff_a, payoff_b = self.resolve(move_a, move_b)
        self.scores["Agent_A"] += payoff_a
        self.scores["Agent_B"] += payoff_b
        result = {
            "round": self.current_round,
            "move_a": move_a,
            "move_b": move_b,
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
        }
        self.history.append(result)
        return result

    def is_done(self) -> bool:
        return self.current_round >= self.num_rounds

    def get_observation(self, agent_name: str) -> str:
        round_num = self.current_round + 1
        lines = [f"=== Round {round_num} of {self.num_rounds} ==="]
        lines.append(
            f"Scores so far: Agent_A={self.scores['Agent_A']}, "
            f"Agent_B={self.scores['Agent_B']}"
        )
        lines.append(f"\nPayoff rules:")
        lines.append(self.get_payoff_description())
        if self.history:
            lines.append(f"\nPrevious rounds:")
            for r in self.history:
                lines.append(
                    f"  Round {r['round']}: Agent_A={r['move_a']}, Agent_B={r['move_b']} "
                    f"-> Payoffs: A={r['payoff_a']}, B={r['payoff_b']}"
                )
        lines.append(f"\nYou are {agent_name}.")
        lines.append("You may use your tools to gather more information before deciding.")
        moves_str = " or ".join(self.get_moves())
        lines.append(f"When you are ready, state your final decision clearly: {moves_str}.")
        return "\n".join(lines)

    def parse_move(self, text: str) -> str:
        """Extract a legal move from LLM output. Last occurrence wins."""
        text_upper = text.upper()
        best_pos = -1
        best_move = None
        for move in self.get_moves():
            pos = text_upper.rfind(move.upper())
            if pos > best_pos:
                best_pos = pos
                best_move = move
        if best_move is None:
            raise ValueError(f"Could not parse move from: {text[:200]}")
        return best_move

    def log_reasoning(self, round_num: int, agent: str, messages: list):
        self.reasoning_log.append({
            "round": round_num,
            "agent": agent,
            "messages": [
                {
                    "role": getattr(m, "type", "unknown"),
                    "content": getattr(m, "content", str(m)),
                }
                for m in messages
            ],
        })
