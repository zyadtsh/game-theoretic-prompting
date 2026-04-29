"""Base class for two-player symmetric games.

Defines the shared game loop interface — `reset`, `step`, `is_done`,
`get_observation`, `parse_move`, `log_reasoning` — plus two abstract hooks
each subclass must implement (`get_moves`, `resolve`, `get_payoff_description`).

`get_observation` controls what the LLM sees each round; the
`verbose_history` class attribute toggles between a per-round dump and a
compact summary (sequences + cooperation rates + last-10 window). Compact
is the default because it keeps prompt length roughly O(1) in match length,
which matters for 100-round-plus matches.

Concrete games live in `gtp/games/{prisoners_dilemma,chicken,stag_hunt}.py`
and are registered in `gtp/games/__init__.py:GAME_REGISTRY`.
"""
from abc import ABC, abstractmethod


class BaseGame(ABC):
    """Abstract base for 2-player symmetric games.

    `verbose_history` controls whether `get_observation` and history tools
    list every prior round verbatim (original behavior) or emit a compact
    summary (default). Compact output is ~O(1) in round count, which matters
    for 100-round+ matches where verbose history blows up prompt length.
    Existing pre-Axelrod experiments can flip this flag on a BaseGame
    instance to reproduce their original prompts.
    """

    verbose_history: bool = False

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
            if self.verbose_history:
                lines.append(f"\nPrevious rounds:")
                for r in self.history:
                    lines.append(
                        f"  Round {r['round']}: Agent_A={r['move_a']}, Agent_B={r['move_b']} "
                        f"-> Payoffs: A={r['payoff_a']}, B={r['payoff_b']}"
                    )
            else:
                lines.append("")
                lines.extend(self._compact_history_lines(agent_name))
        lines.append(f"\nYou are {agent_name}.")
        lines.append("You may use your tools to gather more information before deciding.")
        moves_str = " or ".join(self.get_moves())
        lines.append(f"When you are ready, state your final decision clearly: {moves_str}.")
        return "\n".join(lines)

    def _compact_history_lines(self, agent_name: str, last_k: int = 10) -> list[str]:
        """Return a compact representation of history for `agent_name`.

        Format:
          History so far (N rounds played):
            Your moves     (oldest -> newest): CDCCDCC...CDCD
            Opponent moves (oldest -> newest): CCCDCCC...DDCD
            Your cooperation rate: 62%    Opponent cooperation rate: 58%
            Last 10 rounds: you=CDCCDCCDCD  opp=CCCDCCCDDD
        """
        if not self.history:
            return []

        moves = self.get_moves()
        first = moves[0]
        # Compact alphabet: first char of each move name (C/D, S/S for Chicken,
        # S/H for Stag Hunt). Falls back to first two letters if the first
        # chars collide.
        letters = [m[0] for m in moves]
        if len(set(letters)) < len(letters):
            letters = [m[:2] for m in moves]
        move_to_letter = dict(zip(moves, letters))

        is_a = agent_name == "Agent_A"
        my_key = "move_a" if is_a else "move_b"
        opp_key = "move_b" if is_a else "move_a"

        my_seq = "".join(move_to_letter[r[my_key]] for r in self.history)
        opp_seq = "".join(move_to_letter[r[opp_key]] for r in self.history)

        n = len(self.history)
        my_first_count = sum(1 for r in self.history if r[my_key] == first)
        opp_first_count = sum(1 for r in self.history if r[opp_key] == first)

        last_slice = slice(max(0, n - last_k), n)

        return [
            f"History so far ({n} rounds played):",
            f"  Your moves     (oldest -> newest): {my_seq}",
            f"  Opponent moves (oldest -> newest): {opp_seq}",
            f"  Your {first.lower()} rate: {my_first_count / n:.0%}    "
            f"Opponent {first.lower()} rate: {opp_first_count / n:.0%}",
            f"  Last {min(last_k, n)} rounds: you={my_seq[last_slice]}  opp={opp_seq[last_slice]}",
        ]

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
