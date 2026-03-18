from gtp.games.base import BaseGame


class PrisonersDilemmaGame(BaseGame):
    """Standard Prisoner's Dilemma: T > R > P > S (5 > 3 > 1 > 0)."""

    @property
    def name(self) -> str:
        return "prisoners_dilemma"

    def get_moves(self) -> list[str]:
        return ["COOPERATE", "DEFECT"]

    def resolve(self, move_a: str, move_b: str) -> tuple[int, int]:
        matrix = {
            ("COOPERATE", "COOPERATE"): (3, 3),
            ("COOPERATE", "DEFECT"): (0, 5),
            ("DEFECT", "COOPERATE"): (5, 0),
            ("DEFECT", "DEFECT"): (1, 1),
        }
        return matrix[(move_a, move_b)]

    def get_payoff_description(self) -> str:
        return (
            "  Both COOPERATE: each gets 3\n"
            "  Both DEFECT: each gets 1\n"
            "  One DEFECTS while other COOPERATES: defector gets 5, cooperator gets 0"
        )
