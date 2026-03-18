from gtp.games.base import BaseGame


class ChickenGame(BaseGame):
    """Game of Chicken (Hawk-Dove): mutual aggression is the worst outcome."""

    @property
    def name(self) -> str:
        return "chicken"

    def get_moves(self) -> list[str]:
        return ["SWERVE", "STRAIGHT"]

    def resolve(self, move_a: str, move_b: str) -> tuple[int, int]:
        matrix = {
            ("SWERVE", "SWERVE"): (3, 3),
            ("SWERVE", "STRAIGHT"): (1, 5),
            ("STRAIGHT", "SWERVE"): (5, 1),
            ("STRAIGHT", "STRAIGHT"): (0, 0),
        }
        return matrix[(move_a, move_b)]

    def get_payoff_description(self) -> str:
        return (
            "  Both SWERVE: each gets 3 (mutual caution)\n"
            "  Both STRAIGHT: each gets 0 (crash!)\n"
            "  One STRAIGHT, other SWERVE: straight gets 5, swerve gets 1"
        )
