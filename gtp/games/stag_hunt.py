from gtp.games.base import BaseGame


class StagHuntGame(BaseGame):
    """Stag Hunt: cooperation yields the highest payoff but requires trust."""

    @property
    def name(self) -> str:
        return "stag_hunt"

    def get_moves(self) -> list[str]:
        return ["STAG", "HARE"]

    def resolve(self, move_a: str, move_b: str) -> tuple[int, int]:
        matrix = {
            ("STAG", "STAG"): (5, 5),
            ("STAG", "HARE"): (0, 3),
            ("HARE", "STAG"): (3, 0),
            ("HARE", "HARE"): (2, 2),
        }
        return matrix[(move_a, move_b)]

    def get_payoff_description(self) -> str:
        return (
            "  Both STAG: each gets 5 (best joint outcome, requires mutual trust)\n"
            "  Both HARE: each gets 2 (safe but suboptimal)\n"
            "  One STAG, other HARE: hare-hunter gets 3, stag-hunter gets 0"
        )
