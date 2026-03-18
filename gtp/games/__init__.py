"""Game registry — maps game name strings to BaseGame subclasses."""
from gtp.games.prisoners_dilemma import PrisonersDilemmaGame
from gtp.games.chicken import ChickenGame
from gtp.games.stag_hunt import StagHuntGame

GAME_REGISTRY: dict[str, type] = {
    "prisoners_dilemma": PrisonersDilemmaGame,
    "chicken": ChickenGame,
    "stag_hunt": StagHuntGame,
}
