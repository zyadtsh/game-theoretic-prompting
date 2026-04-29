"""Round-trip tests for the axelrod-library-backed fixed strategies.

Each wrapper in `gtp.agents.fixed` should produce the same move sequence
as the underlying `axelrod` `Player` class given identical opponent moves.
This catches regressions in the history-syncing logic in
`_make_axelrod_wrapper.factory.node`.
"""
from __future__ import annotations

import axelrod as axl
import pytest

from gtp.agents.fixed import make_fixed_node
from gtp.games.prisoners_dilemma import PrisonersDilemmaGame


def _scripted_match(strategy_key: str, opponent_moves: list[str], seed: int | None = None) -> list[str]:
    """Run our wrapper through a scripted match where the opponent plays
    `opponent_moves` deterministically. Returns the wrapper's move sequence."""
    game = PrisonersDilemmaGame()
    game.reset(num_rounds=len(opponent_moves))
    node = make_fixed_node(strategy_key, "Agent_A", game, seed=seed)

    moves: list[str] = []
    for opp_move in opponent_moves:
        out = node({"phase": "agent_a_turn"})
        my_move = out["move_a"]
        moves.append(my_move)
        # Advance game.history as if the resolver had run.
        game.step(my_move, opp_move)
    return moves


def _axelrod_reference(player_cls_name: str, opponent_moves: list[str], seed: int | None = None) -> list[str]:
    """Same scripted match driven by axelrod directly, for the ground truth."""
    cls = getattr(axl, player_cls_name)
    player = cls()
    opponent = axl.Cooperator()
    player.set_match_attributes(length=len(opponent_moves), game=axl.Game())
    opponent.set_match_attributes(length=len(opponent_moves), game=axl.Game())
    if seed is not None:
        player.set_seed(seed)

    moves: list[str] = []
    for opp_str in opponent_moves:
        action = player.strategy(opponent)
        moves.append("COOPERATE" if action == axl.Action.C else "DEFECT")
        opp_action = axl.Action.C if opp_str == "COOPERATE" else axl.Action.D
        player.update_history(action, opp_action)
        opponent.update_history(opp_action, action)
    return moves


def test_tft_fixed_matches_axelrod_titfortat():
    opp = ["COOPERATE", "DEFECT", "COOPERATE", "DEFECT", "COOPERATE"]
    ours = _scripted_match("tft_fixed", opp)
    theirs = _axelrod_reference("TitForTat", opp)
    assert ours == theirs
    # And the sanity check: TFT should literally mirror with one round of lag.
    assert ours == ["COOPERATE", "COOPERATE", "DEFECT", "COOPERATE", "DEFECT"]


def test_grudger_fixed_matches_axelrod_grudger():
    # Grudger cooperates until the first defection then defects forever.
    opp = ["COOPERATE", "COOPERATE", "DEFECT", "COOPERATE", "COOPERATE"]
    ours = _scripted_match("grudger_fixed", opp)
    theirs = _axelrod_reference("Grudger", opp)
    assert ours == theirs
    assert ours == ["COOPERATE", "COOPERATE", "COOPERATE", "DEFECT", "DEFECT"]


def test_joss_fixed_matches_axelrod_joss_with_seed():
    # Joss is stochastic; with a fixed seed our wrapper and axelrod should
    # produce the identical sequence given identical opponent histories.
    opp = ["COOPERATE"] * 30
    seed = 12345
    ours = _scripted_match("joss_fixed", opp, seed=seed)
    theirs = _axelrod_reference("FirstByJoss", opp, seed=seed)
    assert ours == theirs


def test_random_fixed_is_deterministic_with_seed():
    opp = ["COOPERATE"] * 50
    moves_a = _scripted_match("random_fixed", opp, seed=42)
    moves_b = _scripted_match("random_fixed", opp, seed=42)
    assert moves_a == moves_b
    # Different seed should (almost certainly over 50 rounds) produce a
    # different sequence.
    moves_c = _scripted_match("random_fixed", opp, seed=99)
    assert moves_a != moves_c


def test_legacy_tit_for_tat_fixed_still_works():
    opp = ["DEFECT", "COOPERATE", "DEFECT"]
    ours = _scripted_match("tit_for_tat_fixed", opp)
    # First move cooperates (legacy hand-rolled), then mirrors.
    assert ours == ["COOPERATE", "DEFECT", "COOPERATE"]
