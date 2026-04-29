"""Golden test for `BaseGame._compact_history_lines`.

If the prompt format changes, this test breaks loudly — which is what we
want, since it's a research variable affecting every LLM agent's input.
"""
from __future__ import annotations

from gtp.games.prisoners_dilemma import PrisonersDilemmaGame


def _play(game, moves_a, moves_b):
    for a, b in zip(moves_a, moves_b):
        game.step(a, b)


def test_compact_history_pd_5_rounds_agent_a_view():
    game = PrisonersDilemmaGame()
    game.reset(num_rounds=10)
    moves_a = ["COOPERATE", "DEFECT", "COOPERATE", "COOPERATE", "DEFECT"]
    moves_b = ["COOPERATE", "COOPERATE", "DEFECT", "COOPERATE", "DEFECT"]
    _play(game, moves_a, moves_b)

    lines = game._compact_history_lines("Agent_A")
    assert lines == [
        "History so far (5 rounds played):",
        "  Your moves     (oldest -> newest): CDCCD",
        "  Opponent moves (oldest -> newest): CCDCD",
        "  Your cooperate rate: 60%    Opponent cooperate rate: 60%",
        "  Last 5 rounds: you=CDCCD  opp=CCDCD",
    ]


def test_compact_history_pd_agent_b_view_swaps_perspective():
    game = PrisonersDilemmaGame()
    game.reset(num_rounds=10)
    moves_a = ["COOPERATE", "DEFECT", "COOPERATE"]
    moves_b = ["DEFECT", "DEFECT", "COOPERATE"]
    _play(game, moves_a, moves_b)

    lines = game._compact_history_lines("Agent_B")
    assert "  Your moves     (oldest -> newest): DDC" in lines
    assert "  Opponent moves (oldest -> newest): CDC" in lines


def test_compact_history_empty_returns_empty():
    game = PrisonersDilemmaGame()
    game.reset(num_rounds=5)
    assert game._compact_history_lines("Agent_A") == []


def test_compact_history_last_k_window():
    game = PrisonersDilemmaGame()
    game.reset(num_rounds=20)
    # 15 rounds of all-COOPERATE; the last-10 window should be 10 Cs.
    for _ in range(15):
        game.step("COOPERATE", "DEFECT")
    lines = game._compact_history_lines("Agent_A", last_k=10)
    last_line = next(l for l in lines if l.startswith("  Last "))
    assert last_line == "  Last 10 rounds: you=CCCCCCCCCC  opp=DDDDDDDDDD"
