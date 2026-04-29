"""Smoke test for `replay_llm_strategy`.

We stub out `build_agent_subgraph` so no real LLM call is made; the stub
just returns a fixed-content `AIMessage`. The test verifies that the
replay loop:
  - calls the agent once per recorded round,
  - parses the move correctly,
  - advances the replay game state with the correct payoffs (sourced
    from `PrisonersDilemmaGame().resolve()`, not a hardcoded matrix).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from gtp.analysis import replay_llm_strategy


class _StubAgentGraph:
    """Mimics the .invoke(...) -> {'messages': [...]} interface of a
    compiled LangChain ReAct agent. Always yields the same final move."""

    def __init__(self, move: str):
        self._move = move
        self.invocations = 0

    def invoke(self, payload, config=None):
        self.invocations += 1
        return {"messages": [AIMessage(content=self._move)]}


def test_replay_loops_once_per_round_and_advances_state(monkeypatch):
    stub = _StubAgentGraph("COOPERATE")

    # Patch where replay_llm_strategy actually imports the symbol from.
    import gtp.agents.builder as builder_mod
    monkeypatch.setattr(
        builder_mod, "build_agent_subgraph", lambda cfg, tools: stub
    )

    recorded_run = {
        "history": [
            {"round": i + 1, "move_a": "DEFECT", "move_b": "DEFECT",
             "payoff_a": 1, "payoff_b": 1}
            for i in range(4)
        ],
    }
    model_cfg = {
        "model": "fake",
        "base_url": "http://localhost",
        "api_key": "fake",
        "temperature": 0.0,
    }

    moves = replay_llm_strategy("tft_llm", recorded_run, model_cfg)

    # One LLM call per recorded round.
    assert stub.invocations == 4
    # Stub always says COOPERATE, so the replay should produce 4 of them.
    assert moves == ["COOPERATE"] * 4


def test_replay_payoffs_match_pd_resolve(monkeypatch):
    """The replay's internal score bookkeeping must match
    PrisonersDilemmaGame.resolve(); if the hardcoded matrix were ever
    reintroduced the assertions below would catch a drift."""
    stub = _StubAgentGraph("DEFECT")

    import gtp.agents.builder as builder_mod
    monkeypatch.setattr(
        builder_mod, "build_agent_subgraph", lambda cfg, tools: stub
    )

    # Opponent (recorded as Agent_B = minimal in the experiment) cooperates
    # every round, so DEFECT vs COOPERATE yields (5, 0) per round.
    recorded_run = {
        "history": [
            {"round": i + 1, "move_a": "COOPERATE", "move_b": "COOPERATE",
             "payoff_a": 3, "payoff_b": 3}
            for i in range(3)
        ],
    }
    moves = replay_llm_strategy(
        "tft_llm", recorded_run,
        {"model": "fake", "base_url": "http://localhost",
         "api_key": "fake", "temperature": 0.0},
    )
    assert moves == ["DEFECT", "DEFECT", "DEFECT"]
