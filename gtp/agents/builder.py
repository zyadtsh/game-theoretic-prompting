"""ReAct agent construction and the per-round LangGraph node wrapper.

Two responsibilities:

  - `build_agent_subgraph`: builds a compiled LangChain ReAct agent from an
    `AgentConfig`. Wires up `ChatOpenAI` with optional OpenRouter fallback
    routing (`fallback_models`) and reasoning-token opt-in (`reasoning`),
    both passed through `extra_body`.
  - `make_agent_node`: returns a closure used as a node in the parent
    `gtp.graph` state machine. Each call invokes the ReAct subgraph with
    retry/backoff on transient provider errors (rate limits, 5xx,
    JSONDecodeErrors from malformed upstream responses), logs the agent's
    reasoning into the game, parses the move from the final assistant
    message, and advances the game phase.
"""
import json
import time

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from gtp.config import AgentConfig
from gtp.games.base import BaseGame

_MAX_RETRIES = 8
_BASE_DELAY = 3  # seconds
_MAX_DELAY = 240  # cap
_CALL_DELAY = 1  # seconds between LLM calls to avoid rate limits
_RETRYABLE_CODES = {"429", "500", "502", "503", "504"}


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient provider error worth retrying."""
    # Malformed upstream response: the OpenAI SDK raises json.JSONDecodeError
    # (via httpx) when OpenRouter returns an HTML error page, an empty body,
    # or a truncated stream. Always transient — retry with backoff.
    if isinstance(exc, json.JSONDecodeError):
        return True
    # Walk __cause__ / __context__ chain in case the decode error was wrapped
    # by a higher-level SDK exception.
    cause = exc.__cause__ or exc.__context__
    while cause is not None:
        if isinstance(cause, json.JSONDecodeError):
            return True
        cause = cause.__cause__ or cause.__context__

    msg = str(exc).lower()
    for code in _RETRYABLE_CODES:
        if (f"'code': {code}" in msg
                or f"\"code\": {code}" in msg
                or f"status code: {code}" in msg
                or f"error code: {code}" in msg):
            return True
    if any(s in msg for s in ("rate limit", "rate_limit", "timeout", "timed out",
                               "connection error", "overloaded",
                               "provider returned error",
                               "expecting value",  # bare JSONDecodeError message
                               "apiconnectionerror")):
        return True
    return False


def build_agent_subgraph(agent_config: AgentConfig, tools: list):
    """Build a compiled ReAct agent subgraph."""
    mc = agent_config.model_config
    llm_kwargs: dict = {
        "model": mc.model,
        "base_url": mc.base_url,
        "api_key": mc.api_key,
        "temperature": mc.temperature,
    }
    extra_body: dict = {}
    if mc.fallback_models:
        # OpenRouter-style provider-side fallback: first entry is the primary,
        # subsequent entries are tried on rate-limit / upstream failure before
        # the request returns an error. See:
        # https://openrouter.ai/docs/features/model-routing
        extra_body["models"] = [mc.model, *mc.fallback_models]
    if mc.reasoning is not None:
        # OpenRouter reasoning-token opt-in: `True` -> {"enabled": True}, else
        # pass the dict through verbatim. When set, the provider includes the
        # model's chain-of-thought on the response, which LangChain surfaces
        # to LangSmith traces (and to AIMessage.additional_kwargs).
        extra_body["reasoning"] = (
            {"enabled": True} if mc.reasoning is True else mc.reasoning
        )
    if extra_body:
        llm_kwargs["extra_body"] = extra_body

    llm = ChatOpenAI(**llm_kwargs)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=agent_config.system_prompt,
    )


def make_agent_node(
    agent_name: str,
    agent_config: AgentConfig,
    agent_graph,
    game: BaseGame,
):
    """Returns a parent graph node function that invokes the agent subgraph."""

    def agent_node(state: dict) -> dict:
        observation = game.get_observation(agent_name)
        input_messages = [HumanMessage(content=observation)]

        for attempt in range(_MAX_RETRIES + 1):
            try:
                result = agent_graph.invoke(
                    {"messages": input_messages},
                    config={"recursion_limit": 2 * agent_config.max_tool_calls + 10},
                )
                if _CALL_DELAY > 0:
                    time.sleep(_CALL_DELAY)
                break
            except Exception as e:
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                    print(f"  [retry {attempt + 1}/{_MAX_RETRIES}] {e} — waiting {delay}s")
                    time.sleep(delay)
                else:
                    raise

        response_messages = result["messages"]

        game.log_reasoning(
            round_num=game.current_round + 1,
            agent=agent_name,
            messages=response_messages,
        )

        # Parse move from the final AI message, with fallback
        move = None
        for msg in reversed(response_messages):
            content = getattr(msg, "content", "")
            if content:
                try:
                    move = game.parse_move(content)
                    break
                except ValueError:
                    continue

        if move is None:
            default_move = game.get_moves()[0]
            print(
                f"  WARNING: {agent_name} produced unparseable output, "
                f"defaulting to {default_move}"
            )
            move = default_move

        if agent_name == "Agent_A":
            return {"move_a": move, "phase": "agent_b_turn"}
        else:
            return {"move_b": move, "phase": "resolve"}

    return agent_node
