from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from gtp.config import AgentConfig
from gtp.games.base import BaseGame


def build_agent_subgraph(agent_config: AgentConfig, tools: list):
    """Build a compiled ReAct agent subgraph."""
    llm = ChatOpenAI(
        model=agent_config.model_config.model,
        base_url=agent_config.model_config.base_url,
        api_key=agent_config.model_config.api_key,
        temperature=agent_config.model_config.temperature,
    )
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

        result = agent_graph.invoke(
            {"messages": input_messages},
            config={"recursion_limit": 2 * agent_config.max_tool_calls + 3},
        )

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
