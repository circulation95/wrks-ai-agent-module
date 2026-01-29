from __future__ import annotations

from typing import Any, Annotated, TypedDict
from uuid import uuid4
import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent


def _guess_language(text: str) -> str:
    lowered = text.lower()
    if "python" in lowered:
        return "python"
    if "javascript" in lowered or "js" in lowered:
        return "javascript"
    if "typescript" in lowered or "ts" in lowered:
        return "typescript"
    if "sql" in lowered:
        return "sql"
    if "bash" in lowered or "shell" in lowered:
        return "bash"
    return "python"


def _classify_intent_llm(
    user_text: str,
    tools: list[Any],
    router_model: ChatOpenAI,
) -> tuple[str | None, dict]:
    available = [getattr(t, "name", "") for t in tools if getattr(t, "name", "")]
    if not available:
        return None, {}

    tool_schemas = {
        "web_search": {"query": "string"},
        "document_search": {"query": "string"},
        "image_generate": {"prompt": "string"},
        "code_generate": {"task": "string", "language": "string"},
    }
    allowed = {k: v for k, v in tool_schemas.items() if k in available}

    system = (
        "You are a conservative routing classifier for tool use. "
        "Only choose a tool when it is clearly necessary. "
        "If the query can be answered without tools, choose null. "
        "Choose a single best tool or none. "
        "Return ONLY valid JSON with keys: tool, args. "
        "If no tool is needed, set tool to null and args to {}."
    )
    user = (
        "User query:\n"
        f"{user_text}\n\n"
        "Available tools and argument schema:\n"
        f"{json.dumps(allowed, ensure_ascii=False)}\n\n"
        "Response JSON example:\n"
        "{\"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}"
    )

    response = router_model.invoke([SystemMessage(system), HumanMessage(user)])
    content = response.content or ""
    try:
        data = json.loads(content)
    except Exception:
        return None, {}

    tool_name = data.get("tool")
    args = data.get("args", {}) if isinstance(data.get("args", {}), dict) else {}
    if tool_name not in available:
        return None, {}
    if tool_name == "code_generate" and "language" not in args:
        args["language"] = _guess_language(user_text)
    return tool_name, args


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str


def _create_routed_agent_executor(
    model_name: str, tools: list[Any], system_prompt: str | None
):
    model = ChatOpenAI(model_name=model_name, max_tokens=1024)
    router_model = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=256, temperature=0)

    def router_node(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"route": "respond"}
        user_text = messages[-1].content if hasattr(messages[-1], "content") else ""
        tool_name, args = _classify_intent_llm(user_text, tools, router_model)
        if not tool_name:
            return {"route": "respond"}
        tool_call = {"id": str(uuid4()), "name": tool_name, "args": args}
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])], "route": "tool"}

    def agent_node(state: AgentState):
        messages = state.get("messages", [])
        if system_prompt:
            input_messages = [SystemMessage(system_prompt)] + messages
        else:
            input_messages = messages
        response = model.invoke(input_messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("tools", tool_node)
    graph.add_node("agent", agent_node)

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "respond"),
        {"tool": "tools", "respond": "agent"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("agent", END)
    graph.set_entry_point("router")

    return graph.compile()


def create_agent_executor(model_name="gpt-4o", tools=None, system_prompt=None, route_tools=True):
    # ??? ??
    memory = MemorySaver()

    # ?? ??
    model = ChatOpenAI(model_name=model_name, max_tokens=1024)

    # ??? ???? ??
    if tools is None:
        tools = []

    if route_tools:
        return _create_routed_agent_executor(model_name, tools, system_prompt)

    agent_executor = create_react_agent(
        model, tools=tools, checkpointer=memory, prompt=system_prompt
    )

    return agent_executor
