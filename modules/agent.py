from __future__ import annotations

from typing import Any, Annotated, TypedDict
from uuid import uuid4
import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
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


def _get_last_human_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""


def _get_last_tool_message(messages: list) -> ToolMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg
    return None


def _get_last_ai_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg.content or ""
    return ""


def _parse_document_results(tool_content: str) -> tuple[list[str], int]:
    try:
        payload = json.loads(tool_content)
    except Exception:
        return [], 0
    results = payload.get("results", []) if isinstance(payload, dict) else []
    texts = []
    for item in results:
        content = item.get("content", "")
        if content:
            texts.append(content)
    return texts, len(results)


def _grade_documents_llm(
    docs: list[str], question: str, grader_model: ChatOpenAI
) -> list[str]:
    if not docs or not question:
        return []

    system = (
        "You are a grader assessing relevance of a retrieved document to a user question. "
        "If the document contains keyword(s) or semantic meaning related to the question, "
        "grade it as relevant. Return ONLY 'yes' or 'no'."
    )
    filtered: list[str] = []
    for doc in docs:
        prompt = [
            SystemMessage(system),
            HumanMessage(
                f"Retrieved document:\n{doc}\n\nUser question:\n{question}"
            ),
        ]
        decision = (grader_model.invoke(prompt).content or "").strip().lower()
        if "yes" in decision:
            filtered.append(doc)
    return filtered


def _classify_intent_llm(
    user_text: str,
    tools: list[Any],
    router_model: ChatOpenAI,
    router_system_prompt: str | None = None,
    force_document_search: bool = False,
) -> tuple[str | None, dict]:
    available = [getattr(t, "name", "") for t in tools if getattr(t, "name", "")]
    if not available:
        return None, {}

    if force_document_search and "document_search" in available:
        return "document_search", {"query": user_text}

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
        "If the only available tool is document_search, you MUST use it. "
        "Choose a single best tool or none. "
        "Return ONLY valid JSON with keys: tool, args. "
        "If no tool is needed, set tool to null and args to {}."
    )
    if router_system_prompt:
        system = f"{system} {router_system_prompt}"

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
    model_name: str,
    tools: list[Any],
    system_prompt: str | None,
    router_system_prompt: str | None = None,
    force_document_search: bool = False,
    doc_min_results: int = 2,
):
    model = ChatOpenAI(model_name=model_name, max_tokens=1024)
    router_model = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=256, temperature=0)
    rewrite_model = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=256, temperature=0)
    check_model = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=256, temperature=0)
    grade_model = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=128, temperature=0)

    doc_tool_available = any(getattr(t, "name", "") == "document_search" for t in tools)

    def router_node(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"route": "respond"}
        user_text = messages[-1].content if hasattr(messages[-1], "content") else ""
        tool_name, args = _classify_intent_llm(
            user_text,
            tools,
            router_model,
            router_system_prompt=router_system_prompt,
            force_document_search=force_document_search,
        )
        if not tool_name:
            return {"route": "respond"}
        tool_call = {"id": str(uuid4()), "name": tool_name, "args": args}
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])], "route": "tool"}

    def doc_relevance_check_node(state: AgentState):
        messages = state.get("messages", [])
        last_tool = _get_last_tool_message(messages)
        if not last_tool:
            return {"route": "respond"}
        if getattr(last_tool, "name", None) != "document_search":
            return {"route": "respond"}

        doc_texts, result_count = _parse_document_results(last_tool.content or "")
        question = _get_last_human_text(messages)
        graded_docs = _grade_documents_llm(doc_texts, question, grade_model)
        if graded_docs and len(graded_docs) >= doc_min_results:
            return {"route": "respond"}
        return {"route": "rewrite"}

    def query_rewrite_node(state: AgentState):
        messages = state.get("messages", [])
        original = _get_last_human_text(messages)
        if not original:
            return {"route": "respond"}

        system = (
            "You are a question re-writer that converts an input question to a better version "
            "optimized for document search. Focus on the core intent and include key entities."
        )
        prompt = [
            SystemMessage(system),
            HumanMessage(f"Rewrite the question for document search:\n{original}"),
        ]
        rewritten = rewrite_model.invoke(prompt).content or original
        tool_call = {
            "id": str(uuid4()),
            "name": "document_search",
            "args": {"query": rewritten},
        }
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])], "route": "tool"}

    def draft_node(state: AgentState):
        messages = state.get("messages", [])
        if system_prompt:
            input_messages = [SystemMessage(system_prompt)] + messages
        else:
            input_messages = messages
        response = model.invoke(input_messages)
        return {"messages": [response]}

    def hallucination_check_node(state: AgentState):
        if not doc_tool_available:
            return {"route": "final"}

        messages = state.get("messages", [])
        last_tool = _get_last_tool_message(messages)
        if not last_tool or getattr(last_tool, "name", None) != "document_search":
            return {"route": "final"}

        docs, _ = _parse_document_results(last_tool.content or "")
        answer = _get_last_ai_answer(messages)
        if not docs or not answer:
            return {"route": "final"}

        system = (
            "You are a grader assessing whether an answer is grounded in the provided facts. "
            "Return ONLY valid JSON like {\"grounded\": \"yes\"} or {\"grounded\": \"no\"}."
        )
        user = (
            "Facts:\n"
            f"{chr(10).join(docs)}\n\n"
            "Answer:\n"
            f"{answer}"
        )
        response = check_model.invoke([SystemMessage(system), HumanMessage(user)])
        content = response.content or ""
        try:
            data = json.loads(content)
        except Exception:
            return {"route": "final"}

        if data.get("grounded") == "yes":
            return {"route": "final"}

        user_text = _get_last_human_text(messages) or answer
        tool_call = {
            "id": str(uuid4()),
            "name": "document_search",
            "args": {"query": user_text},
        }
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])], "route": "retry"}

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("tools", tool_node)
    graph.add_node("doc_check", doc_relevance_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    def final_agent_node(state: AgentState):
        messages = state.get("messages", [])
        draft = _get_last_ai_answer(messages)
        if not draft:
            return {"messages": []}
        return {"messages": [AIMessage(content=draft)]}

    graph.add_node("draft", draft_node)
    graph.add_node("agent", final_agent_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "respond"),
        {"tool": "tools", "respond": "draft"},
    )
    graph.add_edge("tools", "doc_check")
    graph.add_conditional_edges(
        "doc_check",
        lambda state: state.get("route", "respond"),
        {"respond": "draft", "rewrite": "query_rewrite"},
    )
    graph.add_edge("query_rewrite", "tools")
    graph.add_edge("draft", "hallucination_check")
    graph.add_conditional_edges(
        "hallucination_check",
        lambda state: state.get("route", "final"),
        {"final": "agent", "retry": "tools"},
    )
    graph.add_edge("agent", END)
    graph.set_entry_point("router")

    compiled = graph.compile()
    return compiled.with_config({"recursion_limit": 30})


def create_agent_executor(
    model_name="gpt-4o",
    tools=None,
    system_prompt=None,
    route_tools=True,
    router_system_prompt: str | None = None,
    force_document_search: bool = False,
    doc_min_results: int = 2,
):
    memory = MemorySaver()
    model = ChatOpenAI(model_name=model_name, max_tokens=1024)

    if tools is None:
        tools = []

    if route_tools:
        return _create_routed_agent_executor(
            model_name,
            tools,
            system_prompt,
            router_system_prompt=router_system_prompt,
            force_document_search=force_document_search,
            doc_min_results=doc_min_results,
        )

    agent_executor = create_react_agent(
        model, tools=tools, checkpointer=memory, prompt=system_prompt
    )

    return agent_executor
