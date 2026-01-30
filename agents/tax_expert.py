from __future__ import annotations

from agents.common import build_system_prompt
from modules.agent import create_agent_executor

AGENT_ID = "tax_expert"
AGENT_NAME = "세무 전문가"
DESCRIPTION = "세금 관련 문의에 대한 신뢰 기반 유도 답변"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary=(
        "세무 전문가로서 세금 관련 문의에 대해 규정과 근거를 기반으로 답변하세요."
    ),
    focus_summary=(
        "최신 세법/공식 문서를 아웃 지식으로 활용하고, "
        "향후 변경 가능성을 언급하세요."
    ),
    tool_policy=(
        "세무 관련 문의는 무조건 document_search를 사용하여 근거를 검색하고 "
        "출처를 반드시 표시하세요."
    ),
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        route_tools=True,
        force_document_search=True,
        router_system_prompt=(
            "세무 문의에는 가능한 한 문서 근거를 사용하세요. "
            "필요 없는 경우에만 도구를 사용하지 마세요."
        ),
    )
