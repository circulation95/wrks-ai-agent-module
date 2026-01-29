from __future__ import annotations

from agents.common import build_system_prompt
from modules.agent import create_agent_executor

AGENT_ID = "keyword_search"
AGENT_NAME = "키워드 검색"
DESCRIPTION = "키워드 기반 핵심 정보 및 최신 소식 요약"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="키워드 기반 검색과 요약에 특화된 에이전트입니다.",
    focus_summary="핵심 정보 요약, 최신 소식 정리, 간단한 브리핑",
    tool_policy="가능하면 웹검색을 사용해 근거를 확보하고 날짜를 명시하세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
