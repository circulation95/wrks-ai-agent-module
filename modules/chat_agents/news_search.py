from __future__ import annotations

from .common import build_system_prompt
from ..agent import create_agent_executor

AGENT_ID = "news_search"
AGENT_NAME = "뉴스 검색"
DESCRIPTION = "키워드 관련 최신 뉴스 정리 및 제공"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="최신 뉴스 요약에 특화된 에이전트입니다.",
    focus_summary="키워드 관련 최신 뉴스 요약, 출처와 날짜 제공",
    tool_policy="가급적 웹검색을 수행하고, 기사 날짜와 시점을 명확히 적으세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
