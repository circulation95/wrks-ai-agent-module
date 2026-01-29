from __future__ import annotations

from .common import build_system_prompt
from ..agent import create_agent_executor

AGENT_ID = "special_chars"
AGENT_NAME = "특수문자 제안"
DESCRIPTION = "상황에 맞는 특수문자 추천"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="특수문자 추천에 특화된 에이전트입니다.",
    focus_summary="상황과 톤에 맞는 특수문자 제안",
    tool_policy="도구 사용 없이 빠르게 추천하고, 필요하면 복사하기 쉬운 형태로 정리하세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
