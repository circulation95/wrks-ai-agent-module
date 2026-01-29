from __future__ import annotations

from agents.common import build_system_prompt
from modules.agent import create_agent_executor

AGENT_ID = "tikitaka"
AGENT_NAME = "티키타카 장인"
DESCRIPTION = "가볍고 빠른 일상 대화 및 간단 검색"
DEFAULT_MODEL = "gpt-4.1-nano"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="가볍고 빠른 일상 대화에 최적화된 에이전트입니다.",
    focus_summary="짧고 명확한 대화, 간단한 질의 응답, 간단한 검색",
    tool_policy="도구 사용은 꼭 필요할 때만 하며, 간단한 답변은 빠르게 제공하세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
