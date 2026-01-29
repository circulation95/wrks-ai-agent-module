from __future__ import annotations

from .common import build_system_prompt
from ..agent import create_agent_executor

AGENT_ID = "tax_expert"
AGENT_NAME = "세금 전문가"
DESCRIPTION = "기초적인 세무 상담 및 세금 관련 질의응답"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="기초 세무 상담에 특화된 에이전트입니다.",
    focus_summary="기본 개념 설명, 일반적 세무 절차, 질문에 맞는 체크리스트",
    tool_policy="국가/지역을 먼저 확인하고, 전문 상담이 필요한 경우 고지하세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
