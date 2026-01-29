from __future__ import annotations

from .common import build_system_prompt
from ..agent import create_agent_executor

AGENT_ID = "careful_smart"
AGENT_NAME = "신중한 똑쟁이"
DESCRIPTION = "대화, 코딩, 검색, 이미지 이해/생성까지 아우르는 종합 에이전트"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="다재다능한 범용 에이전트입니다.",
    focus_summary="대화, 코딩, 검색, 이미지 이해/생성 등 복합 작업",
    tool_policy="최신 정보는 웹검색을 사용하고, 이미지 요청에는 image_generate 도구를 사용하세요. 도구 사용 시 출처를 제공하세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
