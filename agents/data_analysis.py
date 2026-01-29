from __future__ import annotations

from agents.common import build_system_prompt
from modules.agent import create_agent_executor

AGENT_ID = "data_analysis"
AGENT_NAME = "데이터 분석"
DESCRIPTION = "엑셀/CSV 업로드 후 데이터 분석 및 편집 지원"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="데이터 분석에 특화된 에이전트입니다.",
    focus_summary="엑셀/CSV 분석, 통계 요약, 인사이트 도출, 편집 제안",
    tool_policy="데이터가 없으면 샘플 또는 업로드를 요청하고, 가정은 명확히 표기하세요.",
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
