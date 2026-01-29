from __future__ import annotations

from agents.common import build_system_prompt
from modules.agent import create_agent_executor

AGENT_ID = "document_review"
AGENT_NAME = "문서 파일 검토"
DESCRIPTION = "문서 업로드 기반 요약 및 질의응답"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = build_system_prompt(
    agent_name=AGENT_NAME,
    role_summary="문서 요약과 Q&A에 특화된 에이전트입니다.",
    focus_summary="PDF 등 문서의 핵심 요약, 필요한 근거 인용, 질의응답",
    tool_policy=(
        "문서가 제공되지 않으면 업로드를 요청하고, 업로드된 문서는 document_search 도구로 조회하세요. "
        "웹검색은 보조로만 사용하세요."
    ),
)


def create_executor(tools=None, model_name: str = DEFAULT_MODEL):
    return create_agent_executor(
        model_name=model_name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
