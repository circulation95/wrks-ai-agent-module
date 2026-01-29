from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class CodeGenInput(BaseModel):
    """Input for code generation."""

    task: str = Field(description="작성할 코드 작업 설명")
    language: str = Field(description="프로그래밍 언어", default="python")


class CodeGenerate(BaseTool):
    """Generate code from a task description using ChatOpenAI."""

    name: str = "code_generate"
    description: str = "Generate code for a given task description."
    args_schema: type[BaseModel] = CodeGenInput

    model_name: str = "gpt-4o-mini"
    max_tokens: int = 800
    temperature: float = 0.2
    client: Optional[ChatOpenAI] = None

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = ChatOpenAI(
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _run(self, task: str, language: str = "python", **kwargs) -> str:
        prompt = (
            "You are a senior software engineer. "
            "Generate concise, correct code for the task. "
            "Return only a single code block.\n\n"
            f"Language: {language}\n"
            f"Task: {task}\n"
        )
        response = self.client.invoke(prompt)
        return response.content
