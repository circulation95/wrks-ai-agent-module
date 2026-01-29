from typing import Any, List
from langchain_core.tools import BaseTool as LCToolBase
from pydantic import BaseModel, Field

from modules.base import BaseTool
from .tavily import TavilySearch
from .openai_image import OpenAIImageGenerate
from .codegen import CodeGenerate
from .rag import RAGIndex


class WebSearchTool(BaseTool[TavilySearch]):
    """웹 검색을 수행하는 도구 클래스"""

    def __init__(
        self,
        topic: str = "general",
        max_results: int = 3,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        format_output: bool = False,
        include_domains: List[str] = [],
        exclude_domains: List[str] = [],
    ):
        """WebSearchTool 초기화 메서드"""
        super().__init__()
        self.topic = topic
        self.max_results = max_results
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.format_output = format_output
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains

    def _create_tool(self) -> TavilySearch:
        """TavilySearch 객체를 생성하고 설정하는 내부 메서드"""
        search = TavilySearch(
            topic=self.topic,
            max_results=self.max_results,
            include_answer=self.include_answer,
            include_raw_content=self.include_raw_content,
            include_images=self.include_images,
            format_output=self.format_output,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
        )
        search.name = "web_search"
        search.description = "Use this tool to search on the web"
        return search

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """도구를 실행하는 메서드"""
        tool = self._create_tool()
        return tool(*args, **kwargs)


class ImageGenTool(BaseTool[OpenAIImageGenerate]):
    """이미지 생성 툴"""

    def __init__(
        self,
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "auto",
        n: int = 1,
    ):
        """ImageGenTool 초기화 메서드"""
        super().__init__()
        self.model = model
        self.size = size
        self.quality = quality
        self.n = n

    def _create_tool(self) -> OpenAIImageGenerate:
        """OpenAIImageGenerate 객체를 생성하고 설정하는 내부 메서드"""
        image_tool = OpenAIImageGenerate(
            model=self.model,
            size=self.size,
            quality=self.quality,
            n=self.n,
        )
        image_tool.name = "image_generate"
        image_tool.description = "Use this tool to generate images from a prompt"
        return image_tool

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """도구를 실행하는 메서드"""
        tool = self._create_tool()
        return tool(*args, **kwargs)


class DocumentSearchInput(BaseModel):
    """Input for document search."""

    query: str = Field(description="문서 검색 질의")


class DocumentSearchTool(LCToolBase):
    """업로드 문서를 대상으로 검색하는 도구"""

    name: str = "document_search"
    description: str = "Search uploaded documents for relevant context"
    args_schema: type[BaseModel] = DocumentSearchInput

    def __init__(self, rag_index: RAGIndex, top_k: int = 4, max_chars: int = 2000):
        super().__init__()
        object.__setattr__(self, "rag_index", rag_index)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "max_chars", max_chars)

    def _run(self, query: str, **kwargs: Any) -> Any:
        import json

        top_k = kwargs.get("top_k", self.top_k)
        max_chars = kwargs.get("max_chars", self.max_chars)
        results = self.rag_index.search(query, top_k=top_k)
        context = self.rag_index.build_context(query, top_k=top_k, max_chars=max_chars)
        payload = {"results": results, "context": context}
        return json.dumps(payload, ensure_ascii=False)


class CodeGenTool(BaseTool[CodeGenerate]):
    """코드 생성 도구"""

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

    def _create_tool(self) -> CodeGenerate:
        code_tool = CodeGenerate(
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        code_tool.name = "code_generate"
        code_tool.description = "Use this tool to generate code from a task description"
        return code_tool

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tool = self._create_tool()
        return tool(*args, **kwargs)
