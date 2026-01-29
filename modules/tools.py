from typing import Any, List
from .tavily import TavilySearch
from .openai_image import OpenAIImageGenerate
from .base import BaseTool
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


class DocumentSearchTool(BaseTool):
    """업로드 문서를 대상으로 검색하는 도구"""

    def __init__(self, rag_index: RAGIndex, top_k: int = 4, max_chars: int = 2000):
        super().__init__()
        self.rag_index = rag_index
        self.top_k = top_k
        self.max_chars = max_chars

    def _create_tool(self):
        return self

    def __call__(self, query: str, **kwargs: Any) -> Any:
        import json

        top_k = kwargs.get("top_k", self.top_k)
        max_chars = kwargs.get("max_chars", self.max_chars)
        results = self.rag_index.search(query, top_k=top_k)
        context = self.rag_index.build_context(query, top_k=top_k, max_chars=max_chars)
        payload = {"results": results, "context": context}
        return json.dumps(payload, ensure_ascii=False)
