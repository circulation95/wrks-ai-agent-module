from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from openai import OpenAI


class ImageGenInput(BaseModel):
    """Input for image generation."""

    prompt: str = Field(description="이미지를 생성할 텍스트 프롬프트")


class OpenAIImageGenerate(BaseTool):
    """
    Tool that generates images using OpenAI Images API.
    """

    name: str = "image_generate"
    description: str = (
        "Generate an image from a text prompt. Input should be a prompt string."
    )
    args_schema: type[BaseModel] = ImageGenInput

    client: Optional[OpenAI] = None
    model: str = "gpt-image-1"
    size: str = "1024x1024"
    quality: str = "auto"
    n: int = 1

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "auto",
        n: int = 1,
    ):
        super().__init__()
        if OpenAI is None:
            raise ImportError("openai is not installed. Run: poetry add openai")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
        self.quality = quality
        self.n = n

    def _run(self, prompt: str, **kwargs) -> str:
        size = kwargs.get("size", self.size)
        quality = kwargs.get("quality", self.quality)
        n = kwargs.get("n", self.n)
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )

        images = []
        image_dir = Path("artifacts") / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for item in response.data:
            if getattr(item, "url", None):
                images.append(item.url)
            elif getattr(item, "b64_json", None):
                image_bytes = base64.b64decode(item.b64_json)
                file_path = image_dir / f"image_{uuid4().hex}.png"
                file_path.write_bytes(image_bytes)
                images.append(str(file_path))

        payload = {"images": images}
        return json.dumps(payload, ensure_ascii=False)
