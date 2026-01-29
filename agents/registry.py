from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from agents.careful_smart import (
    AGENT_ID as CAREFUL_SMART_ID,
    AGENT_NAME as CAREFUL_SMART_NAME,
    DESCRIPTION as CAREFUL_SMART_DESC,
    DEFAULT_MODEL as CAREFUL_SMART_MODEL,
    create_executor as create_careful_smart,
)
from agents.data_analysis import (
    AGENT_ID as DATA_ANALYSIS_ID,
    AGENT_NAME as DATA_ANALYSIS_NAME,
    DESCRIPTION as DATA_ANALYSIS_DESC,
    DEFAULT_MODEL as DATA_ANALYSIS_MODEL,
    create_executor as create_data_analysis,
)
from agents.document_review import (
    AGENT_ID as DOCUMENT_REVIEW_ID,
    AGENT_NAME as DOCUMENT_REVIEW_NAME,
    DESCRIPTION as DOCUMENT_REVIEW_DESC,
    DEFAULT_MODEL as DOCUMENT_REVIEW_MODEL,
    create_executor as create_document_review,
)
from agents.keyword_search import (
    AGENT_ID as KEYWORD_SEARCH_ID,
    AGENT_NAME as KEYWORD_SEARCH_NAME,
    DESCRIPTION as KEYWORD_SEARCH_DESC,
    DEFAULT_MODEL as KEYWORD_SEARCH_MODEL,
    create_executor as create_keyword_search,
)
from agents.news_search import (
    AGENT_ID as NEWS_SEARCH_ID,
    AGENT_NAME as NEWS_SEARCH_NAME,
    DESCRIPTION as NEWS_SEARCH_DESC,
    DEFAULT_MODEL as NEWS_SEARCH_MODEL,
    create_executor as create_news_search,
)
from agents.special_chars import (
    AGENT_ID as SPECIAL_CHARS_ID,
    AGENT_NAME as SPECIAL_CHARS_NAME,
    DESCRIPTION as SPECIAL_CHARS_DESC,
    DEFAULT_MODEL as SPECIAL_CHARS_MODEL,
    create_executor as create_special_chars,
)
from agents.tax_expert import (
    AGENT_ID as TAX_EXPERT_ID,
    AGENT_NAME as TAX_EXPERT_NAME,
    DESCRIPTION as TAX_EXPERT_DESC,
    DEFAULT_MODEL as TAX_EXPERT_MODEL,
    create_executor as create_tax_expert,
)
from agents.tikitaka import (
    AGENT_ID as TIKITAKA_ID,
    AGENT_NAME as TIKITAKA_NAME,
    DESCRIPTION as TIKITAKA_DESC,
    DEFAULT_MODEL as TIKITAKA_MODEL,
    create_executor as create_tikitaka,
)


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    name: str
    description: str
    default_model: str
    factory: Callable


AGENT_SPECS = [
    AgentSpec(
        agent_id=CAREFUL_SMART_ID,
        name=CAREFUL_SMART_NAME,
        description=CAREFUL_SMART_DESC,
        default_model=CAREFUL_SMART_MODEL,
        factory=create_careful_smart,
    ),
    AgentSpec(
        agent_id=TIKITAKA_ID,
        name=TIKITAKA_NAME,
        description=TIKITAKA_DESC,
        default_model=TIKITAKA_MODEL,
        factory=create_tikitaka,
    ),
    AgentSpec(
        agent_id=DOCUMENT_REVIEW_ID,
        name=DOCUMENT_REVIEW_NAME,
        description=DOCUMENT_REVIEW_DESC,
        default_model=DOCUMENT_REVIEW_MODEL,
        factory=create_document_review,
    ),
    AgentSpec(
        agent_id=DATA_ANALYSIS_ID,
        name=DATA_ANALYSIS_NAME,
        description=DATA_ANALYSIS_DESC,
        default_model=DATA_ANALYSIS_MODEL,
        factory=create_data_analysis,
    ),
    AgentSpec(
        agent_id=KEYWORD_SEARCH_ID,
        name=KEYWORD_SEARCH_NAME,
        description=KEYWORD_SEARCH_DESC,
        default_model=KEYWORD_SEARCH_MODEL,
        factory=create_keyword_search,
    ),
    AgentSpec(
        agent_id=NEWS_SEARCH_ID,
        name=NEWS_SEARCH_NAME,
        description=NEWS_SEARCH_DESC,
        default_model=NEWS_SEARCH_MODEL,
        factory=create_news_search,
    ),
    AgentSpec(
        agent_id=SPECIAL_CHARS_ID,
        name=SPECIAL_CHARS_NAME,
        description=SPECIAL_CHARS_DESC,
        default_model=SPECIAL_CHARS_MODEL,
        factory=create_special_chars,
    ),
    AgentSpec(
        agent_id=TAX_EXPERT_ID,
        name=TAX_EXPERT_NAME,
        description=TAX_EXPERT_DESC,
        default_model=TAX_EXPERT_MODEL,
        factory=create_tax_expert,
    ),
]


AGENT_SPEC_BY_ID: Dict[str, AgentSpec] = {spec.agent_id: spec for spec in AGENT_SPECS}
