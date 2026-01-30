from __future__ import annotations

from attr import dataclass
import os
import io
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from pypdf import PdfReader

from modules.util import langsmith
from agents.registry import AGENT_SPEC_BY_ID, AGENT_SPECS
from modules.handler import stream_handler, render_tool_result
from tools import WebSearchTool, ImageGenTool, DocumentSearchTool, CodeGenTool
from tools import RAGIndex

# API KEY 로드
load_dotenv()
langsmith("웍스AI")

st.title("웍스AI")
st.markdown("웍스AI PoC")


def new_thread_id() -> str:
    return str(uuid4())


# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["react_agent"] = None

# include_domains 초기화
if "include_domains" not in st.session_state:
    st.session_state["include_domains"] = []

# thread id 초기화
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = new_thread_id()

# 선택된 에이전트 초기값
if "selected_agent_id" not in st.session_state:
    st.session_state["selected_agent_id"] = AGENT_SPECS[0].agent_id

default_agent = AGENT_SPEC_BY_ID[st.session_state["selected_agent_id"]]
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = default_agent.default_model

# 검색/이미지/코드/문서 지원 에이전트
SEARCH_AGENT_IDS = {
    "careful_smart",
    "keyword_search",
    "news_search",
    "tikitaka",
}

IMAGE_AGENT_IDS = {
    "careful_smart",
}

CODE_AGENT_IDS = {
    "careful_smart",
}

DOC_DB_PATH = "data/doc_index"
DOCS_ROOT = "documents"
TAX_DOC_PATH = os.path.join(DOCS_ROOT, "tax_expert")
TAX_DOC_DB_PATH = "data/doc_index_tax_expert"

DOC_AGENT_IDS = {
    "document_review",
    "tax_expert",
}

# 에이전트별 기본 검색 설정
DEFAULT_SEARCH_CONFIG = {
    "careful_smart": {"topic": "general"},
    "keyword_search": {"topic": "general"},
    "news_search": {"topic": "news"},
    "tikitaka": {"topic": "general"},
}

# 사이드바 구성
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    agent_options = [spec.agent_id for spec in AGENT_SPECS]
    selected_agent_id = st.selectbox(
        "에이전트 선택",
        agent_options,
        format_func=lambda agent_id: AGENT_SPEC_BY_ID[agent_id].name,
        key="selected_agent_id",
        index=agent_options.index(st.session_state["selected_agent_id"]),
    )
    selected_agent = AGENT_SPEC_BY_ID[selected_agent_id]
    st.caption(selected_agent.description)

    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"]
    if selected_agent.default_model not in model_options:
        model_options.append(selected_agent.default_model)
    default_model_index = model_options.index(selected_agent.default_model)
    selected_model = st.selectbox(
        "LLM 선택",
        model_options,
        key="selected_model",
        index=default_model_index,
    )

    use_search = selected_agent_id in SEARCH_AGENT_IDS
    use_image = selected_agent_id in IMAGE_AGENT_IDS
    use_doc = selected_agent_id in DOC_AGENT_IDS
    use_code = selected_agent_id in CODE_AGENT_IDS

    if not use_search:
        st.caption("선택한 에이전트는 검색을 사용하지 않습니다.")
    if use_doc:
        st.caption("문서 업로드 기반 RAG 검색을 사용합니다.")
    if use_code:
        st.caption("코드 생성 도구를 사용합니다.")

    preserve_history = st.toggle(
        "설정 변경 시 기존 대화 유지",
        value=True,
        key="preserve_history",
    )

    search_result_count = st.slider(
        "검색 결과",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not use_search,
    )

    if use_doc:
        if selected_agent_id == "tax_expert":
            st.subheader("세무 전문가 문서")
            st.caption("documents/tax_expert 폴더의 문서로 검색합니다.")
        else:
            st.subheader("문서 업로드")
            uploaded_files = st.file_uploader(
                "PDF 또는 TXT 파일 업로드",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key="doc_uploads",
            )

        st.subheader("문서 RAG 설정")
        st.slider("문서 검색 Top-K", 1, 8, 4, key="doc_top_k")
        st.number_input(
            "청크 크기(문자)",
            min_value=300,
            max_value=2000,
            value=800,
            step=50,
            key="doc_chunk_size",
        )
        st.number_input(
            "청크 오버랩(문자)",
            min_value=0,
            max_value=500,
            value=150,
            step=10,
            key="doc_overlap",
        )
        st.number_input(
            "컨텍스트 최대 길이",
            min_value=500,
            max_value=6000,
            value=2000,
            step=100,
            key="doc_max_chars",
        )
        st.slider("문서 검색 Fetch-K", 5, 50, 30, key="doc_fetch_k")
        st.slider("리랭크 Top-N", 1, 10, 8, key="doc_top_n")
        st.toggle("Jina 리랭커 사용", value=True, key="doc_use_rerank")

    st.subheader("검색 파라미터 설정")
    default_topic = DEFAULT_SEARCH_CONFIG.get(selected_agent_id, {}).get(
        "topic", "general"
    )
    search_topic = st.selectbox(
        "검색 주제",
        ["general", "news"],
        index=0 if default_topic == "general" else 1,
        disabled=not use_search,
    )
    new_domain = st.text_input("추가 도메인 입력", disabled=not use_search)
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("도메인 추가", key="add_domain", disabled=not use_search):
            if new_domain and new_domain not in st.session_state["include_domains"]:
                st.session_state["include_domains"].append(new_domain)

    st.write("등록된 도메인 목록:")
    for idx, domain in enumerate(st.session_state["include_domains"]):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(domain)
        with col2:
            if st.button("삭제", key=f"del_{idx}", disabled=not use_search):
                st.session_state["include_domains"].pop(idx)
                st.rerun()

    apply_btn = st.button("설정 완료", type="primary")



@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


def print_messages():
    for message in st.session_state["messages"]:
        if message.msg_type == "text":
            st.chat_message(message.chat_message.role).write(
                message.chat_message.content
            )
        elif message.msg_type == "tool_result":
            with st.expander(f"툴: {message.tool_name}"):
                render_tool_result(message.tool_name, message.chat_message.content)


def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role="assistant", content=message),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )


if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = new_thread_id()

print_messages()

user_input = st.chat_input("질문을 입력해주세요")
warning_msg = st.empty()


def build_agent_config():
    return {
        "agent_id": selected_agent_id,
        "model_name": selected_model,
        "use_search": use_search,
        "use_image": use_image,
        "use_doc": use_doc,
        "use_code": use_code,
        "search_topic": search_topic,
        "search_result_count": search_result_count,
        "include_domains": tuple(st.session_state["include_domains"]),
        "doc_hashes": tuple(st.session_state.get("doc_hashes", [])),
        "doc_top_k": st.session_state.get("doc_top_k", 4),
        "doc_chunk_size": st.session_state.get("doc_chunk_size", 800),
        "doc_overlap": st.session_state.get("doc_overlap", 150),
        "doc_max_chars": st.session_state.get("doc_max_chars", 2000),
        "doc_fetch_k": st.session_state.get("doc_fetch_k", 30),
        "doc_top_n": st.session_state.get("doc_top_n", 8),
        "doc_use_rerank": st.session_state.get("doc_use_rerank", True),
    }


def _extract_text_from_uploads(files):
    docs = []
    hashes = []
    for f in files or []:
        data = f.read()
        if not data:
            continue
        h = RAGIndex.hash_bytes(data)
        hashes.append(h)
        if f.type == "application/pdf":
            reader = PdfReader(io.BytesIO(data))
            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    docs.append({"text": page_text, "source": f"{f.name} #page {idx}"})
        else:
            try:
                docs.append({"text": data.decode("utf-8"), "source": f.name})
            except Exception:
                docs.append({"text": data.decode("latin-1"), "source": f.name})
    return docs, hashes


def _extract_text_from_dir(dir_path: str):
    docs = []
    hashes = []
    base = Path(dir_path)
    if not base.exists():
        return docs, hashes

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            continue
        data = file_path.read_bytes()
        if not data:
            continue
        hashes.append(RAGIndex.hash_bytes(data))
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    docs.append(
                        {"text": page_text, "source": f"{file_path.name} #page {idx}"}
                    )
        else:
            try:
                text = data.decode("utf-8")
            except Exception:
                text = data.decode("latin-1")
            if text.strip():
                docs.append({"text": text, "source": file_path.name})
    return docs, hashes


def _ensure_doc_index_from_docs(docs, hashes, db_path: str = DOC_DB_PATH):
    if not docs:
        st.session_state["doc_index"] = None
        st.session_state["doc_hashes"] = []
        return

    prev_hashes = st.session_state.get("doc_hashes", [])
    if hashes == prev_hashes and st.session_state.get("doc_index") is not None:
        return

    try:
        rag = RAGIndex(
            db_path=db_path,
            chunk_size=st.session_state.get("doc_chunk_size", 800),
            overlap=st.session_state.get("doc_overlap", 150),
            fetch_k=st.session_state.get("doc_fetch_k", 30),
            top_n=st.session_state.get("doc_top_n", 8),
            use_rerank=st.session_state.get("doc_use_rerank", True),
        )
        rag.ensure_index(docs, hashes)
        st.session_state["doc_index"] = rag
        st.session_state["doc_hashes"] = hashes
    except Exception:
        st.session_state["doc_index"] = None


def _ensure_doc_index(files):
    if not files:
        st.session_state["doc_index"] = None
        st.session_state["doc_hashes"] = []
        return

    docs, hashes = _extract_text_from_uploads(files)
    _ensure_doc_index_from_docs(docs, hashes)


def _ensure_doc_index_from_dir(dir_path: str, db_path: str = DOC_DB_PATH):
    docs, hashes = _extract_text_from_dir(dir_path)
    _ensure_doc_index_from_docs(docs, hashes, db_path=db_path)


def _append_doc_sources(answer: str, tool_args: list[dict]) -> str:
    sources: list[str] = []
    for tool_arg in tool_args:
        if tool_arg.get("tool_name") != "document_search":
            continue
        try:
            payload = json.loads(tool_arg.get("tool_result", "{}"))
        except Exception:
            continue
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for item in results:
            src = item.get("source")
            if src and src not in sources:
                sources.append(src)

    if not sources:
        return answer

    if "**출처**" in answer:
        return answer

    lines = ["**출처**"]
    for i, src in enumerate(sources, start=1):
        lines.append(f"[{i}] {src}  ")
    return answer + " " + " ".join(lines)


def create_agent_from_config(auto=False):
    tools = []
    if use_search:
        tool = WebSearchTool().create()
        tool.max_results = search_result_count
        tool.include_domains = st.session_state["include_domains"]
        tool.topic = search_topic
        tools = [tool]

    if use_image:
        image_tool = ImageGenTool().create()
        tools.append(image_tool)

    if use_code:
        code_tool = CodeGenTool().create()
        tools.append(code_tool)

    if use_doc:
        if selected_agent_id == "tax_expert" and st.session_state.get("doc_index") is None:
            _ensure_doc_index_from_dir(TAX_DOC_PATH, db_path=TAX_DOC_DB_PATH)

        doc_index = st.session_state.get("doc_index")
        if doc_index is not None:
            doc_tool = DocumentSearchTool(
                doc_index,
                top_k=st.session_state.get("doc_top_k", 4),
                max_chars=st.session_state.get("doc_max_chars", 2000),
            )
            doc_tool.name = "document_search"
            if selected_agent_id == "tax_expert":
                doc_tool.description = "Search tax expert documents for relevant context"
            else:
                doc_tool.description = "Search uploaded documents for relevant context"
            tools.append(doc_tool)

    st.session_state["react_agent"] = selected_agent.factory(
        model_name=selected_model,
        tools=tools,
    )
    st.session_state["thread_id"] = new_thread_id()
    st.session_state["last_agent_config"] = build_agent_config()
    st.session_state["graph_viz_shown"] = False
    if auto:
        st.session_state["auto_recreated"] = True

    if not preserve_history:
        st.session_state["messages"] = []


current_config = build_agent_config()
if "last_agent_config" not in st.session_state:
    st.session_state["last_agent_config"] = current_config
    create_agent_from_config()
elif current_config != st.session_state["last_agent_config"]:
    create_agent_from_config(auto=True)

if use_doc:
    if selected_agent_id == "tax_expert":
        _ensure_doc_index_from_dir(TAX_DOC_PATH, db_path=TAX_DOC_DB_PATH)
        if st.session_state.get("doc_index") is None:
            st.warning(
                "세무 문서 폴더에서 문서를 읽어오지 못했습니다."
            )
        else:
            st.success(
                "세무 문서 색인을 완료했습니다."
            )
    elif "doc_uploads" in st.session_state:
        if not st.session_state["doc_uploads"]:
            st.info(
                "문서를 업로드하면 자동으로 색인을 생성합니다."
            )
        else:
            _ensure_doc_index(st.session_state["doc_uploads"])
            if st.session_state.get("doc_index") is None:
                st.warning(
                    "문서를 불러오지 못했습니다."
                )
            else:
                st.success(
                    "문서 색인을 생성했습니다."
                )

if apply_btn:
    create_agent_from_config()

if st.session_state.get("auto_recreated"):
    st.info("설정이 변경되어 새로 시작합니다")
    st.session_state["auto_recreated"] = False

if "graph_viz_shown" not in st.session_state:
    st.session_state["graph_viz_shown"] = False

if not st.session_state["graph_viz_shown"]:
    agent = st.session_state.get("react_agent")
    if agent is not None and hasattr(agent, "get_graph"):
        try:
            graph_png = agent.get_graph().draw_mermaid_png()
            with st.sidebar:
                st.subheader("그래프")
                st.image(graph_png)
            st.session_state["graph_viz_shown"] = True
        except Exception:
            pass

if user_input:
    agent = st.session_state["react_agent"]

    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            container = st.empty()

            container_messages, tool_args, agent_answer = stream_handler(
                container,
                agent,
                {
                    "messages": [
                        ("human", user_input),
                    ]
                },
                config,
            )

            agent_answer = _append_doc_sources(agent_answer, tool_args)

            add_message("user", user_input)
            for tool_arg in tool_args:
                add_message(
                    "assistant",
                    tool_arg["tool_result"],
                    "tool_result",
                    tool_arg["tool_name"],
                )
            add_message("assistant", agent_answer)
    else:
        warning_msg.warning("사이드바에서 설정을 완료해주세요.")
