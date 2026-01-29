from attr import dataclass
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
from modules.chat_agents.registry import AGENT_SPEC_BY_ID, AGENT_SPECS
from modules.handler import stream_handler, format_search_result, render_tool_result
from modules.tools import WebSearchTool, ImageGenTool, DocumentSearchTool
from modules.rag import RAGIndex
from openai import OpenAI
from pypdf import PdfReader
import io
from uuid import uuid4

# API KEY 로드
load_dotenv()

st.title("웍스AI")
st.markdown(
    "웍스AI PoC"
)


def new_thread_id() -> str:
    return str(uuid4())


# 대화 기록 초기화
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

# 선택값 기본 설정
if "selected_agent_id" not in st.session_state:
    st.session_state["selected_agent_id"] = AGENT_SPECS[0].agent_id

default_agent = AGENT_SPEC_BY_ID[st.session_state["selected_agent_id"]]
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = default_agent.default_model

# 웹 검색이 필요한 에이전트
SEARCH_AGENT_IDS = {
    "careful_smart",
    "keyword_search",
    "news_search",
    "tikitaka",
}

IMAGE_AGENT_IDS = {
    "careful_smart",
    "tikitaka",
}

DOC_AGENT_IDS = {
    "document_review",
}

# 에이전트별 기본 툴 설정
DEFAULT_SEARCH_CONFIG = {
    "careful_smart": {"topic": "general"},
    "keyword_search": {"topic": "general"},
    "news_search": {"topic": "news"},
    "tikitaka": {"topic": "general"},
}

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 에이전트 선택
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

    # 모델 선택
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
    if not use_search:
        st.caption("선택한 에이전트는 웹 검색을 사용하지 않습니다.")
    if use_doc:
        st.caption("문서 업로드 기반 RAG 검색을 사용합니다.")

    # 대화 유지/초기화 설정
    preserve_history = st.toggle(
        "설정 변경 시 기존 대화 유지",
        value=True,
        key="preserve_history",
    )

    # 검색 결과 개수 설정
    search_result_count = st.slider(
        "검색 결과",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not use_search,
    )

    # 문서 업로드 (문서 검토 에이전트용)
    if use_doc:
        st.subheader("문서 업로드")
        uploaded_files = st.file_uploader(
            "PDF 또는 TXT 파일 업로드",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="doc_uploads",
        )
        st.subheader("문서 RAG 설정")
        doc_top_k = st.slider("문서 검색 Top-K", 1, 8, 4, key="doc_top_k")
        doc_chunk_size = st.number_input(
            "청크 크기(문자)", min_value=300, max_value=2000, value=800, step=50, key="doc_chunk_size"
        )
        doc_overlap = st.number_input(
            "청크 오버랩(문자)", min_value=0, max_value=500, value=150, step=10, key="doc_overlap"
        )
        doc_max_chars = st.number_input(
            "컨텍스트 최대 길이", min_value=500, max_value=6000, value=2000, step=100, key="doc_max_chars"
        )
        build_docs_btn = None

    # include_domains 설정
    st.subheader("검색 도메인 설정")
    default_topic = DEFAULT_SEARCH_CONFIG.get(selected_agent_id, {}).get(
        "topic", "general"
    )
    search_topic = st.selectbox(
        "검색 주제",
        ["general", "news"],
        index=0 if default_topic == "general" else 1,
        disabled=not use_search,
    )
    new_domain = st.text_input("추가할 도메인 입력", disabled=not use_search)
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("도메인 추가", key="add_domain", disabled=not use_search):
            if new_domain and new_domain not in st.session_state["include_domains"]:
                st.session_state["include_domains"].append(new_domain)

    # 현재 등록된 도메인 목록 표시
    st.write("등록된 도메인 목록:")
    for idx, domain in enumerate(st.session_state["include_domains"]):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(domain)
        with col2:
            if st.button("삭제", key=f"del_{idx}", disabled=not use_search):
                st.session_state["include_domains"].pop(idx)
                st.rerun()

    # 설정 변경 자동 반영
    apply_btn = st.button("설정 완료", type="primary")


@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


# 이전 대화 출력
def print_messages():
    for message in st.session_state["messages"]:
        if message.msg_type == "text":
            st.chat_message(message.chat_message.role).write(
                message.chat_message.content
            )
        elif message.msg_type == "tool_result":
            with st.expander(f"툴: {message.tool_name}"):
                render_tool_result(message.tool_name, message.chat_message.content)


# 메시지 추가
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
                chat_message=ChatMessage(
                    role="assistant", content=message
                ),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )


# 초기화 버튼 처리
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = new_thread_id()

# 이전 대화 기록 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고 메시지용 영역
warning_msg = st.empty()

def build_agent_config():
    return {
        "agent_id": selected_agent_id,
        "model_name": selected_model,
        "use_search": use_search,
        "use_image": use_image,
        "use_doc": use_doc,
        "search_topic": search_topic,
        "search_result_count": search_result_count,
        "include_domains": tuple(st.session_state["include_domains"]),
        "doc_hashes": tuple(st.session_state.get("doc_hashes", [])),
        "doc_top_k": st.session_state.get("doc_top_k", 4),
        "doc_chunk_size": st.session_state.get("doc_chunk_size", 800),
        "doc_overlap": st.session_state.get("doc_overlap", 150),
        "doc_max_chars": st.session_state.get("doc_max_chars", 2000),
    }


def _extract_text_from_uploads(files):
    texts = []
    hashes = []
    for f in files or []:
        data = f.read()
        if not data:
            continue
        h = RAGIndex.hash_bytes(data)
        hashes.append(h)
        if f.type == "application/pdf":
            reader = PdfReader(io.BytesIO(data))
            page_texts = []
            for page in reader.pages:
                page_texts.append(page.extract_text() or "")
            texts.append("\n".join(page_texts))
        else:
            try:
                texts.append(data.decode("utf-8"))
            except Exception:
                texts.append(data.decode("latin-1"))
    return texts, hashes


def _ensure_doc_index(files):
    if not files:
        st.session_state["doc_index"] = None
        st.session_state["doc_hashes"] = []
        return

    texts, hashes = _extract_text_from_uploads(files)
    prev_hashes = st.session_state.get("doc_hashes", [])
    if hashes == prev_hashes and st.session_state.get("doc_index") is not None:
        return

    try:
        client = OpenAI()
        rag = RAGIndex(
            client=client,
            chunk_size=st.session_state.get("doc_chunk_size", 800),
            overlap=st.session_state.get("doc_overlap", 150),
        )
        for text in texts:
            rag.add_text(text)
        st.session_state["doc_index"] = rag
        st.session_state["doc_hashes"] = hashes
    except Exception:
        st.session_state["doc_index"] = None


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

    if use_doc:
        doc_index = st.session_state.get("doc_index")
        if doc_index is not None:
            doc_tool = DocumentSearchTool(
                doc_index,
                top_k=st.session_state.get("doc_top_k", 4),
                max_chars=st.session_state.get("doc_max_chars", 2000),
            ).create()
            doc_tool.name = "document_search"
            doc_tool.description = "Search uploaded documents for relevant context"
            tools.append(doc_tool)

    st.session_state["react_agent"] = selected_agent.factory(
        model_name=selected_model,
        tools=tools,
    )
    st.session_state["thread_id"] = new_thread_id()
    st.session_state["last_agent_config"] = build_agent_config()
    if auto:
        st.session_state["auto_recreated"] = True

    if not preserve_history:
        st.session_state["messages"] = []


# 설정 변경 자동 처리
current_config = build_agent_config()
if "last_agent_config" not in st.session_state:
    st.session_state["last_agent_config"] = current_config
    create_agent_from_config()
elif current_config != st.session_state["last_agent_config"]:
    create_agent_from_config(auto=True)

# 문서 업로드 처리
if use_doc and "doc_uploads" in st.session_state:
    if not st.session_state["doc_uploads"]:
        st.info("문서를 업로드하면 자동으로 색인이 생성됩니다.")
    else:
        _ensure_doc_index(st.session_state["doc_uploads"])
        if st.session_state.get("doc_index") is None:
            st.warning("문서를 불러오지 못했습니다.")
        else:
            st.success("문서 색인이 생성되었습니다.")

# 설정 버튼 처리 (수동 갱신도 지원)
if apply_btn:
    create_agent_from_config()

# 자동 재생성 안내
if st.session_state.get("auto_recreated"):
    st.info("설정이 변경되어 새로 시작합니다.")
    st.session_state["auto_recreated"] = False

# 사용자 입력 처리
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

            # 대화 기록 저장
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
