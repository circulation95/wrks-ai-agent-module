from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def create_agent_executor(model_name="gpt-4o", tools=None, system_prompt=None):
    # 메모리 설정
    memory = MemorySaver()

    # 모델 설정
    model = ChatOpenAI(model_name=model_name, max_tokens=1024)

    # 시스템 프롬프트 설정
    if tools is None:
        tools = []

    agent_executor = create_react_agent(
        model, tools=tools, checkpointer=memory, state_modifier=system_prompt
    )

    return agent_executor
