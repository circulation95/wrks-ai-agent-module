import streamlit as st


def get_current_tool_message(tool_args, tool_call_id):
    """
    Get the tool message corresponding to the given tool call ID.

    Args:
        tool_args (list): List of tool arguments
        tool_call_id (str): ID of the tool call to find

    Returns:
        dict: Tool message if found, None otherwise
    """
    if not tool_call_id:
        return None

    for tool_arg in tool_args:
        if tool_arg.get("tool_call_id") == tool_call_id:
            return tool_arg
    return None


def format_search_result(results):
    """
    Format search results into a markdown string.

    Args:
        results (str): JSON string containing search results

    Returns:
        str: Formatted markdown string with search results
    """
    import json

    results = json.loads(results)

    answer = ""
    for result in results:
        answer += f'**[{result["title"]}]({result["url"]})**\n\n'
        answer += f'{result["content"]}\n\n'
        answer += f"\uc2e0\ub8b0\ub3c4: {result['score']}\n\n"
        answer += "\n-----\n"
    return answer


def stream_handler(streamlit_container, agent_executor, inputs, config):
    """
    Handle streaming of agent execution results in a Streamlit container.

    Args:
        streamlit_container (streamlit.container): Streamlit container to display results
        agent_executor: Agent executor instance
        inputs: Input data for the agent
        config: Configuration settings

    Returns:
        tuple: (container, tool_args, agent_answer)
            - container: Streamlit container with displayed results
            - tool_args: List of tool arguments used
            - agent_answer: Final answer from the agent
    """
    tool_args = []
    agent_answer = ""
    agent_message = None

    container = streamlit_container.container()
    with container:
        for chunk_msg, metadata in agent_executor.stream(
            inputs, config, stream_mode="messages"
        ):
            if hasattr(chunk_msg, "tool_calls") and chunk_msg.tool_calls:
                tool_call = chunk_msg.tool_calls[0]
                tool_arg = {
                    "tool_name": tool_call.get("name", ""),
                    "tool_result": "",
                    "tool_call_id": tool_call.get("id"),
                }
                if tool_arg["tool_name"]:
                    tool_args.append(tool_arg)

            if metadata.get("langgraph_node") == "tools":
                tool_call_id = getattr(chunk_msg, "tool_call_id", None)
                current_tool_message = get_current_tool_message(tool_args, tool_call_id)
                if current_tool_message:
                    current_tool_message["tool_result"] = chunk_msg.content
                    with st.status(f"\ud234: {current_tool_message['tool_name']}"):
                        render_tool_result(
                            current_tool_message["tool_name"],
                            current_tool_message["tool_result"],
                        )

            if metadata.get("langgraph_node") == "agent":
                if chunk_msg.content:
                    if agent_message is None:
                        agent_message = st.empty()
                    agent_answer += chunk_msg.content
                    agent_message.markdown(agent_answer)

        return container, tool_args, agent_answer


def render_tool_result(tool_name: str, tool_result: str):
    import json

    if tool_name == "web_search":
        st.markdown(format_search_result(tool_result))
        return

    if tool_name == "image_generate":
        try:
            payload = json.loads(tool_result)
        except Exception:
            st.markdown(tool_result)
            return

        images = payload.get("images", []) if isinstance(payload, dict) else []
        if not images:
            st.markdown("\uc774\ubbf8\uc9c0 \uacb0\uacfc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
            return

        for idx, image in enumerate(images, start=1):
            st.image(image, caption=f"\uc0dd\uc131 \uc774\ubbf8\uc9c0 {idx}")
        return

    if tool_name == "document_search":
        try:
            payload = json.loads(tool_result)
        except Exception:
            st.markdown(tool_result)
            return

        results = payload.get("results", []) if isinstance(payload, dict) else []
        context = payload.get("context", "") if isinstance(payload, dict) else ""
        if not results:
            st.markdown("\uac80\uc0c9 \uacb0\uacfc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
            return

        if context:
            with st.expander("RAG \ucee8\ud14d\uc2a4\ud2b8"):
                st.markdown(context)

        for idx, item in enumerate(results, start=1):
            content = item.get("content", "")
            source = item.get("source", "uploaded_document")
            st.markdown(
                f"**\ubb38\uc11c \uc870\uac01 {idx} (source: {source})**"
            )
            st.markdown(content)
        return

    st.markdown(tool_result)
