import streamlit as st
import openai


st.title("ChatGPT Clone")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai.api_key = openai_api_key

# set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display messages from chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# receive user input and display it
prompt = st.chat_input("Ask something...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in openai.ChatCompletion.create(
            model=st.session_state.openai_model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")

        # final response
        message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
