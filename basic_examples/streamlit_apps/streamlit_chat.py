import streamlit as st
import numpy as np


st.title("Echo Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display messages from chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# receive user input and display it
prompt = st.chat_input("Say something...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # generate echo response
    response = f"Echo: {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
