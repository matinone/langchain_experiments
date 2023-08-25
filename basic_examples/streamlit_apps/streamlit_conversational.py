import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.title("Streamlit Conversational App")
st.header("I am a sarcastic but helpful assistant...")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [
        SystemMessage(content="You are a sarcastic but helpful assistant.")
    ]


def load_answer(question: str) -> str:
    if chat is None:
        return "Missing OpenAI API key, can't generate a response."

    st.session_state.sessionMessages.append(HumanMessage(content=question))
    assistant_answer = chat(st.session_state.sessionMessages)
    st.session_state.sessionMessages.append(AIMessage(content=assistant_answer.content))

    return assistant_answer.content


def get_text() -> str:
    input_text = st.text_input("You: ", key=input)
    return input_text


chat = None
if not openai_api_key.startswith("sk"):
    st.warning("Please enter your OpenAI API key.")
else:
    chat = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)


user_input = get_text()
submit = st.button("Generate")

if submit:
    response = load_answer(user_input)
    st.subheader("Answer:")

    st.write(response)
