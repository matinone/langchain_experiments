import streamlit as st
from langchain.llms import OpenAI


st.title("Streamlit Basic App")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(input_text: str) -> str:
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm(input_text)


with st.form("my_form"):
    text = st.text_area("Enter text:", "Example: what is the currency of Argentina?")
    submitted = st.form_submit_button("Submit")

    if not openai_api_key.startswith("sk"):
        st.warning("Please enter your OpenAI API key.")

    if submitted and openai_api_key.startswith("sk"):
        response = generate_response(text)
        st.info(response)
