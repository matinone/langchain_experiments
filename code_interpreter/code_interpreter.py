from dotenv import load_dotenv

load_dotenv()

from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool
from langchain.agents import AgentType


def main():
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    python_agent_executor.run(
        """
        Generate and save in the current working directory 3 QR codes that points to www.google.com
        Assume that the qrcode Python package is already installed, do not try to use pip to install it.
        """
    )


if __name__ == "__main__":
    main()
