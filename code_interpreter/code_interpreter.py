from dotenv import load_dotenv

load_dotenv()

from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool, Tool
from langchain.agents import AgentType, create_csv_agent, initialize_agent

# it doesn't work quite well with GPT-3.5
# model_name = "gpt-3.5-turbo"
model_name = "gpt-4"

csv_file = "episode_info.csv"


def main():
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name=model_name),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    python_agent_query = """
        Generate and save in the current working directory 3 QR codes that points to www.google.com
        Assume that the qrcode Python package is already installed, do not try to use pip to install it.
    """

    # python_agent_executor.run(python_agent_query)

    cvs_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model_name=model_name),
        path=csv_file,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # cvs_agent.run("How many columns are there in episode_info.csv?")
    # cvs_agent.run("In the file episode_info.csv, which season has the most episodes?")

    # there is also a Router chain in LangChain
    router_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""Useful when you need to transform natural language and generate and execute Python code from it,
                            returning the results of the code execution.
                            Do not send Python code directly to this tool.""",
            ),
            Tool(
                name="CSVAgent",
                func=cvs_agent.run,
                description=f"""Useful when you need to answer questions about {csv_file} file.
                            It takes the entire question as input and returns the answer after running pandas queries.""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model=model_name),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    router_agent.run(python_agent_query)
    router_agent.run("How many columns are there in episode_info.csv?")


if __name__ == "__main__":
    main()
