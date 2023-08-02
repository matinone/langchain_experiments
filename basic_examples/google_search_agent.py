from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools

model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"

llm = ChatOpenAI(temperature=0, model_name=model_name)
tool_names = ["serpapi"]
tools = load_tools(tool_names=tool_names)

agent = initialize_agent(
    tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True
)

agent.run("What is LangChain?")
agent.run("Is Argentina participating in the FIFA Women World Cup 2023?")
agent.run(
    "Was Argentina eliminated in the first round of the FIFA Women World Cup 2023?"
)
