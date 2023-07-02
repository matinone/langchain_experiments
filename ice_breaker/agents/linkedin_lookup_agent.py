from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from langchain.serpapi import SerpAPIWrapper


class CustomSerpAPIWrapper(SerpAPIWrapper):
    """
    Redefine _process_response to make sure that the LinkedIn URL is returned.
    """

    def __init__(self):
        super(CustomSerpAPIWrapper, self).__init__()

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif (
            "answer_box" in res.keys()
            and "snippet_highlighted_words" in res["answer_box"].keys()
        ):
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif (
            "sports_results" in res.keys()
            and "game_spotlight" in res["sports_results"].keys()
        ):
            toret = res["sports_results"]["game_spotlight"]
        elif (
            "knowledge_graph" in res.keys()
            and "description" in res["knowledge_graph"].keys()
        ):
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["link"]

        else:
            toret = "No good search result found"

        return toret


def get_profile_url(text: str) -> str:
    """
    Search for LinkedIn Profile Page
    """
    search = CustomSerpAPIWrapper()
    response = search.run(f"{text}")

    return response


def lookup(name: str) -> str:
    """ """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    lookup_template = """
        Given the full name {full_name}, I want you to give me the link to the LinkedIn
        profile pafe of that person.
        Your answer should only contain a URL, nothing else.
    """
    prompt_template = PromptTemplate(
        template=lookup_template, input_variables=["full_name"]
    )

    agent_tools = [
        Tool(
            name="Crawl Google for LinkedIn profile page",
            func=get_profile_url,
            description="useful when you need to get the LinkedIn page URL of a person",
        )
    ]

    agent = initialize_agent(
        tools=agent_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    linkedin_profile_url = agent.run(prompt_template.format_prompt(full_name=name))

    return linkedin_profile_url
