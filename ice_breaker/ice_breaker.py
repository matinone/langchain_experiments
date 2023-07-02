from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import scrape_linkedin_profile, get_linkedin_url

load_dotenv()

summary_template = """
    Given the LinkedIn information {information} about a person, I need you to create:
    1. A short summary of the person.
    2. Two interesting facts about the person.
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=summary_prompt_template)

profile_url = get_linkedin_url(name="Eden Marco Udemy")
linkedin_data = scrape_linkedin_profile(profile_url)
response = chain.run(information=linkedin_data)
print(response)
