from dotenv import load_dotenv

load_dotenv()

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_agent import scrape_linkedin_profile, get_linkedin_url
from output_parsers import person_info_parser, PersonInformation


def generate_ice_breaker(name: str) -> dict[str, PersonInformation | str]:
    summary_template = """
        Given the LinkedIn information {information} about a person, I need you to create:
        1. A short summary of the person.
        2. Two interesting facts about the person.
        3. A topic that may be of interest for that person.
        4. Two Ice Breaker phrases to start a conversation with that person.

        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_info_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    profile_url = get_linkedin_url(name=name)
    linkedin_data = scrape_linkedin_profile(profile_url, fake_data=True)
    result = chain.run(information=linkedin_data)

    return {
        "person_information": person_info_parser.parse(result),
        "profile_pic_url": linkedin_data.get("profile_pic_url"),
    }


if __name__ == "__main__":
    name = "Matias Nicolas Brignone"
    ice_breaker = generate_ice_breaker(name=name)
    print(ice_breaker)
