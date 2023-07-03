from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class PersonInformation(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: list[str] = Field(description="Interesting facts about the person")
    topic_of_interest: str = Field(
        description="Topic that may be of interest for the person"
    )
    ice_breakers: list[str] = Field(
        description="Ice breaker phrases to start a conversation with the person"
    )

    def to_dict(self):
        return {
            "summary": self.summary,
            "facts": self.facts,
            "topic_of_interest": self.topic_of_interest,
            "ice_breakers": self.ice_breakers,
        }


person_info_parser = PydanticOutputParser(pydantic_object=PersonInformation)
