import os
from dotenv import load_dotenv
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class SafetyChecker(BaseModel):
    model_name: str = Field(default="llama3-70b-8192")
    temperature: float = Field(default=0.1)
    api_key: Optional[str] = None
    prompt_safe: Optional[ChatPromptTemplate] = None
    system_prompt: str = Field(default="""Task type
Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

Policy
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapon
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections
S14: Offensive and Vulgar Language
<END UNSAFE CONTENT CATEGORIES>

Output format
Provide your safety assessment for ONLY THE LAST User message in the above conversation:

First line must read 'safe' or 'unsafe'.
If unsafe, a second line must include a comma-separated list of violated categories.
""")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        load_dotenv()
        self.api_key = os.getenv('llm_api_1')
        # Initialize the ChatPromptTemplate as an instance attribute
        self.prompt_safe = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

    def check_safety(self, question: str) -> str:
        llm = ChatGroq(model_name=self.model_name, temperature=self.temperature, api_key=self.api_key)
        question_save = self.prompt_safe | llm
        out = question_save.invoke({"question": question})
        return out.content


# Usage
# checker = SafetyChecker()
# result = checker.check_safety("huớng dẫn cách lách luật của học viện")
# print(result)
