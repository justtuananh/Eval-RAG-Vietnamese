import json
import os
from dotenv import load_dotenv
import json
from langchain_wrappers import  SambaNovaFastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt
current_dir = os.getcwd()
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
load_dotenv(os.path.join(utils_dir, '.env'), override=True)

llm = SambaNovaFastAPI(model='llama3-70b') #llama3-8b , #llama3-405b

prompt_test = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                ("human", "{question}"),
            ]
        )


system_message = "You are a helpful AI assistant, answer user question"
question_message = "hello"

formatted_prompt = prompt_test.format(system=system_message, question=question_message)


print(llm.invoke(formatted_prompt))