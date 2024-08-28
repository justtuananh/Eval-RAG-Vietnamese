import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

llm_api = load_dotenv("../.env")
llm_api_key = os.getenv("llm_api")
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0,api_key= llm_api, request_timeout = 120 , max_retries = 20) ## Replace to real LLMs (Cohere / Groq / OpenAI)
