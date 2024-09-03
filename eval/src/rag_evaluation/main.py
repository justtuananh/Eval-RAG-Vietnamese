from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv()
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0,api_key= os.getenv('GROQ_API_KEY1')) ## Replace to real LLMs (Cohere / Groq / OpenAI)


