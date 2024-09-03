from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import os 
from sentence_transformers import SentenceTransformer
from semantic_router import SemanticRouter, Route
from semantic_router.samples import rag_sample, chitchatSample
load_dotenv()
import time
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0,api_key= os.getenv('llm_api_1')) ## Replace to real LLMs (Cohere / Groq / OpenAI)


MTA_ROUTE_NAME = 'mta'
CHITCHAT_ROUTE_NAME = 'chitchat'

embedding = SentenceTransformer('keepitreal/vietnamese-sbert')



mtaRoute = Route(name=MTA_ROUTE_NAME, samples=rag_sample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
router = SemanticRouter(embedding, routes=[mtaRoute, chitchatRoute])


while True:
    
    query = input('Nhập câu hỏi: ')
    start_time = time.time()
    best_route = router.guide(query)
    end_time = time.time()
    print(f"The best route for the query is: {best_route[1]}")
    print(f'time using:{end_time-start_time}')