import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
qdrant_url = os.getenv('URL_QDRANT')
qdrant_api = os.getenv('API_QDRANT')
client = QdrantClient(
   qdrant_url,
    api_key=qdrant_api
)
from BM25 import BM25SRetriever


from langchain_huggingface import HuggingFaceEmbeddings
HF_EMBEDDING = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
gthv = Qdrant(client, collection_name="gioithieuhocvien_db", embeddings= HF_EMBEDDING)

stsv = Qdrant(client, collection_name="sotaysinhvien_db", embeddings= HF_EMBEDDING)

ttts = Qdrant(client, collection_name="thongtintuyensinh_db", embeddings= HF_EMBEDDING)   

retriver = gthv.as_retriever(search_kwargs={'k': 6})

# res = retriver.get_relevant_documents("Cơ cấu tổ chức học viện")
# print(res)


