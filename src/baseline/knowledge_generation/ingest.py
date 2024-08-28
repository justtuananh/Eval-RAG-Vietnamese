from typing import Optional, List, Union
from langchain.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
import os
import torch

from baseline.docs_process.docs_split import split_documents

def load_embeddings(
    langchain_docs: List[LangchainDocument],
    chunk_size: int,
    eval_repo_dir : str,
    embedding_model_name: Optional[str] = "hiieu/halong_embedding",
    vectorstore_type: Optional[str] = "FAISS"
) -> Union[FAISS, Chroma]:
    """
    Creates a vector store index (FAISS or Chroma) from the given embedding model and documents.
    Loads the index directly if it already exists.

    Args:
        langchain_docs: list of documents
        chunk_size: size of the chunks to split the documents into
        embedding_model_name: name of the embedding model to use
        vectorstore_type: type of vector store to use, either "FAISS" or "Chroma"

    Returns:
        FAISS or Chroma index
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')},
        encode_kwargs={
            "normalize_embeddings": True  # Set True to compute cosine similarity
        },
    )

    index_name = (
        f"index_chunk:{chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}_{vectorstore_type}"
    )
    index_folder_path = f"{eval_repo_dir}/data/indexes/{index_name}/"

    if os.path.isdir(index_folder_path):
        if vectorstore_type == "FAISS":
            return FAISS.load_local(
                index_folder_path,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
        elif vectorstore_type == "Chroma":
            return Chroma( embedding_function = embedding_model,persist_directory =index_folder_path )
            #return Chroma.load_local(index_folder_path, embedding_model)

    else:
        print("Index not found, generating it...")
        docs_processed = split_documents(
            chunk_size,
            langchain_docs,
            embedding_model_name,
        )

        if vectorstore_type == "FAISS":
            knowledge_index = FAISS.from_documents(
                docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            knowledge_index.save_local(index_folder_path)
        elif vectorstore_type == "Chroma":
            knowledge_index = Chroma.from_documents(docs_processed, embedding_model, persist_directory=index_folder_path)
            knowledge_index.persist()

        return knowledge_index