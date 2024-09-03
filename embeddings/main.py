from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm
import os

HF_EMBEDDING = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

def load_and_chunk_data(data_path):
    docs = []
    # Load all .txt files from the specified folder
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_path, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            docs.extend(loader.load())

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    chunk_size = 512
    chunk_overlap = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = []

    for doc in docs:

        md_header_splits = markdown_splitter.split_text(doc.page_content)
        chunked_docs.extend(text_splitter.split_documents(md_header_splits))

    return chunked_docs


def main():
    data_path = '/home/justtuananh/AI4TUAN/DOAN2024/eval_rag_vietnamese/thongtintuyensinh'


    print("Loading and chunking data...")
    chunked_data = load_and_chunk_data(data_path)

    # Initialize Qdrant vector store
    vectorstore = Qdrant(
        embeddings=HF_EMBEDDING,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="tintuyensinh",
    )

    print("Embedding documents into the vector store...")
    for doc in tqdm(chunked_data, desc="Embedding Documents"):
        vectorstore.add_document(doc)

    print("Embedding completed.")


if __name__ == "__main__":
    main()
