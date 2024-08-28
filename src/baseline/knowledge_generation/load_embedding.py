from knowledge_generation.ingest import load_embeddings
from docs_process import docs_split
from docs_process.dataset_load import load_eval_data




if __name__ == "__main__" :
    dataset  = load_eval_data("legal")
    doc_splited = docs_split.split_documents(768, load_eval_data("legal"), tokenizer_name = "", is_tiktoken = True)
    knowledge_index = load_embeddings(
        doc_splited,
        512,
        eval_repo_dir= "../../../vectorstore",
        embedding_model_name="hiieu/halong_embedding",
        vectorstore_type="Chroma",
    )
    