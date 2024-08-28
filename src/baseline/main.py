import json
import os
from typing import Optional
from langchain_core.vectorstores import VectorStore
import datasets
from tqdm import tqdm
from RAG import answer_with_rag
from llms.groq_chain import llm as generator
from docs_process import load_eval_data
import configfile
from load_embedding import knowledge_index

def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[str] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r", encoding= "utf-8") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question, llm, knowledge_index, reranker=reranker
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)
            
if __name__ == "__main__" :
    if not os.path.exists(f"{configfile.eval_repo_dir}/output"):
        os.mkdir(f"{configfile.eval_repo_dir}/output")
    
    settings_name = f"chunk:{configfile.chunk_size}_embeddings:{configfile.embeddings.replace('/', '~')}_rerank:no_reader-model:Groq-Llama3-70b-8192_legalRAG"
    output_file_name = f"{configfile.eval_repo_dir}/output/rag_{settings_name}.json"

    run_rag_tests(load_eval_data(configfile.data_name) , llm= generator, knowledge_index= knowledge_index , output_file= output_file_name , verbose= True, test_settings= settings_name)