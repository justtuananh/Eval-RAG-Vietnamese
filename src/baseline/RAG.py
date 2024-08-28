from typing import List, Optional, Tuple
from langchain_core.vectorstores import VectorStore
from langchain.docstore.document import Document as LangchainDocument
from prompts import accurate_rag_prompt



def answer_with_rag(
    question: str,
    llm,
    knowledge_index: VectorStore,
    model_type : Optional[str] = None,
    reranker: Optional[str] = None,
    num_retrieved_docs: int = 3,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]


    context = "\nExtracted docs:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = accurate_rag_prompt.format(question=question, context=context)

    # Redact an answer
    if model_type == "type something else like HF, OPENAI" :
      pass
    else :
      answer = llm.invoke(final_prompt).content

    return answer, relevant_docs
