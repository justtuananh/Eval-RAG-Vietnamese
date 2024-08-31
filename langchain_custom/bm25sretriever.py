from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
import bm25s
from itertools import chain

def default_preprocessing_func(text: str) -> List[str]:
    token_corpus = bm25s.tokenize(texts=text, stopwords = "vi", return_ids= False , show_progress=False)
    return token_corpus


class BM25SRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """
    vectorizer: Any
    """ BM25S vectorizer."""
    docs: List[Document] = Field(repr=False)
    """List of documents to retrieve from."""
    k: int = 4 
    """Number of top results to return"""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""
    save_directory : Optional[str] = None
    """ Directory for saving BM25S index."""
    class Config:
        arbitrary_types_allowed = True
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        save_directory : Optional[str] = save_directory,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25s_params: Parameters to pass to the BM25s vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25SRetriever instance.
        """
        try:
            from bm25s import BM25
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install "
                "bm25s`."
            )
        bm25_params = bm25_params or {}
        if save_directory and Path(save_directory).exists():
            try:
                vectorizer = BM25.load(save_directory)
            except Exception as e:
                print(f"Failed to load BM25 index from {save_directory}: {e}")
                print("Proceeding with indexing from scratch.")
                texts_processed = preprocess_func(texts)
                vectorizer = BM25(**bm25_params)
                vectorizer.index(texts_processed)
                if save_directory:
                    vectorizer.save(save_directory)
        
        else:
            texts_processed = preprocess_func(texts)
            vectorizer = BM25(**bm25_params)
            vectorizer.index(texts_processed)
            if save_directory:
                vectorizer.save(save_directory)

        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, save_directory=save_directory, **kwargs
        )
        
    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs, scores = self.vectorizer.retrieve(processed_query, self.docs, k = self.k)
        return [return_docs[0, i] for i in range(return_docs.shape[1])]
