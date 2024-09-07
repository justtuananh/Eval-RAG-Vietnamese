import json
from encoder_wrapper import sentence_bert
import faiss

def init_cache(embedding_model: str = "all-MiniLM-L6-v2"):
    """Initializes the cache with a Faiss index and an SBERT model.

    Args:
    embedding_model (str): The name of the SBERT model to use.

    Returns:
    tuple: (index, encoder) where
        - index is a Faiss index for storing embeddings.
        - encoder is an SBERT model instance.
    """

    encoder = sentence_bert(embedding_model)
    dimension = encoder.dimension
    print(dimension)
    index = faiss.IndexFlatL2(dimension)
    if index.is_trained:
        print('Index initialized and ready for use')

    return index, encoder


def retrieve_cache(json_file):
  try:
    with open(json_file, 'r') as file:
      cache = json.load(file)
  except FileNotFoundError:
      cache = {'questions': [], 'answers': []}

  return cache



def store_cache(json_file, cache):
  with open(json_file, 'w', encoding = 'utf-8') as file:
    json.dump(cache, file)