import numpy as np
from encoder_wrapper.base import BaseEmbedding



from sentence_transformers import SentenceTransformer  # pylint: disable=C0413


class SBERT(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models of Sentence Transformers.

    :param model: model name, defaults to 'all-MiniLM-L6-v2'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import SBERT

            test_sentence = 'Hello, world.'
            encoder = SBERT('all-MiniLM-L6-v2')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.model.eval()
        self.__dimension = None

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        emb = self.model.encode(data)
        _, dim = emb.shape
        emb =emb / np.linalg.norm(emb)
        if not self.__dimension:
            self.__dimension = dim
        return np.array(emb).astype("float32")

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        if not self.__dimension:
           embd = self.model.encode(["foo"])
           _, self.__dimension = embd.shape
        return self.__dimension
    

if __name__ == "__main__" :
    embd = SBERT()
    string = "This is a test str"
    embd = embd.to_embeddings(string)
    print(embd)