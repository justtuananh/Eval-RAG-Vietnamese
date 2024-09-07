import time
from adapter import init_cache, retrieve_cache, store_cache
class Cache:
    def __init__(self, embedding = "all-MiniLM-L6-v2" ,  json_file="cache_file.json", thresold=0.5, max_response=100, eviction_policy='FIFO'):
        """Initializes the semantic cache.

        Args:
        json_file (str): The name of the JSON file where the cache is stored.
        thresold (float): The threshold for the Euclidean distance to determine if a question is similar.
        max_response (int): The maximum number of responses the cache can store.
        eviction_policy (str): The policy for evicting items from the cache.
                                This can be any policy, but 'FIFO' (First In First Out) has been implemented for now.
                                If None, no eviction policy will be applied.
        """

        # Initialize Faiss index with Euclidean distance
        self.index, self.encoder = init_cache(embedding)

        # Set Euclidean distance threshold
        # a distance of 0 means identicals sentences
        # We only return from cache sentences under this thresold
        self.euclidean_threshold = thresold
        self.is_missed = True
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.eviction_policy = eviction_policy

    def evict(self):

        """Evicts an item from the cache based on the eviction policy."""
        if self.eviction_policy and len(self.cache["questions"]) > self.max_response:
            for _ in range((len(self.cache["questions"]) - self.max_response)):
                if self.eviction_policy == 'FIFO':
                    self.cache["questions"].pop(0)
                    self.cache["answers"].pop(0)
    def cached_hit(self, question: str) -> str:
            """Handles the cache hit logic by retrieving the answer from the cache.

            Args:
            question (str): The input question.
            embedding: The embedding of the question.

            Returns:
            str: The cached answer.
            """
            # Search for the nearest neighbor in the index
            embedding = self.encoder.to_embeddings([question])
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)
            print(D)
            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] / 100 <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    print('Answer recovered from Cache.')
                    print(f'Distance: {D[0][0]:.3f} (Threshold: {self.euclidean_threshold})')
                    print(f'Found in cache at row: {row_id} with score: {D[0][0]:.3f}')
                    self.is_missed =False
                    return self.cache['answers'][row_id]
            self.is_missed = True
            return embedding , self.is_missed

    
    def cache_miss(self, question: str, embedding , answer) -> str:
        """Handles the cache miss logic by querying the model and updating the cache.

        Args:
        question (str): The input question.
        embedding: The embedding of the question take from cache_hit if hit nothing
        answer (str) : The answer from LLMs
        Returns:
        Append to cache and return answer.
        """

        # Update the cache with the new question, embedding, and answer
        self.cache['questions'].append(question)
        self.cache['answers'].append(answer)

        print('Answer not found in cache, appending new answer.')
        print(f'Response: {answer}')

        # Add the new embedding to the index
        self.index.add(embedding)

        # Evict items if necessary
        self.evict()

        # Save the updated cache to the JSON file
        store_cache(self.json_file, self.cache)
        self.is_missed = False
        return answer
  
