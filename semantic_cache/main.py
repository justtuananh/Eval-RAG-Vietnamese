import time
import random
from Cache import Cache as gpt_cache

# Initialize the cache
cache = gpt_cache(json_file='./cache_file.json', embedding= "dangvantuan/vietnamese-embedding")

# List of sample answers
sample_answers = [
    "Năm nay học viện chỉ tiêu nhiều lắm",
    "Học viện không thay đổi chỉ tiêu",
    "Chỉ tiêu tuyển sinh năm nay giảm",
    "Chỉ tiêu ổn định như năm ngoái",
    "Có tăng chút ít so với năm trước"
]

def get_random_answer():
    return random.choice(sample_answers)

def main():
    while True:
        # Get user input for the question
        temp_q = input("Nhập câu hỏi: ")
        
        start_time = time.time()
        
        result = cache.cached_hit(temp_q)
        
        if isinstance(result, str):
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Cache hit! Time taken: {elapsed_time:.3f} seconds")
            print(f"Answer: {result}")
        else:
            # Cache miss
            embedding, flag_cache_missed = result
            if flag_cache_missed:
                temp_answer = get_random_answer()  # Get a random answer
                cached_answer = cache.cache_miss(temp_q, embedding, temp_answer)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Cache miss. Time taken: {elapsed_time:.3f} seconds")
                print(f"New answer added to cache: {cached_answer}")

if __name__ == "__main__":
    main()
