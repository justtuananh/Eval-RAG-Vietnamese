import os
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

llm_api = load_dotenv(".env")
api_keys = [os.getenv(f"llm_api_{i}") for i in range(1, 6)]  # Adjust range according to the number of API keys available

rate_limits = {
    'llama3-70b-8192': {'requests_per_minute': 30, 'tokens_per_minute': 6000, 'requests_per_day': 14400},
}

class Groq_Routing:
    def __init__(self, model_name="llama3-70b-8192", temperature=0, request_timeout=120, max_retries=50):
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.api_keys = api_keys
        self.current_key_index = 0
        self.api_usage = {key: {'requests': 0, 'tokens': 0, 'last_used': time.time()} for key in api_keys}
        self.llm = self.initialize_llm(self.api_keys[self.current_key_index])
    
    def initialize_llm(self, api_key):
        """
        Initialize the LLM model with the given API key.
        """
        return ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries
        )

    def switch_api_key(self):
        """
        Switch to the next available API key.
        """
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.llm = self.initialize_llm(self.api_keys[self.current_key_index])

    def check_limits(self, api_key):
        """
        Check if the current API key has reached its limits.
        """
        usage = self.api_usage[api_key]
        limits = rate_limits[self.model_name]
        current_time = time.time()
        time_since_last_use = current_time - usage['last_used']

        if time_since_last_use >= 60:
            usage['requests'] = 0
            usage['tokens'] = 0
            usage['last_used'] = current_time
        
        if usage['requests'] >= limits['requests_per_minute']:
            return False
        if usage['tokens'] >= limits['tokens_per_minute']:
            return False
        return True
    def wait_for_reset(self, api_key):
        """
        Wait until the API key's usage resets based on rate limits.
        """
        current_time = time()
        last_used_time = self.api_usage[api_key]['last_used']
        if last_used_time is None:
            wait_time = 60  # Default wait time if `last_used` is not set
        else:
            elapsed_time = current_time - last_used_time
            # Calculate remaining time until the API key can be used again
            wait_time = max(0, 60 - elapsed_time)  # Assuming 60 seconds for rate limit reset

        print(f"Waiting {wait_time:.2f} seconds for API key {api_key} to reset...")
        time.sleep(wait_time)
        # Reset the usage after waiting
        self.api_usage[api_key]['requests'] = 0
        self.api_usage[api_key]['tokens'] = 0
            
    def query_llm(self, prompt):
        """
        Send a prompt to the LLM and handle exceptions by switching API keys if necessary.
        """
        for attempt in range(len(self.api_keys)):
            current_api_key = self.api_keys[self.current_key_index]
            if not self.check_limits(current_api_key):
                print(f"API key {current_api_key} has reached its limit, waiting for reset...")
                self.switch_api_key()
                continue

            try:
                response = self.llm.invoke(prompt)
                token_usage = response.response_metadata['token_usage']
                self.api_usage[current_api_key]['requests'] += 1
                self.api_usage[current_api_key]['tokens'] += token_usage['total_tokens']
                self.api_usage[current_api_key]['last_used'] = time.time()  # Update the last used time
                return response.content
            except Exception as e:
                print(f"Error: {e}, switching to the next API key...")
                self.switch_api_key()
                continue
        current_api_key = self.api_keys[self.current_key_index] 
        print(f"All API keys have been used. Waiting for API key {current_api_key} to reset...")
        self.wait_for_reset(current_api_key) 
        return self.query_llm(prompt)
if __name__ == "__main__" :
    llm_router = Groq_Routing()
    response = llm_router.query_llm("What is the weather today?")
    if response:
        print("LLM Response:", response)
    else:
        print("Failed to get a response from LLM.")