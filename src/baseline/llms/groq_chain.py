import os
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
        self.api_usage = {key: {'requests': 0, 'tokens': 0} for key in api_keys}
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
        
        if usage['requests'] >= limits['requests_per_minute']:
            return False
        if usage['tokens'] >= limits['tokens_per_minute']:
            return False
        return True
    
    def query_llm(self, prompt):
        """
        Send a prompt to the LLM and handle exceptions by switching API keys if necessary.
        """
        for attempt in range(len(self.api_keys)):
            current_api_key = self.api_keys[self.current_key_index]
            if not self.check_limits(current_api_key):
                print(f"API key {current_api_key} has reached its limit, switching to the next key...")
                self.switch_api_key()
                continue
            try:
                response = self.llm.invoke(prompt)
                token_usage = response.response_metadata['token_usage']
                self.api_usage[current_api_key]['requests'] += 1
                self.api_usage[current_api_key]['tokens'] += token_usage['total_tokens']
                return response.content
            except Exception as e:
                print(f"Error: {e}, switching to the next API key...")
                self.api_usage[current_api_key]['requests'] = 0
                self.api_usage[current_api_key]['tokens'] = 0
                self.switch_api_key()
                continue
        print("All API keys have failed.")
if __name__ == "__main__" :
    llm_router = Groq_Routing()
    response = llm_router.query_llm("What is the weather today?")
    if response:
        print("LLM Response:", response)
    else:
        print("Failed to get a response from LLM.")