import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

llm_api = load_dotenv(".env")
api_keys = [os.getenv(f"llm_api_{i}") for i in range(1, 5)]  # Adjust range according to the number of API keys available
print(api_keys)
class Groq_Routing:
    def __init__(self, model_name="llama3-70b-8192", temperature=0, request_timeout=120, max_retries=50):
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.api_keys = api_keys
        self.current_key_index = 0
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

    def query_llm(self, prompt):
        """
        Send a prompt to the LLM and handle exceptions by switching API keys if necessary.
        """
        for attempt in range(len(self.api_keys)):
            try:
                response = self.llm.invoke(prompt).content
                return response  # Return the successful response
            except Exception as e:
                print(f"Error: {e}, switching to the next API key...")
                self.switch_api_key()
        
        print("All API keys have failed.")
        return None
if __name__ == "__main__" :
    llm_router = Groq_Routing()
    response = llm_router.query_llm("What is the weather today?")
    if response:
        print("LLM Response:", response)
    else:
        print("Failed to get a response from LLM.")