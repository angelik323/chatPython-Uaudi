from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define adapters for different LLMs
class ChatAdapterFactory:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY no está configurado.")
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        self.adapters = {
            "ollama": ChatOllama(
                model="llama3.2",
                temperature=0,
            ),
            "openai": ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key
            )
        }

    def get_adapter(self, model_name: str):
        return self.adapters.get(model_name, self.adapters["ollama"])