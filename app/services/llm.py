# app/services/llm.py
from langchain_openai import ChatOpenAI
import os

class LLMService:
    def __init__(self):
        self.client = ChatOpenAI(
            model_name="gpt-4o-mini",  # à adapter
            temperature=0.3, 
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate(self, prompt: str) -> str:
        return self.client(prompt)
