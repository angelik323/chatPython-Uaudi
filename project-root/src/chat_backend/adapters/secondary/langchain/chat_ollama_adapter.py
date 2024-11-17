from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class ChatOllamaAdapter:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            # other params...
        )

    def ask(self, input_language: str, output_language: str, user_input: str) -> str:
        # Construir el template del prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that translates {input_language} to {output_language}.",
                ),
                ("human", "{input}"),
            ]
        )

        # Crear la cadena usando el prompt y el modelo LLM
        chain = prompt | self.llm

        # Invocar el modelo
        response = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "input": user_input,
            }
        )

        return response
