from adapters.secondary.langchain.chat_ollama_adapter import ChatOllamaAdapter

class AskQuestionUseCase:
    def __init__(self):
        self.chat_adapter = ChatOllamaAdapter()

    def execute(self, input_language: str, output_language: str, question: str) -> str:
        # Usar el adaptador para hacer la pregunta con los parámetros adecuados
        response = self.chat_adapter.ask(input_language, output_language, question)
        return response
