from adapters.secondary.langchain.chat_adapter_factory import ChatAdapterFactory
from langchain_core.prompts import ChatPromptTemplate

class AskQuestionUseCase:
    def __init__(self):
        self.chat_factory = ChatAdapterFactory()

    def execute(self, input_language: str, output_language: str, question: str, model: str) -> str:
        # Get the correct adapter based on the requested model
        chat_adapter = self.chat_factory.get_adapter(model)

        # Prepare the prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ])

        # Create the chain and invoke the model
        chain = prompt | chat_adapter
        response = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "input": question,
            }
        )

        return response