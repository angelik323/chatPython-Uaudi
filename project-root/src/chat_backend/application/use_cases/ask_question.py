from adapters.secondary.langchain.chat_adapter_factory import ChatAdapterFactory
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Optional

# Modify use case to interact with different LLMs
class AskQuestionUseCase:
    def __init__(self):
        self.chat_factory = ChatAdapterFactory()

    def execute(self, query: str, model: str, additional_params: Optional[Dict[str, str]]) -> str:
        # Get the correct adapter based on the requested model
        chat_adapter = self.chat_factory.get_adapter(model)

        # Determine which prompt to use based on query context
        if additional_params and "query_type" in additional_params:
            query_type = additional_params["query_type"].lower()
            if query_type == "company_info":
                prompt = ChatPromptTemplate.from_template(
                    "You are an assistant for Audifarma. You answer questions about company details, such as office hours, contact numbers, and services offered.\nHuman: {query}"
                ).format(query=query)
            elif query_type == "translation":
                input_language = additional_params.get("input_language", "English")
                output_language = additional_params.get("output_language", "Spanish")
                prompt = ChatPromptTemplate.from_template(
                    "You are a helpful assistant that translates {input_language} to {output_language}.\nHuman: {query}"
                ).format(input_language=input_language, output_language=output_language, query=query)
            else:
                prompt = ChatPromptTemplate.from_template(
                    "You are a helpful assistant.\nHuman: {query}"
                ).format(query=query)
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant.\nHuman: {query}"
            ).format(query=query)

        # Invoke the model with the prompt
        response = chat_adapter.invoke(prompt)

        return response
