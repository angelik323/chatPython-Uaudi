from adapters.secondary.langchain.chat_adapter_factory import ChatAdapterFactory

from typing import Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define the graph workflow for the conversation
workflow = StateGraph(state_schema=MessagesState)

# Define a class to store user context
class UserContext:
    def __init__(self):
        self.context = {}

    def update_context(self, session_id: str, key: str, value: str):
        if session_id not in self.context:
            self.context[session_id] = {}
        self.context[session_id][key] = value

    def get_context(self, session_id: str, key: str):
        return self.context.get(session_id, {}).get(key, None)

# Create a global user context object
user_context = UserContext()

# Define the function that calls the model
def call_model(state: MessagesState):
    # Get adapter for the model
    model_name = state.get("model", "ollama")
    chat_adapter = ChatAdapterFactory().get_adapter(model_name)
    # Invoke the model with the messages in the state
    response = chat_adapter.invoke(state["messages"])
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory persistence to the workflow
memory = MemorySaver()
langgraph_app = workflow.compile(checkpointer=memory)

# Modify use case to interact with different LLMs
class AskQuestionUseCase:
    def execute(self, session_id: str, query: str, model: str, additional_params: Optional[Dict[str, str]] = None) -> str:
        # Update user context if the query provides user information
        if "soy" in query.lower():
            name = query.split("soy")[-1].strip().split(",")[0].strip()
            user_context.update_context(session_id, "name", name)

        # Retrieve user context if available
        user_name = user_context.get_context(session_id, "name")

        # Prepare input message based on query context
        if additional_params and "query_type" in additional_params:
            query_type = additional_params["query_type"].lower()
            if query_type == "company_info":
                message_content = f"You are an assistant for Audifarma. You answer questions about company details, such as office hours, contact numbers, and services offered.\n"
                if user_name:
                    message_content += f"Hello {user_name}! "
                message_content += f"Human: {query}"
                message = HumanMessage(content=message_content)
            elif query_type == "translation":
                input_language = additional_params.get("input_language", "English")
                output_language = additional_params.get("output_language", "Spanish")
                message = HumanMessage(content=f"You are a helpful assistant that translates {input_language} to {output_language}.\nHuman: {query}")
            else:
                message = HumanMessage(content=f"You are a helpful assistant.\nHuman: {query}")
        else:
            message_content = f"You are a helpful assistant.\n"
            if user_name:
                message_content += f"Hello {user_name}! "
            message_content += f"Human: {query}"
            message = HumanMessage(content=message_content)

        # Invoke the LangGraph app with the current message and session configuration
        input_messages = {"messages": [message], "model": model}
        config = {"configurable": {"thread_id": session_id}}
        output = langgraph_app.invoke(input_messages, config)

        # Return the last AI message
        return output["messages"][-1].content