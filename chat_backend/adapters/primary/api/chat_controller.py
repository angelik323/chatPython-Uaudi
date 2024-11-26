from fastapi import APIRouter
from pydantic import BaseModel

from typing import Dict, Optional
import os

from application.use_cases.ask_question import AskQuestionUseCase


class QuestionRequest(BaseModel):
    session_id: str  # Unique identifier for the session
    query: str  # The actual content of the query
    model: str = "ollama"  # Optional parameter to specify the LLM
    additional_params: Optional[Dict[str, str]] = None  # Additional parameters for context (e.g., input_language, output_language)


router = APIRouter()

# Define a mapping of models to their max token limits
MODEL_MAX_TOKENS = {
    "ollama": 2048,
    "gpt-3.5": 4096,
    "gpt-4": 8192,
}

@router.post("/chat")
async def ask_question(request: QuestionRequest):
    try:
        use_trim = os.getenv("USE_TRIM", "false").lower() == "true"
        max_tokens = MODEL_MAX_TOKENS.get(request.model, 2048)  # Set max_tokens based on the model specified
        use_case = AskQuestionUseCase()
        response = use_case.execute(
            session_id=request.session_id,
            query=request.query,
            model=request.model,
            additional_params=request.additional_params,
            use_trim=use_trim,
            max_tokens=max_tokens
        )
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
