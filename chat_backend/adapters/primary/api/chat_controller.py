from fastapi import APIRouter
from pydantic import BaseModel

from typing import Dict, Optional

from application.use_cases.ask_question import AskQuestionUseCase


class QuestionRequest(BaseModel):
    session_id: str  # Unique identifier for the session
    query: str  # The actual content of the query
    model: str = "ollama"  # Optional parameter to specify the LLM
    additional_params: Optional[Dict[str, str]] = None  # Additional parameters for context (e.g., input_language, output_language)


router = APIRouter()

@router.post("/chat")
async def ask_question(request: QuestionRequest):
    try:
        use_case = AskQuestionUseCase()
        response = use_case.execute(request.session_id, request.query, request.model, request.additional_params)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
