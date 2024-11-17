from fastapi import APIRouter
from pydantic import BaseModel
from application.use_cases.ask_question import AskQuestionUseCase

# Definir un modelo para la entrada del cuerpo de la solicitud
class QuestionRequest(BaseModel):
    input_language: str
    output_language: str
    question: str

router = APIRouter()

@router.post("/chat")
async def ask_question(request: QuestionRequest):
    use_case = AskQuestionUseCase()
    response = use_case.execute(request.input_language, request.output_language, request.question)
    return {"response": response}
