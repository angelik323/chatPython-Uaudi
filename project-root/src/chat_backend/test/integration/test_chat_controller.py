from fastapi.testclient import TestClient
from adapters.secondary.langchain.chat_ollama_adapter import ChatOllamaAdapter
from main import router  # Suponiendo que tu archivo principal tenga el nombre main.py

# Utilizamos el TestClient de FastAPI
client = TestClient(router)


def test_ask_question():
    # Definir la entrada para la solicitud
    data = {
        "input_language": "English",
        "output_language": "German",
        "question": "I love programming."
    }

    # Realizar la solicitud POST al endpoint "/chat"
    response = client.post("/chat", json=data)

    # Mostrar la respuesta para verificar la salida
    print(response.json())
    assert response.status_code == 200
    assert "response" in response.json()
