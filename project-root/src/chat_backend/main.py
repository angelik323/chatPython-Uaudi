from fastapi import FastAPI
from adapters.primary.api.chat_controller import router

app = FastAPI()

# Incluir el router para los endpoints del chat
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
