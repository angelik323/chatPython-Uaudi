from fastapi import FastAPI
from adapters.primary.api.chat_controller import router

app = FastAPI()

# Include the router for the chat endpoints
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
