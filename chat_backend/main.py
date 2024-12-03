from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from adapters.primary.api.chat_controller import router

app = FastAPI()

# Configurar el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5501"],  # Permitir solicitudes desde el frontend en 5500
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Manejo global de solicitudes OPTIONS (preflight)
@app.options("/{path:path}")
async def preflight_handler(path: str):
    """
    Responder a solicitudes preflight (OPTIONS) necesarias para CORS.
    """
    return JSONResponse(content={}, status_code=200)

# Incluir el router para los endpoints del chat
app.include_router(router)

# Prueba de endpoint simple para verificar conectividad
@app.get("/")
async def root():
    return {"message": "El servidor está funcionando correctamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
