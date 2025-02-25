# backend_chatIA_Uaudi
Development with python apikey GPT

# Arquitectura del Proyecto

Este proyecto sigue el principio de la Arquitectura Hexagonal (también conocida como Arquitectura Limpia), organizando los diferentes componentes para maximizar la separación de responsabilidades y la facilidad de mantenimiento.

## Estructura de Directorios

project-root/
│
├── src/
│   ├── application/
│   │   ├── use_cases/          # Casos de uso - Contiene la lógica de negocio
│   │   └── __init__.py
│   │
│   ├── domain/
│   │   ├── entities/           # Entidades de Dominio - Representan los objetos principales del negocio
│   │   └── __init__.py
│   │
│   ├── infrastructure/
│   │   ├── adapters/
│   │   │   ├── input/          # Adaptadores de entrada - Controladores para la gestión de las entradas
│   │   │   │   ├── controllers/   # Controladores - Gestionan la comunicación entre la aplicación y el mundo exterior
│   │   │   │   └── __init__.py
│   │   │   ├── output/         # Adaptadores de salida - Comunicación con infraestructura externa (bases de datos, APIs, etc.)
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── ports/
│   │   ├── input/              # Puertos de entrada - Interfaces que describen cómo la aplicación interactúa con la capa de infraestructura
│   │   ├── output/             # Puertos de salida - Interfaces para los adaptadores de salida
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── tests/                      # Pruebas - Contiene las pruebas unitarias y de integración
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   └── __init__.py




### Descripción de las Carpetas

- **src/**: Contiene todo el código fuente del proyecto.
  - **application/**: Contiene los casos de uso, que son las acciones específicas que puede realizar la aplicación, manteniendo la lógica de negocio pura y separada de las dependencias.
    - **use_cases/**: Casos de uso que describen acciones del negocio.
  - **domain/**: Contiene las entidades de dominio, que representan los conceptos principales del negocio.
    - **entities/**: Clases que representan los objetos del dominio, tales como Cliente, Pedido, etc.
  - **infrastructure/**: Contiene los adaptadores y la implementación técnica específica.
    - **adapters/input/controllers/**: Controladores que manejan la interacción con los usuarios u otras aplicaciones (por ejemplo, APIs REST).
    - **adapters/output/**: Implementaciones específicas para interactuar con servicios externos, como bases de datos o APIs.
  - **ports/**: Define las interfaces (puertos) de entrada y salida.
    - **input/**: Interfaces que conectan los controladores (entrada) con la aplicación.
    - **output/**: Interfaces que conectan la lógica de negocio con adaptadores externos (salida).
- **tests/**: Carpeta para las pruebas, incluyendo las pruebas unitarias y de integración que validan el funcionamiento de la aplicación.

Esta estructura ayuda a que la aplicación sea fácilmente extensible y testeable, con una clara separación de la lógica del negocio de la infraestructura. Cada módulo cumple una función específica dentro de la arquitectura, lo que facilita el mantenimiento y la evolución del software.

## Dependencias

pip install "fastapi[standard]" 
pip install -qU langchain-ollama   
pip install -qU langchain-openai

pip install langchain-core langgraph>0.2.27
pip install -qU langchain-openai

%pip install -qU pypdf
pip install langchain-community
pip install -U langchain_openai
pip install -qU langchain-ollama



Test
uvicorn main:router --reload
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload


Postman 
curl --location 'http://127.0.0.1:8000/chat' \
--header 'Content-Type: application/json' \
--data '{
    "input_language": "English",
    "output_language": "Spanish",
    "question": "I love programming."
}'