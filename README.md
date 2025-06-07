# Título del proyecto

2199 - Python y Gemini: Orquestando LLMs con LangChain

## 🔨 Funcionalidades del proyecto

En este proyecto, utilizaremos LangChain como framework principal para orquestar una solución integrada de análisis y organización de imágenes enriquecidas con anotaciones inteligentes. LangChain será empleado debido a su capacidad para conectar y gestionar flujos complejos que combinan IA multimodal y modelos de lenguaje, lo que permite un desarrollo más modular y escalable.

![](img/amostra.gif)

## ✔️ Técnicas y tecnologías utilizadas

Las técnicas y tecnologías utilizadas son:

- Programación en Python  
- Uso de la API Gemini  
- Uso del framework LangChain  
- Cadenas simples  
- Agente orquestador  
- Agente como herramientas  

## 🛠️ Abrir y ejecutar el proyecto

Después de descargar el proyecto, puedes abrirlo con Visual Studio Code. A continuación, es necesario preparar tu entorno. Para ello:

### venv en Windows:

```bash
python -m venv .venv-gemini-3
.\.venv-gemini-3\Scripts\activate
````

### venv en Mac/Linux:

```bash
python3 -m venv .venv-gemini-3
source .venv-gemini-3/bin/activate
```

Después, instala los paquetes utilizando:

```bash
pip install -r requirements.txt
```

## 🔑 Generar API\_KEY y asociarla al archivo .env

```python
GEMINI_API_KEY = "TU_API_KEY_AQUÍ"
COHERE_API_KEY = "TU_API_KEY_AQUÍ"
```
