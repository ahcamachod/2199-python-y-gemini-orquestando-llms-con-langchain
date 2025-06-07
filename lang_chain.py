from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from my_helper import encode_image


llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

respuesta = llm.invoke("Cuáles canales de youtube me recomiendas para saber más sobre teléfonos inteligentes?")

print(f"Gemini: ",respuesta.content)

#Con Cohere

llm = ChatCohere(cohere_api_key=COHERE_API_KEY)

respuesta = [HumanMessage(content="Cuáles canales de youtube me recomiendas para saber más sobre teléfonos inteligentes?")]

print(f"\n\nCohere: ",llm.invoke(respuesta).content)

imagen = encode_image('datos/ejemplo_grafico.jpg')

pregunta = 'Describe la imagen: '

mensaje = HumanMessage(
    content = [
        {
            "type":"text",
            "text": pregunta            
        },
        {
            "type":"image_url",
            "image_url": f"data:image/jpeg;base64,{imagen}"
        }
    ]
)

respuesta = llm.invoke([mensaje])

print(respuesta)