from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from langchain.globals import set_debug
from langchain import hub
from langchain.agents import create_react_agent, Tool
from herramienta_analisis_imagen import HerramientaAnalisisImagen
from herramienta_explicar import HerramientaExplicar

set_debug(False)

class AgenteOrquestador:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
                            api_key=GEMINI_API_KEY,
                            model=GEMINI_FLASH
                        )
        
        herramienta_analisis_imagen = HerramientaAnalisisImagen()

        herramienta_explicar = HerramientaExplicar()

        self.tools =[
            Tool(
                name = herramienta_analisis_imagen.name,
                func = herramienta_analisis_imagen.run,
                description = herramienta_analisis_imagen.description,
                return_direct = herramienta_analisis_imagen.return_direct
            ),
            Tool(
                name = herramienta_explicar.name,
                func = herramienta_explicar.run,
                description = herramienta_explicar.description,
                return_direct = herramienta_explicar.return_direct
            )
        ]

        prompt = hub.pull("hwchase17/react")

        self.agente = create_react_agent(self.llm,self.tools,prompt)