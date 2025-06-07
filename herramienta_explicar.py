from langchain.tools import BaseTool
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from my_keys import COHERE_API_KEY
import ast

class HerramientaExplicar(BaseTool):
    name:str = "HerramientaExplicar"
    description:str = """
                        Utiliza esta herramienta siempre que sea solicitada la explicación
                        de un contenido a las personas.

                        # ENTRADA REQUERIDA
                        - 'tema' (str) : Tema principal informado en la pregunta del usuario.
                      """
    return_direct : bool = True

    def _run(self, accion):
        accion = ast.literal_eval(accion)
        tema_parametro = accion.get("tema","") # Si no encuentra tema, devuelve una string vacía.
        llm = ChatCohere(cohere_api_key=COHERE_API_KEY)

        template_respuesta = PromptTemplate(
                                    template="""
                                            Asume el papel se un profesor con aspectos de didáctica del usuario.

                                            1. Elabora una explicación sobre el tema {tema} que sea de fácil 
                                            comprensión para estudiantes de secundaria.
                                            2. Utiliza ejemplos cotidianos para volver la explicación más sencilla.
                                            3. En caso de que surja algún recurso para apoyar la explicación, recuerde
                                            el escenario del contexto colombiano.
                                            4. En caso que presentes algún script de código, sé didáctico y utiliza Python.

                                            Tema pregunta: {tema}
                                            """,
                                            input_variables=["tema"]
                                        )
        cadena = template_respuesta | llm | StrOutputParser()

        respuesta = cadena.invoke({"tema" : tema_parametro})

        return respuesta