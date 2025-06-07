from langchain.agents import AgentExecutor
from orquestador import AgenteOrquestador

def main():
    agente = AgenteOrquestador()
    ejecutor = AgentExecutor(
                    agent = agente.agente,
                    tools = agente.tools,
                    verbose = True
                )
    
    pregunta = "Quiero que me expliques cómo funcionan los desvíos condicionales"

    respuesta = ejecutor.invoke({"input" : pregunta})

    print(respuesta)

if __name__=="__main__":
    main()