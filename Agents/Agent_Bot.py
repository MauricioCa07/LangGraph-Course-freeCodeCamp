# The ones that i already known
from typing import TypedDict, List,Union
from langgraph.graph import StateGraph, START, END

# The new ones
from langchain_core.messages import HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values



load_dotenv()


llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    #temperature=0,
    #max_tokens=None,
    #timeout=None,
    #max_retries=2,
)


class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]



def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(response.content)

    return state



graph = StateGraph(AgentState)

graph.add_node("chat_node", process)


graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)


agent = graph.compile()


conversation_history = []

message = input("-> ")
while message:
    conversation_history.append(HumanMessage(content=message))
    result = agent.invoke({"messages":conversation_history})
    conversation_history = result["messages"]
    print(conversation_history)
    message = input("-> ")


