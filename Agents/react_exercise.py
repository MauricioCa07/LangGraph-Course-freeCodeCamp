from typing import TypedDict,Sequence,Annotated
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage

from dotenv import load_dotenv
from langchain_core.tools import tool

from langgraph.graph.message import add_messages  # is a reduce function that changes the behavior when updating the state
from langgraph.graph import  StateGraph, START, END 
from langgraph.prebuilt import ToolNode



load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def add(a: int, b: int) -> int:
    """
    This function add two numbers
    """
    return a+b

@tool
def multiply(a: int, b: int) -> int:
    """
    This function multiply two given values
    """
    return a*b

tools = [add,multiply]

llm = ChatDeepSeek(model="deepseek-v4-pro").bind_tools(tools=tools)



def model_call(state: AgentState) -> AgentState:
    System_prompt = SystemMessage(content="You are a calculator, randomly add a joke to your answer")
    query_result = llm.invoke([System_prompt]+state["messages"])

    return {"messages":[query_result]}


def should_continue(state: AgentState) -> AgentState:
    msj = state["messages"]
    last_message = msj[-1]

    if not last_message.tool_calls:
        return "end" 
    else:
        return "continue"



graph = StateGraph(AgentState)

graph.add_node("model",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.add_edge(START,"model")
graph.add_conditional_edges(
    "model",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
    )


graph.add_edge("tools","model")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()



inputs = {"messages": [("user", "Add 40 + 12 , then multiply this value *3 ")]}
print_stream(app.stream(inputs, stream_mode="values"))
