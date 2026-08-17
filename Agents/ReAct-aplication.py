from typing import Annotated, Sequence, TypedDict #Anotated add metadata for a specific field, Sequence Avoid list manipulation directly
from dotenv import load_dotenv  


from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage # The foundational class for all message types in LangGraph
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek


from langgraph.graph.message import add_messages  # is a recude function that changes the behavior when updating the state
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a:int, b:int)  ->int:
    """
    This is a function that sums that adds two number together 
    """

    return a+b 



tools = [add]

llm = ChatDeepSeek(model = "deepseek-v4-pro").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    query_result = llm.invoke([system_prompt] + state["messages"])

    return {"messages":[query_result]}

def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)

graph.add_node("agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool",tool_node)


graph.add_edge(START,"agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool",
        "end": END
    }
)

graph.add_edge("tool","agent")


app = graph.compile()



def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 .")]}
print_stream(app.stream(inputs, stream_mode="values"))
